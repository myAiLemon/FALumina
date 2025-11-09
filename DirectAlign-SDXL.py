#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SDXL LoRA 训练脚本，使用 Deepspeed 和 Direct-Align
"""

import tomllib
import math
import random
import sys
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import (
    UNet2DConditionModel, AutoencoderKL,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image

# Deepspeed 导入
import deepspeed
import json # 用于解析 deepspeed config

# ------------------------------------------------------------------ #
# 0. 日志记录器设置
# ------------------------------------------------------------------ #
# 使用标准日志记录
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# 1. ARB 辅助类
# ------------------------------------------------------------------ #
class AspectRatioBucketManager:
    """管理宽高比分桶"""
    def __init__(self, target_resolution=1024, min_dim=512, max_dim=2048, step_size=64):
        self.target_area = target_resolution * target_resolution
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.step_size = step_size
        self.buckets = {}  # { (h, w): [image_path, ...] }
        self.image_to_bucket = {}  # { image_path: (h, w) }

    def _get_closest_resolution(self, height, width):
        """计算最接近的有效分桶分辨率"""
        aspect_ratio = height / width
        new_w = math.sqrt(self.target_area / aspect_ratio)
        new_h = new_w * aspect_ratio
        new_w = round(new_w / self.step_size) * self.step_size
        new_h = round(new_h / self.step_size) * self.step_size
        new_w = max(self.min_dim, min(self.max_dim, int(new_w)))
        new_h = max(self.min_dim, min(self.max_dim, int(new_h)))
        return (new_h, new_w)

    def add_image(self, image_path, height, width):
        """添加图片到分桶"""
        bucket_res = self._get_closest_resolution(height, width)
        if bucket_res not in self.buckets:
            self.buckets[bucket_res] = []
        self.buckets[bucket_res].append(image_path)
        self.image_to_bucket[image_path] = bucket_res

    def get_bucket_indices(self):
        """返回每个分桶内的图片索引列表，用于 BatchSampler"""
        bucket_indices = {}  # { (h, w): [idx1, idx2, ...] }
        image_path_to_index = {path: i for i, path in enumerate(self.image_to_bucket.keys())}
        for res, paths in self.buckets.items():
            bucket_indices[res] = [image_path_to_index[p] for p in paths]
        return bucket_indices

    def get_image_path_list(self):
        return list(self.image_to_bucket.keys())


class BucketSampler(Sampler):
    """
    自定义批次采样器，实现 ARB，并兼容 Deepspeed 分布式训练
    """
    def __init__(self, bucket_indices, batch_size, shuffle=True, num_replicas=1, rank=0):
        super().__init__(data_source=None) # 兼容 Sampler 基类
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        # 展平所有分桶的索引列表
        self.flat_batches = []
        for res, indices in self.bucket_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:  # 丢弃不完整的批次
                    self.flat_batches.append(batch)
        
        # 【Deepspeed 兼容】
        # 每个 rank 只获取自己那部分的批次
        self.flat_batches = self.flat_batches[self.rank::self.num_replicas]
        self.num_batches = len(self.flat_batches)

    def __iter__(self):
        # 在每个 epoch 开始时重新打乱批次顺序
        if self.shuffle:
            # 确定一个固定的、基于 epoch 的种子，确保所有 rank 上的 shuffle 顺序一致
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.flat_batches), generator=g).tolist()
            batches_this_epoch = [self.flat_batches[i] for i in indices]
        else:
            batches_this_epoch = self.flat_batches
            
        return iter(batches_this_epoch)

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch: int):
        """
        DistributedSampler 需要此方法来确保 shuffle 在不同 epoch 间不同
        """
        self.epoch = epoch


class SDXLDataCollator(object):
    """
    自定义数据整理器
    """
    def __init__(self, tokenizer_one, tokenizer_two):
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two

    def __call__(self, batch):
        pixel_values_list = []
        original_sizes = []
        crop_top_lefts = []
        texts = []
        
        target_height, target_width = batch[0]["target_size"]
        
        for item in batch:
            texts.append(item["text"])
            img = Image.open(item["image_path"]).convert("RGB")
            original_sizes.append((img.height, img.width))
            
            img_resized = transforms.functional.resize(
                img, 
                max(target_height, target_width),
                interpolation=InterpolationMode.BILINEAR
            )
            
            top, left, height, width = transforms.RandomCrop.get_params(
                img_resized, output_size=(target_height, target_width)
            )
            img_cropped = transforms.functional.crop(img_resized, top, left, height, width)
            crop_top_lefts.append((top, left))
            
            img_tensor = transforms.functional.normalize(
                transforms.functional.to_tensor(img_cropped),
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            )
            pixel_values_list.append(img_tensor)

        inputs_one = self.tokenizer_one(
            texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        )
        inputs_two = self.tokenizer_two(
            texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        )
        
        pixel_values = torch.stack(pixel_values_list)
        original_sizes = torch.tensor(original_sizes, dtype=torch.long)
        crop_top_lefts = torch.tensor(crop_top_lefts, dtype=torch.long)
        target_sizes = torch.tensor([[target_height, target_width]] * len(batch), dtype=torch.long)
        
        add_time_ids = torch.cat(
            [original_sizes, crop_top_lefts, target_sizes], 
            dim=1
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids_one": inputs_one.input_ids,
            "input_ids_two": inputs_two.input_ids,
            "add_time_ids": add_time_ids,
            "texts_for_logging": texts
        }

# ------------------------------------------------------------------ #
# 2. TOML 配置加载
# ------------------------------------------------------------------ #
def load_config(toml_path):
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)
    
    # 添加Direct-Align所需的配置项 (重命名以避免与deepspeed config冲突)
    cfg.setdefault("direct_align", {})
    cfg["direct_align"].setdefault("timestep_min_ratio", 0.2)
    cfg["direct_align"].setdefault("timestep_max_ratio", 0.8)
    cfg["direct_align"].setdefault("prior_strength", 1.0)
    
    # 确保基础配置存在
    cfg.setdefault("mixed_precision", "bf16")
    cfg.setdefault("output_dir", "./output")
    cfg.setdefault("model_path", "stabilityai/stable-diffusion-xl-base-1.0")
    cfg.setdefault("lr", 1e-5)
    cfg.setdefault("batch_size", 2) # 这是 per-GPU-micro-batch-size
    cfg.setdefault("resolution", 1024)
    cfg.setdefault("max_train_steps", 10000)
    cfg.setdefault("lr_scheduler", "cosine")
    cfg.setdefault("dataloader_num_workers", 4)
    cfg.setdefault("dataset_folder", "./your_dataset")
    cfg.setdefault("lora_rank", 32)
    cfg.setdefault("lora_alpha", 32)
    cfg.setdefault("num_warmup_steps", 0)
    cfg.setdefault("model_save_steps", 200)
    cfg.setdefault("arb_min_dim", 512)
    cfg.setdefault("arb_max_dim", 2048)
    cfg.setdefault("arb_step_size", 64)
    
    return cfg


# ------------------------------------------------------------------ #
# 3. Direct-Align 核心函数
# ------------------------------------------------------------------ #
def get_alphas_sigmas_for_timesteps(scheduler, timesteps, device, dtype=torch.float32):
    """为批次中的每个时间步计算alpha和sigma"""
    alphas_cumprod = scheduler.alphas_cumprod.clone().detach().to(device=device, dtype=dtype)
    alpha_t = alphas_cumprod[timesteps]
    sigma_t = torch.sqrt(1.0 - alpha_t)
    alpha_t = alpha_t.view(-1, 1, 1, 1)
    sigma_t = sigma_t.view(-1, 1, 1, 1)
    return alpha_t, sigma_t

def sample_direct_align_timesteps(scheduler, batch_size, device, cfg_dict):
    """采样Direct-Align策略的时间步"""
    num_timesteps = scheduler.config.num_train_timesteps
    min_t = int(num_timesteps * cfg_dict["direct_align"]["timestep_min_ratio"])
    max_t = int(num_timesteps * cfg_dict["direct_align"]["timestep_max_ratio"])
    
    return torch.randint(
        min_t, 
        max_t + 1, 
        (batch_size,), 
        device=device
    ).long()

def construct_direct_align_target(latents, scheduler, timesteps, cfg_dict):
    """构造Direct-Align的目标epsilon"""
    device = latents.device
    dtype = latents.dtype
    
    alpha_t, sigma_t = get_alphas_sigmas_for_timesteps(
        scheduler, 
        timesteps, 
        device, 
        dtype=dtype
    )
    
    noise_prior = torch.randn_like(latents) * float(cfg_dict["direct_align"]["prior_strength"])
    x_t = torch.sqrt(alpha_t) * latents + sigma_t * noise_prior
    epsilon_target = noise_prior
    
    return x_t, epsilon_target, alpha_t, sigma_t


# ------------------------------------------------------------------ #
# 4. 数据集加载
# ------------------------------------------------------------------ #
def get_dataset_records(cfg):
    """加载本地数据集（图片+同目录同名txt描述）"""
    dataset_folder = Path(cfg["dataset_folder"])
    if not dataset_folder.exists():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_folder}")

    img_extensions = ["*.png", "*.jpg", "*.jpeg"]
    img_files = []
    for ext in img_extensions:
        for img_path in dataset_folder.glob(f"**/{ext}"):
            if "checkpoint" in img_path.name.lower():
                continue
            img_files.append(img_path)

    img_files = list(set(img_files))
    img_files.sort()

    if not img_files:
        raise ValueError(f"未在{dataset_folder}找到有效图片文件")

    records = []
    for img_path in img_files:
        img_path = Path(img_path)
        txt_path = img_path.with_suffix(".txt")
        caption = ""
        if txt_path.exists():
            try:
                caption = txt_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(f"读取文本失败 {txt_path}: {e}")
        
        records.append({
            "image_path": str(img_path),
            "text": caption
        })

    logger.info(f"数据集加载完成：共{len(records)}个样本")
    return records


# ------------------------------------------------------------------ #
# 5. 主函数 (已修复)
# ------------------------------------------------------------------ #
def main():
    # --- 5.1 参数解析 ---
    parser = argparse.ArgumentParser(
        description="使用 Deepspeed 和 Direct-Align 训练 SDXL LoRA"
    )
    parser.add_argument(
        "toml_path",
        type=str,
        help="配置文件 (.toml) 的路径"
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        required=True,
        help="Deepspeed config JSON 文件的路径"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Deepspeed 注入的 local rank"
    )
    args = parser.parse_args()

    cfg = load_config(toml_path=args.toml_path)

    # --- 5.2 Deepspeed 初始化 ---
    deepspeed.init_distributed()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    
    # 获取分布式训练信息
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    is_main_process = (rank == 0)

    if is_main_process:
        logger.info(f"Deepspeed 已初始化。World size: {world_size}, Rank: {rank}")
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)


    # --- 5.3 文本编码器和分词器 ---
    tokenizer_one = CLIPTokenizer.from_pretrained(
        cfg["model_path"], subfolder="tokenizer"
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        cfg["model_path"], subfolder="tokenizer_2"
    )

    text_encoder_one = CLIPTextModel.from_pretrained(
        cfg["model_path"], subfolder="text_encoder"
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        cfg["model_path"], subfolder="text_encoder_2"
    )

    # --- 5.4 VAE & UNet ---
    vae = AutoencoderKL.from_pretrained(
        cfg["model_path"], subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg["model_path"],
        subfolder="unet",
        use_flash_attention_2=True,
    )
    
    # 冻结 VAE 和文本编码器，它们将作为推理组件
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    # 注意：此处不再调用 vae.to(device) 等，因为将在 Deepspeed 初始化前统一处理 dtye/device 转换


    # --- 5.5 LoRA 配置 ---
    unet_lora_config = LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet.enable_gradient_checkpointing()
    # unet = prepare_model_for_kbit_training(unet) # 如果使用8bit/4bit，则启用此行
    unet = get_peft_model(unet, unet_lora_config)
    
    if is_main_process:
        unet.print_trainable_parameters()

    # --- 5.6 优化器 & LR 调度器 ---
    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg["lr"])
    
    lr_scheduler = get_scheduler(
        cfg["lr_scheduler"], 
        optimizer=optimizer,
        num_warmup_steps=cfg["num_warmup_steps"],
        num_training_steps=cfg["max_train_steps"],
    )

    # --- 5.7 数据集 & ARB (Deepspeed 适配) ---
    if is_main_process:
        logger.info("正在加载数据集记录...")
    records = get_dataset_records(cfg)
    
    if is_main_process:
        logger.info("正在初始化 ARB 分桶...")
    bucket_manager = AspectRatioBucketManager(
        target_resolution=cfg["resolution"],
        min_dim=cfg["arb_min_dim"],
        max_dim=cfg["arb_max_dim"],
        step_size=cfg["arb_step_size"]
    )
    
    if is_main_process:
        logger.info("正在预缓存图像尺寸并分配分桶...")
    for record in records:
        img_path = record["image_path"]
        try:
            with Image.open(img_path) as img:
                width, height = img.size
            bucket_manager.add_image(img_path, height, width)
        except Exception as e:
            logger.error(f"处理图片 {img_path} 失败：{e}", exc_info=True) 
            
    bucket_indices = bucket_manager.get_bucket_indices()
    
    image_path_list = bucket_manager.get_image_path_list()
    record_map = {r["image_path"]: r["text"] for r in records}
    
    final_records = []
    for img_path in image_path_list:
        if img_path in record_map:
            final_records.append({
                "image_path": img_path,
                "text": record_map[img_path],
                "target_size": bucket_manager.image_to_bucket[img_path]
            })

    if is_main_process:
        logger.info(f"已创建 {len(bucket_manager.buckets)} 个分桶。")
    
    train_dataset = Dataset.from_list(final_records)
    
    train_sampler = BucketSampler(
        bucket_indices, 
        cfg["batch_size"], 
        shuffle=True, 
        num_replicas=world_size, 
        rank=rank
    )
    
    train_collator = SDXLDataCollator(tokenizer_one, tokenizer_two)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,  # 使用自定义 Batch Sampler
        collate_fn=train_collator,
        num_workers=cfg["dataloader_num_workers"],
        pin_memory=True, # 提升性能
    )

    # --- 5.8 噪声调度器和管道 (用于生成样本) ---
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg["model_path"], subfolder="scheduler"
    )
    noise_scheduler.register_to_config(prediction_type="epsilon")

    # --- 5.9 Deepspeed 模型封装 (关键修复部分) ---
    # 获取 Deepspeed 自动设置的混合精度类型
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    
    # 解析 bf16.enabled 和 fp16.enabled 的值
    bf16_enabled = ds_config.get("bf16", {}).get("enabled", False)
    fp16_enabled = ds_config.get("fp16", {}).get("enabled", False)

    # 根据 bf16/fp16 开关设置数据类型
    if bf16_enabled:
        ds_dtype = torch.bfloat16
    elif fp16_enabled:
        ds_dtype = torch.float16
    else:
        ds_dtype = torch.float32 
    
    # FIX: 核心修复：将所有模型（训练的UNet，冻结的VAE/Text Encoders）
    # 移动到GPU设备，并转换为Deepspeed确定的低精度数据类型（ds_dtype）。
    # 这是解决 RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same 的关键。
    unet = unet.to(device).to(ds_dtype)
    vae = vae.to(device).to(ds_dtype)
    text_encoder_one = text_encoder_one.to(device).to(ds_dtype)
    text_encoder_two = text_encoder_two.to(device).to(ds_dtype)
    
    if is_main_process:
        # 仅在主进程中设置管道，用于保存样本
        from diffusers import StableDiffusionXLPipeline
        pipeline = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            unet=unet, # 临时
            scheduler=noise_scheduler,
        ).to(device)


    # Deepspeed 会接管 UNet, optimizer, 和 lr_scheduler
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=unet,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=args.deepspeed_config
    )

    # --- 5.10 训练循环 ---
    global_step = 0
    if is_main_process:
        logger.info("=====> 开始 Deepspeed 训练 <=====")
        logger.info(f"  总步数 = {cfg['max_train_steps']}")
        logger.info(f"  每个 Rank 的批次数 = {cfg['batch_size']}")
        logger.info(f"  Deepspeed Dtype = {ds_dtype}")
        
    # 用于监控Direct-Align效果的指标
    direct_align_metrics = {
        "recovery_l2": [],
        "sigma_values": []
    }

    
    while global_step < cfg["max_train_steps"]:
        
        # 为分布式 Sampler 设置 epoch
        train_sampler.set_epoch(global_step // len(train_dataloader))
        
        for step, batch in enumerate(train_dataloader):
            
            # --- 训练步骤 ---
            # 1. 图像编码到latent空间
            # 确保 pixel_values 的 Dtype 与 VAE 模型一致 (ds_dtype)
            pixel_values = batch["pixel_values"].to(device).to(ds_dtype)
            
            # 由于 VAE 现在已是 ds_dtype，它将接受并输出对应 dtype 的 Latent
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents = latents.to(ds_dtype) # 确保 Latent 也是 ds_dtype

            # 2. Direct-Align: 采样时间步并构造目标
            bsz = latents.shape[0]
            timesteps = sample_direct_align_timesteps(
                noise_scheduler, bsz, device=device, cfg_dict=cfg
            )
            noisy_latents, target, alpha_t, sigma_t = construct_direct_align_target(
                latents, noise_scheduler, timesteps, cfg_dict=cfg
            )
            
            # 3. 计算文本嵌入 (Text Encoders 也在 ds_dtype 上)
            inputs_one = {"input_ids": batch["input_ids_one"].to(device)}
            inputs_two = {"input_ids": batch["input_ids_two"].to(device)}

            encoder_hidden_states_one = text_encoder_one(**inputs_one, return_dict=True).last_hidden_state
            outputs_two = text_encoder_two(**inputs_two, return_dict=True)
            encoder_hidden_states_two = outputs_two.last_hidden_state
            pooled_prompt_embeds = outputs_two.text_embeds
            
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states_one, encoder_hidden_states_two], 
                dim=-1
            )
            
            add_time_ids = batch["add_time_ids"].to(device)
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids
            }
            
            # 4. 模型预测
            model_pred = model_engine(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=True
            ).sample
            
            # 5. 计算损失 (使用 .float() 进行损失计算，以保证稳定性)
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # 6. Deepspeed 反向传播和优化
            model_engine.backward(loss)
            model_engine.step() # 自动处理 grad accum, optim.step, lr.step, zero_grad

            global_step += 1
            
            # --- 日志记录和保存 (仅主进程) ---
            if is_main_process:
                # 计算 L2 恢复指标
                with torch.no_grad():
                    # 评估时也使用 .float()
                    x0_recovered = (noisy_latents.float() - sigma_t.float() * model_pred.float()) / torch.sqrt(alpha_t.float())
                    recovery_l2 = torch.nn.functional.mse_loss(x0_recovered, latents.float()).item()
                direct_align_metrics["recovery_l2"].append(recovery_l2)
                direct_align_metrics["sigma_values"].append(float(sigma_t.mean().cpu()))

                # 日志输出
                if global_step % 50 == 0:
                    recent_recovery = direct_align_metrics["recovery_l2"][-10:]
                    avg_recovery = sum(recent_recovery) / len(recent_recovery) if recent_recovery else 0
                    logger.info(
                        f"step {global_step}/{cfg['max_train_steps']} | "
                        f"loss={loss.item():.4f} | "
                        f"avg_recovery_l2={avg_recovery:.6f} | "
                        f"sigma_mean={direct_align_metrics['sigma_values'][-1]:.4f} | "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    )

                # 保存检查点和样本
                if global_step % cfg["model_save_steps"] == 0 or global_step == cfg["max_train_steps"]:
                    save_path = Path(cfg["output_dir"]) / f"checkpoint-{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    # 1. 保存 LoRA 权重 (用于推理)
                    # 【关键】从 Deepspeed engine 中解包模型
                    unwrapped_unet = model_engine.module
                    unwrapped_unet.save_pretrained(save_path / "unet_lora")
                    
                    # 2. 生成样本
                    # with torch.no_grad():
                    #     # 更新 pipeline 中的 UNet
                    #     pipeline.unet = unwrapped_unet 
                        
                    #     texts = batch["texts_for_logging"]
                    #     sample_prompt = texts[0] if texts else "1girl"
                        
                    #     sample_h, sample_w = batch["pixel_values"].shape[-2:]
                    #     sample_orig_size = batch["add_time_ids"][0, 0:2].tolist()
                    #     sample_crop_coords = batch["add_time_ids"][0, 2:4].tolist()

                    #     image = pipeline(
                    #         prompt=sample_prompt,
                    #         num_inference_steps=20,
                    #         guidance_scale=5.0,
                    #         height=sample_h,
                    #         width=sample_w,
                    #         original_size=sample_orig_size,
                    #         crops_coords_top_left=sample_crop_coords
                    #     ).images[0]
                    # image.save(save_path / "sample.jpg")
                    
                    # 3. 保存指标
                    with open(save_path / "direct_align_metrics.txt", "w") as f:
                        avg_total_recovery = sum(direct_align_metrics["recovery_l2"]) / len(direct_align_metrics["recovery_l2"]) if direct_align_metrics["recovery_l2"] else 0
                        f.write(f"Average recovery L2: {avg_total_recovery:.6f}\n")
                        
                    # 4. 可选：保存 Deepspeed 检查点
                    # model_engine.save_checkpoint(save_path) 

            # 达到最大步数则停止
            if global_step >= cfg["max_train_steps"]:
                break
        
        if global_step >= cfg["max_train_steps"]:
            break

    logger.info(f"Rank {rank} 训练完成!")
    # 等待所有进程完成
    torch.distributed.barrier()


if __name__ == "__main__":
    main()