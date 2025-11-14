import os
import argparse
import math
import logging
import json
import warnings

# 尝试导入 Py3.11+ 的 tomllib，否则回退到 tomli
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import deepspeed
from diffusers import AutoencoderKL
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from torchvision import transforms
import lpips
from glob import glob

# --- 日志设置 ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 忽略 PIL 可能的 DecompressionBombWarning
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# === 1. 数据集 (高分辨率图块) ===
class HighResPatchDataset(Dataset):
    """
    从高分辨率图像目录加载图像，并随机裁剪出指定分辨率的图块。
    """
    def __init__(self, image_dir, resolution):
        self.image_paths = glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)
        self.image_paths.extend(glob(os.path.join(image_dir, "**", "*.png"), recursive=True))
        self.image_paths.extend(glob(os.path.join(image_dir, "**", "*.webp"), recursive=True))
        
        if not self.image_paths:
            logger.error(f"在 {image_dir} 中未找到任何图像。请检查 'data.image_dir' 配置。")
            raise FileNotFoundError(f"在 {image_dir} 中未找到图像")
            
        logger.info(f"找到了 {len(self.image_paths)} 张图像。")
        
        # 定义图块变换
        self.transform = transforms.Compose([
            # 随机裁剪指定分辨率的图块
            transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # VAE需要输入范围在 [-1, 1]
            transforms.Normalize([0.5], [0.5]),
        ])
        self.resolution = resolution

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            # 加载高分辨率图像 (例如 1536p 或 2048p)
            image = Image.open(image_path).convert("RGB")
            
            # 确保图像至少和裁剪分辨率一样大
            if image.width < self.resolution or image.height < self.resolution:
                logger.warning(f"图像 {image_path} ({image.size}) 小于裁剪分辨率 {self.resolution}。正在跳过。")
                return self.__getitem__((idx + 1) % len(self))

            # 应用变换（裁剪、翻转、ToTensor、标准化）
            patch = self.transform(image)
            return patch
        except Exception as e:
            logger.warning(f"加载图像失败 {image_path}: {e}. 跳过。")
            # 如果失败，加载下一张
            return self.__getitem__((idx + 1) % len(self))

# === 2. 损失函数 (核心策略) ===

class MultiScaleL1Loss(nn.Module):
    """
    计算多尺度L1损失。
    """
    def __init__(self, scales=3, downsample_mode='bilinear'):
        super().__init__()
        self.scales = scales
        self.downsample_mode = downsample_mode
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        total_loss = 0.0
        for i in range(self.scales):
            total_loss += self.loss(pred, target)
            if i < self.scales - 1:
                # 下采样
                pred = F.interpolate(pred, scale_factor=0.5, mode=self.downsample_mode, align_corners=False)
                target = F.interpolate(target, scale_factor=0.5, mode=self.downsample_mode, align_corners=False)
        return total_loss

class CombinedVAELoss(nn.Module):
    """
    组合 L1/MultiScale L1 和 LPIPS 感知损失。
    """
    def __init__(self, loss_config, device):
        super().__init__()
        self.cfg = loss_config
        self.device = device
        
        self.l1_weight = self.cfg.get('l1_weight', 0.0)
        self.use_l1 = self.cfg.get('use_l1_loss', False) and self.l1_weight > 0
        self.use_multi_scale = self.cfg.get('use_multi_scale_loss', False)
        
        if self.use_multi_scale:
            logger.info(f"使用 Multi-Scale L1 损失，权重: {self.l1_weight}")
            self.l1_loss_fn = MultiScaleL1Loss().to(self.device)
            # 确保 L1 不会被重复计算
            self.use_l1 = False 
        elif self.use_l1:
            logger.info(f"使用 Standard L1 损失，权重: {self.l1_weight}")
            self.l1_loss_fn = nn.L1Loss().to(self.device)
        
        self.use_perceptual = self.cfg.get('use_perceptual_loss', False)
        self.perceptual_weight = self.cfg.get('perceptual_weight', 0.0)
        
        if self.use_perceptual and self.perceptual_weight > 0:
            logger.info(f"使用 LPIPS 感知损失 (AlexNet)，权重: {self.perceptual_weight}")
            try:
                # 'alex' net, v0.1 在 ImageNet 上训练, 期望输入在 [-1, 1]
                self.perceptual_loss_fn = lpips.LPIPS(net='alex', version='0.1').to(self.device)
                # 冻结 LPIPS 模型的参数
                self.perceptual_loss_fn.eval()
                for param in self.perceptual_loss_fn.parameters():
                    param.requires_grad = False
            except Exception as e:
                logger.error(f"初始化 LPIPS 失败: {e}")
                logger.error("这可能是因为无法下载 LPIPS 预训练权重。请检查网络连接。")
                raise e
        else:
            self.use_perceptual = False
            self.perceptual_loss_fn = None

    def forward(self, pred, target):
        total_loss = 0.0
        loss_dict = {}

        # 确保计算在 float32 上进行以保证精度
        pred_f32 = pred.float()
        target_f32 = target.float()

        # 1. 计算 L1 或 Multi-Scale L1
        if self.use_multi_scale or self.use_l1:
            l1_loss = self.l1_loss_fn(pred_f32, target_f32)
            total_loss += self.l1_weight * l1_loss
            loss_name = 'multi_scale_l1_loss' if self.use_multi_scale else 'l1_loss'
            loss_dict[loss_name] = l1_loss.item()

        # 2. 计算 LPIPS
        if self.use_perceptual and self.perceptual_loss_fn:
            # LPIPS 期望 [-1, 1] 范围，我们的 VAE 输出已在此范围
            p_loss = self.perceptual_loss_fn(pred_f32, target_f32).mean()
            total_loss += self.perceptual_weight * p_loss
            loss_dict['perceptual_loss'] = p_loss.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

# === 3. 主训练函数 ===

def main():
    parser = argparse.ArgumentParser(description="DeepSpeed VAE LoRA Finetuning Script")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="config.toml", 
        help="Path to the TOML configuration file."
    )
    
    # 手动添加 deepspeed_config 参数而不是使用 deepspeed.add_config_arguments
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="ds_config.json",  # 添加默认值
        help="Path to the DeepSpeed configuration file."
    )
    # 添加这个关键参数，DeepSpeed 会自动传入
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank. Necessary for DeepSpeed to function properly."
    )
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    logger.info(f"加载配置文件: {args.config_path}")
    try:
        with open(args.config_path, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        logger.error(f"配置文件 {args.config_path} 未找到！")
        raise
    
    # 检查 DeepSpeed 配置文件是否存在
    if not os.path.exists(args.deepspeed_config):
        logger.error(f"DeepSpeed 配置文件 {args.deepspeed_config} 不存在！")
        logger.error(f"请确保文件 {args.deepspeed_config} 存在")
        raise FileNotFoundError(f"DeepSpeed 配置文件 {args.deepspeed_config} 不存在！")
    
    logger.info(f"加载 DeepSpeed 配置: {args.deepspeed_config}")
    try:
        with open(args.deepspeed_config, "r") as f:
            ds_config = json.load(f)
    except Exception as e:
        logger.error(f"解析 DeepSpeed 配置文件失败: {e}")
        raise

    # --- 2. 初始化 DeepSpeed ---
    deepspeed.init_distributed()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    
    if local_rank == 0:
        logger.info(f"DeepSpeed 初始化完成。 World Size: {world_size}")
        os.makedirs(config['model']['save_path'], exist_ok=True)
    
    # --- 3. 准备数据 ---
    try:
        dataset = HighResPatchDataset(
            image_dir=config['data']['image_dir'],
            resolution=config['data']['patch_resolution']
        )
    except FileNotFoundError:
        logger.error("数据加载失败，请检查 config.toml 中的 'data.image_dir'。")
        return # 退出

    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    train_loader = DataLoader(
        dataset,
        batch_size=config['training']['micro_batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # --- 4. 准备模型 (VAE + LoRA on Decoder) ---
    logger.info(f"正在从 {config['model']['vae_path']} 加载 VAE...")
    try:
        # 强制 VAE 使用 float32 加载，DeepSpeed 稍后会处理混合精度
        vae = AutoencoderKL.from_single_file(config['model']['vae_path'], torch_dtype=torch.float32)
    except Exception as e:
        logger.error(f"加载 VAE 失败: {e}. 请检查 'model.vae_path'。")
        raise e
        
    # 核心思路 1：冻结编码器 (Freeze Encoder)
    vae.encoder.requires_grad_(False)
    logger.info("VAE 编码器已冻结。")

    # 核心思路 1 (续)：为解码器添加 LoRA
    # 我们只在解码器(vae.decoder)上应用LoRA
    
    # 查找解码器中所有的 Conv2d 模块
    target_modules_list = []
    for name, module in vae.decoder.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # 确保我们只添加叶子模块
            if '.' not in name.split('.')[-1]:
                 target_modules_list.append(name)
            
    if local_rank == 0:
        logger.info(f"在 VAE 解码器中找到 {len(target_modules_list)} 个 Conv2d 模块用于 LoRA。")
        # logger.debug(f"Target modules: {target_modules_list}")

    lora_config = LoraConfig(
        r=config['model']['lora_rank'],
        lora_alpha=config['model']['lora_rank'],
        target_modules=target_modules_list,
        lora_dropout=0.1,
        bias="none",
    )
    
    # 关键：只将 peft 应用于 vae.decoder
    try:
        vae.decoder = get_peft_model(vae.decoder, lora_config)
    except Exception as e:
        logger.error(f"应用 LoRA 到 VAE 解码器失败: {e}")
        logger.error("这可能是因为 target_modules 列表为空或不正确。")
        raise e
    
    if local_rank == 0:
        logger.info("已将 LoRA 应用于 VAE 解码器。")
        vae.decoder.print_trainable_parameters()

    # 获取所有可训练参数（即 LoRA 权重）
    trainable_params = [p for p in vae.parameters() if p.requires_grad]
    logger.info(f"总可训练参数量: {sum(p.numel() for p in trainable_params)}")

    # --- 5. 准备损失函数 ---
    criterion = CombinedVAELoss(config['loss'], device=torch.device(f"cuda:{local_rank}"))
    
    # --- 6. 更新并验证 DeepSpeed 配置 ---
    
    # 根据 TOML 配置动态更新 DeepSpeed JSON
    train_dtype = torch.bfloat16
    # if config['training']['mixed_precision'] == 'bf16':
    #     ds_config['bf16']['enabled'] = true
    #     ds_config['fp16']['enabled'] = false # 确保 fp16 关闭
    #     train_dtype = torch.bfloat16
    #     logger.info("启用 BF16 混合精度。")

    # 计算总步数
    steps_per_epoch = math.ceil(len(dataset) / (config['training']['micro_batch_size'] * world_size * config['training']['gradient_accumulation_steps']))
    total_training_steps = steps_per_epoch * config['training']['epochs']
    
    # 更新 ds_config
    ds_config['train_micro_batch_size_per_gpu'] = config['training']['micro_batch_size']
    ds_config['gradient_accumulation_steps'] = config['training']['gradient_accumulation_steps']
    ds_config['optimizer']['params']['lr'] = config['training']['learning_rate']
    ds_config['scheduler']['params']['warmup_max_lr'] = config['training']['learning_rate']
    ds_config['scheduler']['params']['warmup_num_steps'] = config['training']['warmup_steps']
    ds_config['scheduler']['params']['total_num_steps'] = total_training_steps
    
    if local_rank == 0:
        logger.info(f"每个 Epoch 步数: {steps_per_epoch}")
        logger.info(f"总训练步数: {total_training_steps} ({config['training']['epochs']} epochs)")

    # --- 7. 初始化 DeepSpeed 引擎 ---
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=vae,
        model_parameters=trainable_params,
        config=ds_config
    )
    
    # --- 8. 训练循环 ---
    global_step = 0
    logger.info("***** 开始训练 *****")

    for epoch in range(config['training']['epochs']):
        train_sampler.set_epoch(epoch) # 确保分布式采样器在每 epoch shuffle
        
        for step, batch in enumerate(train_loader):
            try:
                # 将图块数据移至 GPU
                images = batch.to(model_engine.device)
                
                # 1. 编码 (在 no_grad 和 混合精度下)
                # model_engine.module 访问被包装的原始 vae 模型
                with torch.no_grad():
                    # 编码器是冻结的，不需要梯度
                    latents = model_engine.module.encode(images.to(train_dtype)).latent_dist.sample()
                    latents = latents * model_engine.module.config.scaling_factor
                
                # 2. 解码 (需要梯度，在混合精度下)
                # LoRA 权重 (在 vae.decoder 中) 需要梯度
                reconstructions = model_engine.module.decode(latents.to(train_dtype)).sample
                
                # 3. 计算损失
                # 损失函数内部会将输入转为 float32 计算
                loss, loss_dict = criterion(reconstructions, images)
                
                # 4. DeepSpeed 反向传播
                model_engine.backward(loss)
                
                # 5. DeepSpeed 优化器步骤
                model_engine.step()
                
                global_step += 1
                
                # --- 日志和保存 ---
                if global_step % config['training']['log_steps'] == 0 and local_rank == 0:
                    loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                    logger.info(f"[Epoch {epoch+1}/{config['training']['epochs']}] [Step {global_step}] [LR: {lr_scheduler.get_last_lr()[0]:.2e}] {loss_str}")

                if global_step % config['model']['save_steps'] == 0 and local_rank == 0:
                    save_dir = os.path.join(config['model']['save_path'], f"checkpoint-{global_step}")
                    logger.info(f"保存 LoRA 权重到: {save_dir}")
                    
                    # model_engine.module 是原始的 vae (PeftModel)
                    # vae.decoder 是 PeftModel
                    model_engine.module.decoder.save_pretrained(save_dir)

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"在 global_step {global_step} 发生 CUDA OOM 错误: {e}")
                logger.error("显存不足！请尝试在 config.toml 中减小 'micro_batch_size'。")
                # 在 OOM 时无法恢复，必须退出
                return
            except Exception as e:
                logger.error(f"在训练循环中发生未知错误: {e}", exc_info=True)
                # 跳过这个 batch
                continue


    if local_rank == 0:
        logger.info("训练完成。")
        final_save_dir = os.path.join(config['model']['save_path'], "final")
        model_engine.module.decoder.save_pretrained(final_save_dir)
        logger.info(f"最终 LoRA 权重已保存到: {final_save_dir}")

if __name__ == "__main__":
    main()