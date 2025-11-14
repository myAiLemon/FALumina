import os
import torch
import argparse
from diffusers import AutoencoderKL
from peft import PeftModel
import logging

# --- 日志设置 ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def merge_lora_to_vae(vae_path, lora_path, output_path, output_dtype=torch.float32, 
                     device="cpu", save_safetensors=True):
    """
    将训练好的 LoRA 权重合并到原始 VAE 模型中
    
    Args:
        vae_path: 原始 VAE 模型路径 (.bin 或 .safetensors)
        lora_path: 训练好的 LoRA 权重路径
        output_path: 合并后的模型保存路径
        output_dtype: 输出模型的数据类型 (float32, float16, bfloat16)
        device: 处理时使用的设备 (cpu, cuda)
        save_safetensors: 是否保存为 safetensors 格式 (True: safetensors, False: .bin)
    """
    logger.info(f"开始合并 LoRA 权重到 VAE...")
    logger.info(f"原始 VAE 路径: {vae_path}")
    logger.info(f"LoRA 权重路径: {lora_path}")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"输出数据类型: {output_dtype}")
    logger.info(f"处理设备: {device}")
    logger.info(f"保存格式: {'safetensors' if save_safetensors else 'bin'}")
    
    try:
        # 1. 确定输入文件格式
        use_safetensors = vae_path.endswith('.safetensors')
        logger.info(f"输入文件格式: {'safetensors' if use_safetensors else 'bin'}")
        
        # 2. 加载原始 VAE 模型
        logger.info("正在加载原始 VAE 模型...")
        
        # 先加载到 CPU，避免显存问题
        vae = AutoencoderKL.from_single_file(
            vae_path, 
            torch_dtype=torch.float32,  # 先用 float32 加载确保精度
            use_safetensors=use_safetensors  # 根据文件扩展名选择加载方式
        )
        logger.info("原始 VAE 模型加载成功")
        
        # 3. 加载 LoRA 权重并应用到解码器
        logger.info("正在加载 LoRA 权重...")
        
        # 检查 LoRA 权重路径是否存在必要的文件
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        for file in required_files:
            if not os.path.exists(os.path.join(lora_path, file)):
                # 尝试 .bin 格式
                if file == 'adapter_model.safetensors':
                    bin_file = 'adapter_model.bin'
                    if os.path.exists(os.path.join(lora_path, bin_file)):
                        logger.info(f"找到 {bin_file} 替代 {file}")
                        continue
                logger.error(f"LoRA 权重缺少必要文件: {os.path.join(lora_path, file)}")
                raise FileNotFoundError(f"Missing {file} in {lora_path}")
        
        # 将 VAE 移动到指定设备
        vae = vae.to(device)
        
        # 应用 LoRA 到解码器
        vae.decoder = PeftModel.from_pretrained(vae.decoder.to(device), lora_path)
        logger.info("LoRA 权重加载成功")
        
        # 4. 合并 LoRA 权重
        logger.info("正在合并 LoRA 权重...")
        vae.decoder = vae.decoder.merge_and_unload()
        logger.info("LoRA 权重合并完成")
        
        # 5. 转换数据类型
        logger.info(f"转换模型数据类型为: {output_dtype}")
        vae = vae.to(output_dtype)
        
        # 6. 设置为评估模式
        vae.eval()
        
        # 7. 保存合并后的模型
        logger.info(f"正在保存合并后的模型到: {output_path}")
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 保存完整模型 - 根据选项选择格式
        vae.save_pretrained(
            output_path,
            safe_serialization=save_safetensors,  # 根据参数选择保存格式
            variant=None  # 不使用变体命名
        )
        
        logger.info("合并后的 VAE 模型保存成功！")
        
        # 8. 验证合并结果
        logger.info("正在验证合并结果...")
        try:
            # 重新加载合并后的模型验证
            merged_vae = AutoencoderKL.from_pretrained(
                output_path, 
                torch_dtype=output_dtype,
                use_safetensors=save_safetensors  # 根据保存格式加载
            )
            logger.info("合并后的模型验证加载成功！")
            
            # 比较参数数量（可选验证）
            original_decoder_params = sum(p.numel() for p in vae.decoder.parameters())
            merged_decoder_params = sum(p.numel() for p in merged_vae.decoder.parameters())
            
            if original_decoder_params == merged_decoder_params:
                logger.info(f"参数数量验证通过: {original_decoder_params:,} parameters")
            else:
                logger.warning(f"参数数量不匹配: 原始 {original_decoder_params:,}, 合并后 {merged_decoder_params:,}")
                
            # 显示模型大小信息
            total_params = sum(p.numel() for p in merged_vae.parameters())
            logger.info(f"模型总参数量: {total_params:,}")
            
            # 计算模型文件大小
            model_size_mb = 0
            for file in os.listdir(output_path):
                if file.endswith(('.safetensors', '.bin', '.pt')):
                    file_path = os.path.join(output_path, file)
                    model_size_mb += os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"模型文件总大小: {model_size_mb:.2f} MB")
            
            # 列出保存的文件
            saved_files = [f for f in os.listdir(output_path) if f.endswith(('.safetensors', '.bin'))]
            logger.info(f"保存的模型文件: {saved_files}")
                
        except Exception as e:
            logger.error(f"验证合并结果时出错: {e}")
            # 即使验证失败，模型文件已经保存
        
        return True
        
    except Exception as e:
        logger.error(f"合并过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Merge trained LoRA weights back to VAE model")
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="原始 VAE 模型路径 (.bin 或 .safetensors)"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="训练好的 LoRA 权重路径（包含 adapter_config.json 和 adapter_model.safetensors/bin）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="合并后模型的保存路径"
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="输出模型的数据类型"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="处理时使用的设备"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="safetensors",
        choices=["safetensors", "bin"],
        help="保存模型的格式"
    )
    
    args = parser.parse_args()
    
    # 解析数据类型
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    output_dtype = dtype_map[args.output_dtype]
    
    # 解析保存格式
    save_safetensors = (args.save_format == "safetensors")
    
    logger.info("="*70)
    logger.info("VAE LoRA 合并脚本 - 支持多种格式输入输出")
    logger.info("="*70)
    logger.info(f"原始 VAE: {args.vae_path}")
    logger.info(f"LoRA 权重: {args.lora_path}")
    logger.info(f"输出路径: {args.output_path}")
    logger.info(f"输出数据类型: {args.output_dtype}")
    logger.info(f"处理设备: {args.device}")
    logger.info(f"保存格式: {args.save_format}")
    logger.info("="*70)
    
    success = merge_lora_to_vae(
        vae_path=args.vae_path,
        lora_path=args.lora_path,
        output_path=args.output_path,
        output_dtype=output_dtype,
        device=args.device,
        save_safetensors=save_safetensors
    )
    
    if success:
        logger.info("="*70)
        logger.info("✅ LoRA 合并成功！")
        logger.info(f"合并后的模型已保存到: {args.output_path}")
        logger.info(f"模型数据类型: {args.output_dtype}")
        logger.info(f"模型格式: {args.save_format}")
        logger.info("="*70)
    else:
        logger.error("="*70)
        logger.error("❌ LoRA 合并失败！")
        logger.error("请检查错误日志并重试")
        logger.error("="*70)
        exit(1)

if __name__ == "__main__":
    main()