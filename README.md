# FALumina
支持SDXL模型(E预测)`DirectAlign`训练方法
支持对于VAE的`Decoder`进行LoRA微调 (bfloat16)

## 开源模型
|HuggingFace🤗|ModelScope🤖|
|:--:|:--:|
|[ilemon/DirectAlignMitigatesRewardTamperingLoRA](https://huggingface.co/ilemon/DirectAlignMitigatesRewardTamperingLoRA)|[AiLieLemon/DirectAlignMitigatesRewardTamperingLoRA](https://www.modelscope.cn/models/AiLieLemon/DirectAlignMitigatesRewardTamperingLoRA)|

## 环境配置
项目使用*python3.11*搭建
```
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu118 # 国内用户
pip install deepspeed
pip install diffusers transformers peft datasets pillow lpips
```

## 启动命令
### DirectAlign

```
deepspeed --num_gpus=1 DirectAlign-SDXL.py \
    configs/DA_xl_config.toml \
    --deepspeed_config configs/DA_ds_config.json
```
### VAE-Decoder

#### 训练
```
deepspeed --num_gpus=1 VAEtrainer.py \
    --config_path configs/VD_xl_config.toml \
    --deepspeed_config configs/VD_ds_config.json
```

#### 合并权重
```
python merge.py \
    --vae_path /root/vae/diffusion_pytorch_model.bin \
    --lora_path /root/outputvae/final \
    --output_path /root/merged_vae/final-merged-bf16 \
    --output_dtype bfloat16 \ # float32 or float16
    --save_format safetensors \ # or bin
    --device cuda # or cpu
```