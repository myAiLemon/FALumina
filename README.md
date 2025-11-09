# FALumina
一个暂时只支持SDXL模型(E预测)的`DirectAlign`训练方法的脚本
## 开源模型
[DirectAlign Mitigates Reward Tampering LoRA](https://civitai.com/models/2112333/directalign-mitigates-reward-tampering-lora) `2000 steps`验证模型

## 环境配置
项目使用*python3.11*搭建
```
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu118 # 国内用户
pip install deepspeed
pip install diffusers transformers peft datasets pillow
```

## 启动命令
```
deepspeed --num_gpus=1 DirectAlign-SDXL.py \
    configs/DA_xl_config.toml \
    --deepspeed_config configs/ds_config.json
```
