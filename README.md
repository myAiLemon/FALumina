# FALumina

## 环境配置
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