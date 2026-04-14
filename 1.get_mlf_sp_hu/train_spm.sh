#!/bin/bash

# 激活环境 (根据你提供的路径)
source /workl/asrdictt/zhyou2/anaconda3/bin/activate /workl/asrdictt/zhyou2/anaconda3/envs/pth39_cdl1l_tch191

# 执行现有的 SPM 训练脚本
# 注意：我们显式地将 <blank> 放在 user_defined_symbols 的第一位，确保其 ID 为 0。
python3 /workl/asrdictt/taoyu/python/spm/spm_train.py \
  --input=./out.mlf_sy.sent \
  --model_prefix=spm_hu_bpe_2000 \
  --user_defined_symbols="<blank>,<pad>,<SIL>" \
  --character_coverage=0.999999999 \
  --model_type=bpe \
  --train_extremely_large_corpus=true \
  --num_threads=200 \
  --vocab_size=2000

echo "----------------------------------------------------"
echo "匈牙利语 SPM 训练完成！"
echo "请执行 'head -n 5 spm_hu_bpe_2000.vocab' 检查 <blank> 是否在 ID 0。"
