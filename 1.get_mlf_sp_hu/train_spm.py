import sentencepiece as spm
import sys
import os

# 配置参数 (复用你提供的现有代码逻辑)
input_file = "out.mlf_sy.sent"
model_prefix = "spm_hu_bpe_2000"
vocab_size = 2000
character_coverage = 0.999999999
model_type = 'bpe'
num_threads = 200

# 关键符号定义
# 根据项目强制要求，<blank> 必须占据 ID 0。
# 按照你的习惯，我们也加入了 <pad> 和 <SIL>。
user_defined_symbols = ["<blank>", "<pad>", "<SIL>"]

print(f"开始训练匈牙利语子词模型 (BPE {vocab_size})...")

spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    user_defined_symbols=user_defined_symbols,
    character_coverage=character_coverage,
    model_type=model_type,
    train_extremely_large_corpus=True,
    num_threads=num_threads,
    # 针对该项目 RNNT 架构的特殊设置
    unk_id=len(user_defined_symbols), # unk 紧跟在自定义符号后
    bos_id=-1,
    eos_id=-1,
    pad_id=-1
)

print("-" * 30)
print(f"训练完成！")
print(f"模型文件: {model_prefix}.model")
print(f"词表文件: {model_prefix}.vocab")
print(f"请执行 'head -n 5 {model_prefix}.vocab' 确认 <blank> 是否在 ID 0。")
