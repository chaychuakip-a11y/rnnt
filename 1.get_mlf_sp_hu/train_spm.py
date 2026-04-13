import sentencepiece as spm
import sys

# 输入文件：由 get_sent_mlf.py 生成的清洗后的文本
input_file = "out.mlf_sy.sent"
# 模型前缀
model_prefix = "spm_hu_bpe_5000"
# 词表大小 (针对车载项目统一使用 2000)
vocab_size = 2000

# 参考你提供的参数进行训练
spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=0.99999999,  # 参考你提供的高覆盖率
    model_type='bpe',
    # 关键：确保 <blank> 占据 ID 0 以适配 RNNT 损失函数
    user_defined_symbols=["<blank>", "<pad>", "<SIL>"], 
    train_extremely_large_corpus=True,
    num_threads=200,
    unk_id=len(["<blank>", "<pad>", "<SIL>"]), # unk 紧跟在自定义符号后面
    bos_id=-1,
    eos_id=-1,
    pad_id=-1
)

print(f"匈牙利语 SPM 模型训练完成！")
print(f"输出文件: {model_prefix}.model 和 {model_prefix}.vocab")
print(f"请检查 .vocab 文件确认 <blank> 是否在 ID 0。")
