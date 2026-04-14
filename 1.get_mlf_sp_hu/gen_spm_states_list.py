#!/usr/bin/env python3
"""
gen_spm_states_list.py — 从 spm_hu_bpe_2000.vocab 生成两个词表文件：

  states_list.hu.spm2.0k
      每行一个 BPE piece，按 ID 顺序（第 0 行 = ID 0）
      供 qnfiletransfer_subword_encdec.bin 读取 states 索引

  states_list.hu.spm2.0k.map
      格式：<piece> <piece>（自映射 dict）
      供 qnfiletransfer_subword_encdec.bin 将 MLF 中的 piece 字符串映射到状态
      也供 GetOOVWordFromCorpus_v2.pl 做 OOV 检查

用法：
    python3 gen_spm_states_list.py [vocab_file]
    默认 vocab_file = spm_hu_bpe_2000.vocab（与本脚本同目录）

vocab 文件格式（SentencePiece 输出）：
    <token>\t<log_prob>
    第 0 行 = ID 0，依此类推
    特殊 token 顺序（由 train_spm.py 的 user_defined_symbols 决定）：
      ID 0: <blank>
      ID 1: <pad>
      ID 2: <SIL>
      ID 3: <unk>
      ID 4+: BPE pieces
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

vocab_file   = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "spm_hu_bpe_2000.vocab")
states_file  = os.path.join(script_dir, "states_list.hu.spm2.0k")
map_file     = os.path.join(script_dir, "states_list.hu.spm2.0k.map")

tokens = []
with open(vocab_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        # vocab 格式：<token>\t<score>
        parts = line.split("\t")
        token = parts[0]
        tokens.append(token)

print(f"Loaded {len(tokens)} tokens from {vocab_file}")

# states_list: one token per line, in ID order
with open(states_file, "w", encoding="utf-8") as f:
    for token in tokens:
        f.write(token + "\n")
print(f"Written: {states_file}")

# map_list: <token> <token>  (self-mapping dictionary)
# qnfiletransfer_subword_encdec.bin: 每个 BPE piece 本身即是一个状态
# GetOOVWordFromCorpus_v2.pl: 用于判断 sent.mlf_sp 里的 piece 是否在词表中
with open(map_file, "w", encoding="utf-8") as f:
    for token in tokens:
        f.write(f"{token} {token}\n")
print(f"Written: {map_file}")

print("Done.")
print(f"  states count : {len(tokens)}")
print(f"  ID 0 (<blank>): {tokens[0]}")
print(f"  ID 1 (<pad>)  : {tokens[1] if len(tokens) > 1 else 'N/A'}")
print(f"  ID 2 (<SIL>)  : {tokens[2] if len(tokens) > 2 else 'N/A'}")
print(f"  ID 3 (<unk>)  : {tokens[3] if len(tokens) > 3 else 'N/A'}")
print(f"  ID 4 (first BPE piece): {tokens[4] if len(tokens) > 4 else 'N/A'}")
