# 匈牙利语适配进展报告 (2026-04-13)

## 1. 项目目标
- **语种**: 匈牙利语 (Hungarian)
- **场景**: 车载项目适配 (Car-borne ASR)
- **子词规模**: BPE 2000 (已根据硬件及推理延迟要求进行优化)
- **核心约束**: `<blank>` 必须固定在 **ID 0**，以支持 Frame Skipping (跳帧) 机制。

## 2. 已完成工作 (数据准备阶段)

### 2.1 目录结构
- 创建了专用处理目录 `1.get_mlf_sp_hu/`。
- 从 `common` 目录同步了基础处理脚本，并进行了针对性适配。

### 2.2 文本清洗逻辑 (`get_sent_mlf.py`)
- **字符集适配**: 加入了匈牙利语特有字符 `á, é, í, ó, ö, ő, ú, ü, ű` 及其小写支持。
- **编码修正**: 自动将常见的 `õ`/`Õ` 错误编码替换为正确的 `ő`。
- **格式支持**: 修改了段落识别逻辑，由原来的 `.mlf_sy"` 适配为本地标签常见的 **`.lab"`** 后缀。
- **鲁棒性增强**: 增加了对 `#!MLF!#` 文件头和单独点号 `.` 的过滤，防止非文本行干扰词表训练。

### 2.3 词表训练脚本 (`train_spm.py`)
- 基于项目惯例编写了 BPE 训练脚本。
- **参数配置**: 
    - `vocab_size=2000`
    - `character_coverage=0.99999999`
    - `user_defined_symbols=["<blank>", "<pad>", "<SIL>"]` (确保 `<blank>` 为 ID 0)。
- **语料规模**: 目前已准备约 **50 万行** 匈牙利语清洗后的文本 (`out.mlf_sy.sent`)。

## 3. 下一步计划
1. **训练词表**: 运行 `python train_spm.py` 生成 `.model` 文件。
2. **标签编码**: 修改 `spm_encode.sh` 或 `get_sent_mlf.py` 生成最终的 `out.mlf_sy.mlf_sp` 训练标签。
3. **模型适配**: 修改 `11_train_2000_cectc` 目录下的 `net_*.py`，将 `voc_size` 设为 2000，并调整输出层维度。

## 4. 修改文件清单
- `HUNGARIAN_TODO.md`: 任务清单。
- `HUNGARIAN_PROGRESS.md`: 进度报告。
- `1.get_mlf_sp_hu/get_sent_mlf.py`: 适配后的清洗脚本。
- `1.get_mlf_sp_hu/train_spm.py`: BPE 训练脚本。
