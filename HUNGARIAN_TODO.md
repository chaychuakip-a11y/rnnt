# TODO: 匈牙利语语种适配 (RNNT 项目)

本文档记录了将 Whisper ASR 流水线适配至匈牙利语所需的任务。

## 1. 数据准备阶段
- [ ] **目录设置**：基于 `1.get_mlf_sp_common` 创建 `1.get_mlf_sp_hu`。
- [ ] **文本清洗 (`get_sent_mlf.py`)**：
    - [ ] 定义匈牙利语字符集：`á, é, í, ó, ö, ő, ú, ü, ű` (及对应小写)。
    - [ ] 实现规范化逻辑：处理 `õ` -> `ő` 的编码替换。
- [ ] **SPM 训练 (SentencePiece)**：
    - [ ] 训练 BPE 模型，词表大小建议为 5000 (5k)。
    - [ ] **关键约束**：确保 `<blank>` 占位符固定在 **ID 0**。
- [ ] **Pfile 生成**：
    - [ ] 准备音素级 (Phone-level / FA) 和 BPE 级标签。
    - [ ] 打包成 Pfile 格式 (包含 Header + Data + Tail Index)。

## 2. 模型架构适配 (`net_*.py`)
参考文件：`11_train_2000_cectc/net_relu__addS_phCTC-add-hidden_skip_try2_para.py`

- [ ] **词表大小 (Vocabulary Size)**：更新 `voc_size` (第 25 行)，匹配新的 BPE 数量 (如 5000)。
- [ ] **CTC 脑袋 (分支 A)**：
    - [ ] 更新 `self.dnn_skip_out` (第 317 行) 的输出维度为 `voc_size + 1`。
- [ ] **Joiner 脑袋 (分支 B)**：
    - [ ] 更新 `self.joint` (第 602 行) 的 `vocab_size` 为 `voc_size + 1`。
- [ ] **Phone-CE 分支 (音素分支)**：
    - [ ] 更新 `self.phone_dim` (第 607 行)，匹配匈牙利语音素个数 (由 FA 结果决定)。
- [ ] **Decoder Embedding (解码器嵌入层)**：
    - [ ] 验证 `MaskEmbedding` (第 440 行) 是否正确接收了更新后的 `vocab_size`。

## 3. 训练与演进流水线
- [ ] **跨语种初始化 (Cross-lingual Init)**：
    - [ ] 从预训练的法语/俄语模型中提取 Encoder 权重。
    - [ ] 使用 `get_init_pt_rand.py` 初始化新的匈牙利语模型。
- [ ] **Step 11: CE 训练 (声学基础)**：
    - [ ] 使用匈牙利语音素标签训练声学编码器基础。
- [ ] **Step 11_cectc: 联合训练 (Joint Training)**：
    - [ ] 同时训练 RNNT + CTC + Phone-CE 三种损失函数。
    - [ ] **验证指标**：监控 `loss_binary` 和 `skip_rate` (跳帧率)，确保推理优化有效。
- [ ] **后期处理 (Post-Processing)**：
    - [ ] 运行 `11_clamp` 进行数值范围限制 (QAT)。
    - [ ] 运行 `12_quant` 进行 8-bit 量化系数搜索。

## 4. 验证与部署
- [ ] **跳帧机制检查**：验证 CTC 分支的 `blank` 概率 (>0.9) 能在 `decode_skip` 中正确触发跳帧。
- [ ] **匈牙利语 CER 评估**：在匈牙利语测试集上评估字符错误率 (CER)。
