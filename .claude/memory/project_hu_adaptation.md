---
name: Hungarian RNN-T adaptation — current progress
description: 匈牙利语适配进度、所有已修改/新建文件、待填占位符、下一步行动
type: project
originSessionId: d99ac202-8e86-4cd4-ad17-65bdcfb646d3
---
## 目标
将多语种 RNN-T ASR Pipeline 从法语适配到匈牙利语（通用数据，暂无车载数据）。

## 关键参数（已确认）
- SPM vocab_size = 2000，blank=ID 0，user_defined_symbols=["<blank>","<pad>","<SIL>"]，unk_id=3
- FA 模型：韩语 FA 模型复用，9004 states，104 phones，fb72 特征
- 训练特征：fbnocmn40（40维，无CMN）
- GPU：8×A100，BMUF 分布式
- 训练路径：CE预训练 → CTC预训练 → RNNT联合训练

## 所有内网路径（已确认）
| 用途 | 路径 |
|---|---|
| HDFS 原始音频 | `/workdir/asrdictt/dasrdictt/tyliu23/hu/src_wav/tongyong` |
| HDFS FA 输出 | `/workdir/asrdictt/dasrdictt/tyliu23/hu/dnnfa/tongyong` |
| HDFS FA 结果子目录 | `.../hungarian_wav_dnnfa` |
| FA AM 模型（wts）| `/yrfs4/asrdictt/tyliu23/am/hu/ubctc/2_dnn_fb72/3_train/mlp-ring-h2048_2048_2048_2048_2048_2048_512-cw11-targ9004-step1_b4096_jumpframe3-2-5_discard0/mlp.9.merge` |
| FA fea.norm | `/yrfs4/asrdictt/tyliu23/am/hu/ubctc/2_dnn_fb72/1_down_pfile/lib_fb72/fea.norm` |
| FA states.count | `/yrfs4/asrdictt/tyliu23/am/hu/ubctc/2_dnn_fb72/1_down_pfile/lib_fb72/states.count.low100.txt` |
| hmmlist（phone states_list）| `/yrfs4/asrdictt/tyliu23/am/hu/ubctc/1_mle_mfc/final_s9k/hmmlist.final` |
| FA 语言模型 | `/yrfs4/asrdictt/tyliu23/am/hu/ubctc/res_fa/out_package_1gram` |
| atom FA config | `/work1/asrdictt/taoyu/bin/atom-v20151016b/atom_hadoop_dnnfa.fb72.cfg` |
| 用户工作根目录 | `/yrfs4/asrdictt/tyliu23/` |

## 已修改文件（4个）

### `1.get_mlf_sp_hu/spm_encode.sh`
- `--model` 从法语硬路径 → `$(dirname "$0")/spm_hu_bpe_2000.model`

### `1.get_mlf_sp_hu/train_spm.py`
- 匈牙利语训练语料路径和参数（已完成）

### `11_train_2000_cectc/0_run_a100.sh`
- `-d hu_phcectc`，`-n train-hu-phcectc`，`--modelPath /yrfs4/asrdictt/tyliu23/`

### `11_train_2000_cectc/net_relu__addS_phCTC-add-hidden_skip_try2_para.py`
- `word_dict` → `/yrfs4/asrdictt/tyliu23/am/hu/ubctc/1_mle_mfc/final_s9k/hmmlist.final`
- `phone_dim` → `104`（Line ~608）

## 已新建文件（11个）

### FA 处理：`0.data_process/2.hungarian_common_placeholder/`
- `config.json` — 所有内网路径的 JSON 配置
- `1_dnnfa.pl` — Hadoop DNN-FA 作业脚本（fb72，9004 states）
- `utils.pl` — JSON::PP 配置加载
- `run1_base.sh` — 串行运行 FA + GenSeedMlf

### pfile 打包：`2.down_pfile_hu/`
- `ce/102_get_pfile_from_hdfs_ce.pl` — CE phone 标签 pfile（fbnocmn40，qnfiletransfer_fast_cdph）
- `ce/103.create_norm.pl` — 计算并合并 10 part fea.norm
- `ce/9.5_down_pfile.sh` — 并行跑 10 part CE
- `sp/9.down_pfile_spm2.0k.pl` — BPE 标签 pfile（addtail + qnfiletransfer_subword_encdec.bin）
- `sp/9.5_down_pfile.sh` — 并行跑 10 part SP
- `11.get_same_labpfile.pl` — CE/SP 句子数对齐，生成 index.scp
- `12.run_pfile_rand_by_index.sh` — 按 index 重排 SP pfile
- `15.pfile_paste.pl` — pfile_paste 合并成 2列 mix/lab.pfile（列1=BPE，列2=phone）
- `run.sh` — 合并主流程

### 训练配置
- `11_train_2000_cectc/config_all.ini` — 训练配置（含所有 BMUF/8xA100 参数，占位符待填）
- `1.get_mlf_sp_hu/gen_spm_states_list.py` — 从 `.vocab` 生成 `states_list.hu.spm2.0k` 和 `.map`

## 待填占位符（内网同步后处理）

| 占位符 | 位置 | 说明 |
|---|---|---|
| `PLACEHOLDER_HU_PFILE_OUT_DIR` | `ce/102_...pl`, `sp/9.down...pl`, `run.sh`, `ce/103...pl` | pfile 输出目录（yrfs4 或 raw15） |
| `PLACEHOLDER_HU_SPM_MLF` | `sp/9.down_pfile_spm2.0k.pl` | `out.mlf_sy.mlf.danzi.mlf_sp` 全路径（注意是合并总文件，不是 per-part sent.mlf_sp） |
| `PLACEHOLDER_HU_SPM_STATES_LIST` | `sp/9.down_pfile_spm2.0k.pl` | `states_list.hu.spm2.0k` 全路径 |
| `PLACEHOLDER_HU_SPM_MAP_LIST` | `sp/9.down_pfile_spm2.0k.pl` | `states_list.hu.spm2.0k.map` 全路径 |
| `PLACEHOLDER_HU_FBANK_PFILE_DIR` | `config_all.ini [DataSetting]` | fea.pfile 所在目录（mix/ 的父目录） |
| `PLACEHOLDER_HU_LABEL_PFILE_DIR` | `config_all.ini [DataSetting]` | lab.pfile 所在目录（mix/） |
| `PLACEHOLDER_HU_NORM_FILE` | `config_all.ini [DataSetting]` | fea.norm 全路径（103.create_norm.pl 生成后） |
| `initEncoderModel`（注释中）| `config_all.ini [TrainSetting]` | 法语 CECTC 预训练模型路径（跨语种 Encoder 初始化） |

## Stage 1 执行顺序（FA 完成后）

```bash
cd 1.get_mlf_sp_hu/
bash run.sh                        # HDFS FA 输出解包 → 生成 0/~9/ 各 part 目录
bash 0.5.run_cp_get_sent_mlf.sh    # get_sent_mlf.py → out.mlf_sy.sent + out.mlf_sy.scp
bash 2.run_cp_spm_encode.sh        # spm_encode.sh → out.mlf_sy.sent.mlf_sp（per part）
bash 3.run_cp_get_mlf_sp.sh        # get_mlf_sp.pl → out.mlf_sy.mlf.danzi.mlf_sp（per part）
bash 4.cat_mlf_sp.sh               # 合并 → out.mlf_sy.mlf.danzi.mlf_sp（总文件）
python3 gen_spm_states_list.py     # 生成 states_list.hu.spm2.0k + .map（需要 .vocab 已在）
```

## 注意事项 / 潜在风险

1. **`<s>=39, </s>=40` 硬编码**：`dataloader.py:369-372` 中有两个硬编码索引，需要对照匈牙利语 `hmmlist.final` 中的 `<s>` 和 `</s>` 实际行号验证。

2. **pfile 2列格式**：`mix/lab.pfile` 列1=BPE（→RNNT+CTC损失），列2=phone FA（→Phone-CE损失）。`dataloader.py` 中 `meta["att_label"]` 读列1，`meta["label_ctc_ph"]` 读列2。

3. **跨语种初始化**：`config_all.ini` 中 `initEncoderModel` 加载法语 CECTC 权重，Decoder Embedding 和 Joint 输出层因 vocab_size 不同会自动跳过（shape mismatch）。

4. **`states_list` 文件含义区分**：
   - `hmmlist.final` = FA phone 状态列表（用于 CE pfile，`word_dict` 参数）
   - `states_list.hu.spm2.0k` = SPM BPE token 列表（用于 SP pfile，`qnfiletransfer` 参数）

5. **车载数据**：目前没有，`2.hungarian_common_placeholder/` 是通用数据占位。车载分支后续参考 `_cz` 目录新建。

**Why:** 记录所有历史会话中确认的路径和设计决策，避免下次会话重新推导。
**How to apply:** 新会话开始时先读取此文件，直接从"待填占位符"继续，不必重新探索已完成的部分。
