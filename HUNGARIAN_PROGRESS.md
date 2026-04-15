# 匈牙利语适配进展报告 (2026-04-16)

## 1. 项目目标
- **语种**: 匈牙利语 (Hungarian)
- **场景**: 车载项目适配 (Car-borne ASR)
- **子词规模**: BPE 2000
- **核心约束**: `<blank>` 固定在 **ID 0**，支持 Frame Skipping (跳帧) 机制

---

## 2. 训练流水线

```
0_get_init_pt_rand.sh       俄语模型提取 Encoder → model.init
        ↓
10_train_2000_p1_ctc        Word CTC 预训练（SP 单列标签）
        ↓
11_train_2000_ce            Phone CE 训练（CE 单列标签）
        ↓
11_train_2000_cectc         RNNT + CTC + Phone-CE 联合训练（mix 双列标签）
        ↓
11_train_2000_cectc_clamp   数值范围限制 (QAT)
        ↓
12_quant                    8-bit 量化
```

---

## 3. 已完成工作

### 3.1 数据准备 (`1.get_mlf_sp_hu/`)
- 匈牙利语字符集适配（`á é í ó ö ő ú ü ű` 及编码修正 `õ→ő`）
- BPE 训练脚本 `train_spm.py`（vocab_size=2000，`<blank>` 固定 ID 0）
- SPM encode/decode 脚本

### 3.2 Pfile 制作 (`2.down_pfile_hu/`) ✅ 内网已完成
- **CE pfile**（`ce/`）: FA 对齐的 phone 帧级标签 + Fbank 特征，已生成 `lib_fb40/lab.pfileN`
- **SP pfile**（`sp/lib/`）: BPE 序列标签，已生成 `lab.pfile.N`（原始）和 `lab.pfileN`（对齐重排后）
- **mix pfile**（`ce/lib_fb40/mix/`）: CE(col1) + SP(col2) 双列合并，已验证
  - `pfile_info` 确认：2 labels，句子数/帧数正常

#### 关键脚本改动（相对 cz 模板）
| 文件 | 改动 |
|------|------|
| `ce/102_get_pfile_from_hdfs_ce.pl` | 替换 HU 路径；去掉 qnnorm（由 103 负责） |
| `ce/103.create_norm.pl` | 填入 HU lib_fb40 路径 |
| `sp/9.down_pfile_spm2.0k.pl` | 替换 HU 路径；mlf 按 part 读（`$num` 在路径中） |
| `15.pfile_paste.pl` | 改为 3 参数（`sp_dir ce_dir split_id`），适配 HU 目录结构 |
| `run.sh` | `sp_dir` / `ce_dir` 分离，内联实际路径 |

### 3.3 训练配置 (`10_train_2000_p1_ctc_russianinitmodel/`)
- 新增 `config_all.ini`（基于法语 cz 模板）
- 数据路径：`DataDir`=CE lib_fb40，`LabelDir`=SP lib（单列 BPE），`NormFile`=fea.norm0
- 验证集：`ValidationDatadir`=SP lib（需在 SP lib 下建 fea.pfile0 软链接）
- 网络文件 `net_relu__addS_phce-add-hidden_skip_try2_wordctc.py`：
  - `voc_size=2001` ✓（BPE 2000 + blank）
  - `phone_dim=41`：CE 分支 loss 已注释，此步不影响训练

---

## 4. 待完成事项

### Step 10: Word CTC 预训练
- [ ] 修复训练脚本 `train_multi_local_v100.sh` 的 bashrc/conda 环境路径
- [ ] 修复 `0_get_init_pt_rand.sh` 的 bashrc/conda 环境路径
- [ ] 运行 `bash 0_get_init_pt_rand.sh` 生成 `train_onlywordCTC_v0/model.init`
- [ ] 在 SP lib 下建软链接：`ln -s ce/lib_fb40/fea.pfile0 sp/lib/fea.pfile0`（验证集用）
- [ ] 确认 `asr/c.so-v` 存在
- [ ] 启动训练

### Step 11: CE 训练
- [ ] 配置 `11_train_2000_ce/config_all.ini`（CE 单列标签）
- [ ] 确认 `phone_dim` 与匈牙利语 phone 数量一致
- [ ] 从 Step 10 输出初始化

### Step 11: CECTC 联合训练
- [ ] 配置 `11_train_2000_cectc/config_all.ini`（mix 双列标签）
- [ ] 从 Step 11 CE 输出初始化

### Step 12: 量化
- [ ] 配置 `12_quant/`

---

## 5. 关键路径（内网）

| 类型 | 路径 |
|------|------|
| CE pfile | `/yrfs4/asrdictt/tyliu23/am/hu/rnnt/2.down_pfile_hu/ce/lib_fb40/` |
| SP pfile | `/yrfs4/asrdictt/tyliu23/am/hu/rnnt/2.down_pfile_hu/sp/lib/` |
| mix pfile | `/yrfs4/asrdictt/tyliu23/am/hu/rnnt/2.down_pfile_hu/ce/lib_fb40/mix/` |
| 俄语初始模型 | `/train8/asrmlg/ddye2/RNNT/russian/russian_gongban_xd_20250107/7_train_zhy_step2_3_ctc/out_train_002/model.iter1.part6` |
| SPM states_list | `/raw15/asrdictt/permanent/tyliu23/hu/rnnt/spm/states_list.hu.spm2.0k` |
| SPM map | `/raw15/asrdictt/permanent/tyliu23/hu/rnnt/spm/states_list.hu.spm2.0k.map` |
