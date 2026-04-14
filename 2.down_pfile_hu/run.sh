#!/bin/bash
# 2.down_pfile_hu — 匈牙利语 pfile 合并主流程
#
# 前置条件：
#   1. ce/9.5_down_pfile.sh 已跑完（生成 lib_fb40/fea.pfile* + lab.pfile*）
#   2. sp/9.5_down_pfile.sh 已跑完（生成 lab.pfile.* BPE 标签）
#   3. ce/103.create_norm.pl 已跑完（生成 lib_fb40/fea.norm）
#
# 本脚本完成：
#   - 逐 part 对齐 CE/SP 两套 pfile 的句子数
#   - 按索引重排 SP pfile，与 CE pfile 一一对应
#   - pfile_paste 合并成 mix/lab.pfile*（2列：列1=BPE, 列2=phone）
#   - 在 mix/ 下创建 fea.pfile* 的软链接

## 关键路径  ### set（与 ce/102_get_pfile_from_hdfs_ce.pl 中 dir_pwd 一致）
ED_DIR="PLACEHOLDER_HU_PFILE_OUT_DIR"
CE_DIR="$ED_DIR/lib_fb40"
MIX_DIR="$CE_DIR/mix"

mkdir -p "$MIX_DIR"

PFILE_INFO="/work1/asrdictt/hjwang11/bin/pfile_info"

part=($(seq 0 9))
for i in ${part[*]}
do
    echo "=== processing part $i ==="

    ED_LEN="$ED_DIR/lab.len.$i"
    CE_LEN="$CE_DIR/lab.len$i"

    # 1. 获取每个 part 的句长信息
    $PFILE_INFO -p -q "$ED_DIR/lab.pfile.$i" > "$ED_LEN"
    $PFILE_INFO -p -q "$CE_DIR/lab.pfile$i"  > "$CE_LEN"

    # 2. 生成对齐索引
    perl 11.get_same_labpfile.pl "$CE_LEN" "$ED_LEN" "$i" > get_same_labpfile.$i.log

    # 3. 按索引重排 SP pfile（lab.pfile.$i -> lab.pfile$i）
    sh 12.run_pfile_rand_by_index.sh "$ED_DIR" "$i" > run_pfile_rand_by_index.$i.log

    # 4. 软链接 fea.pfile 到 mix/（训练时从 mix/ 读 fea + lab）
    ln -sf "$CE_DIR/fea.pfile$i" "$MIX_DIR/fea.pfile$i"

    # 5. pfile_paste: 合并 CE 标签（列1）和 SP 标签（列2）
    perl 15.pfile_paste.pl "$ED_DIR" "$i" > pfile_paste.$i.log 2>&1 &
done

wait
echo "=== all parts done. mix/ ready for training. ==="

# 验证集软链接（config_all.ini 中 ValidationFeature/Label 指向 part0）
ln -sf "$CE_DIR/fea.pfile0" "$MIX_DIR/fea.pfile0" 2>/dev/null || true
