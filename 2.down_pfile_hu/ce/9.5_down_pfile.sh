#!/bin/bash
# 并行拉取 10 个 part 的 CE pfile（phone 对齐标签 + Fbank 特征）
part=($(seq 0 9))
for i in ${part[*]}
do
    echo "starting part $i"
    perl 102_get_pfile_from_hdfs_ce.pl $i >102_get_pfile_from_hdfs_ce.$i.log 2>&1 &
done

wait
echo "all ce pfile parts done"
