#!/bin/bash
# 并行生成 10 个 part 的 SPM BPE 标签 pfile
part=($(seq 0 9))
for i in ${part[*]}
do
    echo "starting part $i"
    perl 9.down_pfile_spm2.0k.pl $i > 9.down_pfile_spm2.0k.$i.log 2>&1 &
done

wait
echo "all sp pfile parts done"
