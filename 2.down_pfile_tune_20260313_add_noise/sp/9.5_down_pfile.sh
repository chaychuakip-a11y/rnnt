part=($(seq 0 9))
for i in ${part[*]}
do
    echo $i
    perl 9.down_pfile_spm2.0k.pl $i > 9.down_pfile_spm2.0k.$i.log 2>&1 &
done
