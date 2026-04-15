sp_dir=/raw15/asrdictt/permanent/tyliu23/hu/rnnt/traindata/car_hu_common_sp2.0k/sp/lib  ###set: SP pfile 根目录（lab.pfile.N 所在）
ce_dir=/raw15/asrdictt/permanent/tyliu23/hu/rnnt/traindata/car_hu_common_sp2.0k/lib_fb40  ###set: CE pfile 目录（lab.pfileN 所在）
mix_dir=$ce_dir/mix

mkdir -p $mix_dir

part=($(seq 0 9))
for i in ${part[*]}
do
    echo $i
    sp_len=$sp_dir/lab.len.$i
    ce_len=$ce_dir/lab.len$i
    /work1/asrdictt/hjwang11/bin/pfile_info -p -q $sp_dir/lab.pfile.$i > $sp_len
    /work1/asrdictt/hjwang11/bin/pfile_info -p -q $ce_dir/lab.pfile$i  > $ce_len
    perl 11.get_same_labpfile.pl $ce_len $sp_len $i >get_same_labpfile.$i.log
    sh 12.run_pfile_rand_by_index.sh $sp_dir $i >run_pfile_rand_by_index.$i.log
    ln -sf $ce_dir/fea.pfile$i $mix_dir/fea.pfile$i
    perl 15.pfile_paste.pl $sp_dir $ce_dir $i >pfile_paste.$i.log 2>&1 &
done

wait
echo "all parts done"
