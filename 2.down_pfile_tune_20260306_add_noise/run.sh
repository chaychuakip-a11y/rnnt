###注意15.pfile_paste.pl中$dir_lib_supervised_ctcTriphone_ed/下的lab.pfile.$i 还是 lab.pfile$i
####ed和ctc数据条目不一致（mismatch导致），则需要先进行数据对齐，然后再paste操作
part=($(seq 0 9))
for i in ${part[*]}
do
    echo $i
    ed_dir=/raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_tune_20260306_add_noise_sp2.0k
    ce_dir=$ed_dir/lib_fb40
    ed_len=$ed_dir/lab.len.$i
    ce_len=$ce_dir/lab.len$i
    /work1/asrdictt/hjwang11/bin/pfile_info -p -q $ed_dir/lab.pfile.$i > $ed_len
    /work1/asrdictt/hjwang11/bin/pfile_info -p -q $ce_dir/lab.pfile$i > $ce_len
    perl 11.get_same_labpfile.pl $ce_len $ed_len $i >get_same_labpfile.$i.log
    sh 12.run_pfile_rand_by_index.sh $ed_dir $i >run_pfile_rand_by_index.$i.log
    ln -s /raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_tune_20260306_add_noise_sp2.0k/lib_fb40/fea.pfile$i /raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_tune_20260306_add_noise_sp2.0k/lib_fb40/mix/fea.pfile$i
    perl 15.pfile_paste.pl $ed_dir $i >pfile_paste.$i.log 2>&1 &
done

