use strict;
use lib "/work1/asrdictt/hjwang11/aBak/clwu4/sbin";
use share_hadoop;

for(my $ii=0; $ii<1;$ii=$ii+1){
    my $num = $ARGV[0];
    my $dir_lib             = "/raw15/asrdictt/permanent/tyliu23/hu/rnnt/traindata/car_hu_common_sp2.0k";  ###set
    system("mkdir -p $dir_lib") if(!-e $dir_lib);

    my $bin_qnfiletransfer  = "/work1/asrdictt/hjwang11/aBak/clwu4/qnfiletranser_subword_encdec.bin";
    my $bin_qnnorm          = "/work1/asrdictt/hjwang11/aBak/clwu4/qnnorm";
    my $bin_cmvn            = "/work1/asrdictt/hjwang11/aBak/clwu4/cmvn_simple";
    my $bin_rm_bad          = "/work1/asrdictt/hjwang11/aBak/clwu4/pak_rm_bad_label.bin";
    my $bin_en_addjing      = "/work1/asrdictt/hjwang11/aBak/clwu4/pak_label_add_en_start_end.bin";

    my $rmsent              = "/raw15/asrdictt/permanent/zhyou2/ps2/DATA/5.rand/tool/rmsent";
    my $scp_filter          = "/work1/asrdictt/zhyou2/workspace/jobs/202311_car_mlg/turkish/0.data_prepare/all.rm0.scp";

    my $selecttail          = "/work1/asrdictt/taoyu/bin/selecttail";
    my $addtail             = "/work1/asrdictt/taoyu/bin/addtail";
    my $mlf_sp_file         = "/raw15/asrdictt/permanent/tyliu23/hu/rnnt/1.get_mlf_sp_hu/$num/out.mlf_sy.mlf.danzi.mlf_sp";  ###set

    my $descript_fea        = "fbnocmn40";
    my $descript_lab        = "mlf_sy";
    my $descript_lab_sp     = "mlf_sp";
    my $states_list         = "/raw15/asrdictt/permanent/tyliu23/hu/rnnt/spm/states_list.hu.spm2.0k";  ###set
    my $map_list            = "/raw15/asrdictt/permanent/tyliu23/hu/rnnt/spm/states_list.hu.spm2.0k.map";  ###set

    my $split_id            = $num;
    my $file_norm           = "$dir_lib/fea.norm.$split_id";
    my $pfile_fea           = "$dir_lib/fea.pfile.$split_id";
    my $pfile_lab           = "$dir_lib/lab.pfile.$split_id";
    my $scp_lab             = "$dir_lib/lab.scp.$split_id";
    my $trans_log           = "$dir_lib/trans.log.$split_id";
    my $inputdir            = "/workdir/asrdictt/dasrdictt/tyliu23/hu/dnnfa/tongyong/hungarian_wav_dnnfa/*$split_id";  ###set

    if(!-e "$pfile_fea.done.finish")
    {
        system("touch $pfile_fea.done.start");
        my $cmd = "hdfs dfs -cat $inputdir |$selecttail fbnocmn40 | $addtail $mlf_sp_file $descript_lab_sp| $bin_qnfiletransfer $map_list $states_list $descript_lab_sp $pfile_lab $descript_fea 40 - $scp_lab 0 >$trans_log 2>&1";

        print $cmd."\n";
        system($cmd);
        !system("$bin_qnnorm norm_ftrfile=$pfile_fea output_normfile=$file_norm") || die "qnnorm failed: $file_norm.\n";
        system("touch $pfile_fea.done.finish");
    }
}
