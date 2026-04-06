use strict;

@ARGV >= 1 || die "usage: pl split_id\n";

my (@split_id)   = @ARGV;

my $dfs          = "";#"hdfs://192.168.89.100:9040";
my $dfs_param    = $dfs eq "" ? "" : "-fs $dfs";

my $hdir         = "/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_common_1000h_merge_cz_400h_add_cn_noise5.7h_pure_noise190h_20260306";  ###set
my $nSplit       = 1;
my $nPart        = 100;
my $states_list  = "/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/get_phone_ubctc/phone.map.list.reverse";
my $scp_filter   = "";
my $descript_fea = "fbnocmn40";
my $descript_lab = "mlf_fa_ph";
my $IsCTClab     = "0";

##out
my $dir_pwd      = '/raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_tune_20260306_add_noise_sp2.0k';  ###set
my $dir_lib      = "$dir_pwd/lib_fb40";

#tool
my $sbin_wait_hdir                     = "/work1/asrdictt/taoyu/tools/sbin/wait_dir_hdfs.pl";
my $bin_qnfiletransfer                 = "/work1/asrdictt/taoyu/tools/bin/qnfiletransfer_fast_cdph";    ##state fa
my $bin_qnnorm                         = "/work1/asrdictt/taoyu/tools/QN/bin/qnnorm";
my $bin_stat                           = "/work1/asrdictt/taoyu/tools/bin/stat_state_count_with_pfile";
my $bin_pak_low_frame_rate             = "/work1/asrdictt/taoyu/bin/pak_low_frame_rate";
my $bin_pak_low_frame_rate_labelexpand = "/work1/asrdictt/taoyu/bin/pak_low_frame_rate_labelexpand";
my $cmd;

my $partPerSplit = $nPart / $nSplit;
$nPart % $nSplit == 0 || die "Error: not support";

$cmd = "perl $sbin_wait_hdir $hdir/_SUCCESS $dfs";
system($cmd);

foreach my $split_id(@split_id)
{
    my $hdir_src     = "$hdir/*$split_id";
    my $file_norm    = "$dir_lib/fea.norm$split_id";
    my $pfile_fea    = "$dir_lib/fea.pfile$split_id";
    my $pfile_lab    = "$dir_lib/lab.pfile$split_id";
    my $scp_lab      = "$dir_lib/lab.scp$split_id";
    my $trans_log    = "$dir_lib/trans.log$split_id";

    system("mkdir -p $dir_lib") if(!-e $dir_lib);

    {
        system("touch $pfile_fea.done.start");
        $cmd = "/home3/hadoop/hadoop2/bin/hdfs dfs $dfs_param -cat $hdir_src |$bin_qnfiletransfer - $descript_fea $pfile_fea $descript_lab $pfile_lab $states_list $scp_lab $IsCTClab $scp_filter >$trans_log 2>&1";
        print $cmd."\n";
        system($cmd);
        system("touch $pfile_fea.done.finish");
    }
}
