use strict;

use lib "/work1/asrdictt/taoyu/sbin";
use share_hadoop;

my $jobname           = "maeopen";#SET
my $jobqueue          = "asr_trans";#SET
my $num_reduce        = 10; #SET
my $in_blocksize      = 512*1024*1024;
my $block_size        = 64*1024*1024;
my $replication       = 2;
my $thread_nums       = 10;

my @hdir_src     = ( ## 1/10
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_byd_0.2/*9",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_car_0.2/*9",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_duodian_0.1/*9",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_gswan_0.2/*9",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_jiaju15s_0.1/*9",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_music0_0.02/*9",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_music1_0.02/*9",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_music2_0.02/*9",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_onenoise_music_0.02/*9",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_tv_0.02/*9",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_white_0.1/*9",
                   );
my $hdir_out     = ("/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/wav_mix_all_noisy_maeOpen_0.1");
my $hdir_src          = join(" -input ", @hdir_src);

my $dir_bin           = "/work1/asrdictt/taoyu/bin";
my $bin_randname      = "$dir_bin/randname";
my $bin_randnamered   = "$dir_bin/randnamered";

my $bin_stream        = "$dir_bin/streamingAC-2.5.0.jar";
my $bin_selecttail    = "$dir_bin/selecttail";
my $bin_renametail    = "$dir_bin/renametail";

##### MAE process
my $bin_amp           = "$dir_bin/wavAmplify";#"/lustre2/asrdictt/yjjiang/jst-data/tools/wavAmplify/Debug/wavAmplify"
my $noisedataL        = "/raw15/asrdictt/permanent/taoyu/ps/mae/7.5kh_srfmae/noise_L.scp.tmp.pak.1";#"/lustre2/asrdictt/yjjiang/jst-data/jobs/for_lishang/NOCMN-model/chezai_local_ce/Project1_chnengphone/nocmn/data_process/7.5kh_srfmae/noise_L.scp.tmp.pak.1";#IN
my $noisedataR        = "/raw15/asrdictt/permanent/taoyu/ps/mae/7.5kh_srfmae/noise_R.scp.tmp.pak.1";#"/lustre2/asrdictt/yjjiang/jst-data/jobs/for_lishang/NOCMN-model/chezai_local_ce/Project1_chnengphone/nocmn/data_process/7.5kh_srfmae/noise_R.scp.tmp.pak.1";#IN

my $RIRX              = "/raw15/asrdictt/permanent/taoyu/ps/mae/RIRs/jiashi_open.txt";#"/lustre1/embed/lishang/lishang/jst-data/code/mt_mae/mt_mae/Debug/RIRs/jiashi_open.txt";
my $bin_mt_MAE        = "$dir_bin/mt_mae";                       #"/raw15/asrdictt/permanent/taoyu/ps/code/mt_mae/mt_mae/Debug/mt_mae";

##### feature extraction
my $bin_raw_fea       = "$dir_bin/htk-0.1.4/bin/raw_fea";
my $bin_cmvn          = "$dir_bin/htk-0.1.4/bin/cmvn_simple";
my $config1           = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB24_0_D_A_P3";
my $config2           = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB40";

my @cmd_map;
my @cmd_red;
my @files;
my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

@cmd_map = (
::CmdToLocal("$bin_selecttail mlf_sy wav"),
::CmdToLocal("$bin_mt_MAE $RIRX 1.0 $noisedataL $noisedataR"),
#::CmdToLocal("$bin_raw_fea $config1 fbp3"),
#::CmdToLocal("$bin_cmvn 2 24 1 fbp3"),
# ::CmdToLocal("$bin_raw_fea $config2 fbnocmn40"),
# ::CmdToLocal("$bin_raw_fea $config2 fb40"),
# ::CmdToLocal("$bin_cmvn 2 40 1 fb40"),
::CmdToLocal("$bin_selecttail wav mlf_sy"),
::CmdToLocal("$bin_randname"),
);

@cmd_red = (
::CmdToLocal("$bin_randnamered"),
);

@files   = (
$bin_mt_MAE,
$RIRX,
$noisedataL,
$noisedataR,
$bin_amp,

$bin_raw_fea,
$bin_cmvn,
$config1,
$config2,

$bin_selecttail,
$bin_renametail,
$bin_randname,
$bin_randnamered,
);

$cmd_map = join(" | ", @cmd_map);
$cmd_red = join(" | ", @cmd_red);

if(@cmd_map > 1)
{
	open(OUT, ">", "./mapper.$jobname.sh") || die $!;
	print OUT "#!/bin/bash\n";
	print OUT "$cmd_map\n";
	close OUT;
	push(@files, "./mapper.$jobname.sh");
	$cmd_map = "bash ./mapper.$jobname.sh";
}

if(@cmd_red > 1)
{
	open(OUT, ">", "./reducer.$jobname.sh") || die $!;
	print OUT "#!/bin/bash\n";
	print OUT "$cmd_red\n";
	close OUT;
	push(@files, "./reducer.$jobname.sh");
	$cmd_red = "bash ./reducer.$jobname.sh";
}

$files = join(",", @files);

$cmd =  "hadoop jar $bin_stream "
."-Dmapreduce.map.java.opts=\"-Xmx36000m\" "
."-Dmapreduce.reduce.java.opts=\"-Xmx4096m\" "
."-Dmapreduce.map.cpu.vcores=$thread_nums "
."-Dmapreduce.map.failures.maxpercent=20 "
."-Dmapreduce.job.queuename=$jobqueue "
."-Dmapreduce.job.name=$jobname "
."-Dmapreduce.job.reduces=$num_reduce "
."-Dmapreduce.map.memory.mb=14000 "
."-Dmapreduce.reduce.memory.mb=2000 "
."-Ddc.input.block.size=$in_blocksize "
."-Ddfs.blocksize=$block_size "
."-Ddfs.replication=$replication "
."-files \"$files\" "
."-input $hdir_src "
."-output $hdir_out "
;

$cmd .= "-mapper \"$cmd_map\" " if(@cmd_map > 0);
$cmd .= "-reducer \"$cmd_red\" " if(@cmd_red > 0);

::RemoveHadoopDirIfExist($hdir_out);
::PR($cmd);
::SuccessOrDie("$hdir_out");
