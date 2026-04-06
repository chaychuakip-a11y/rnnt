use strict;

use lib "/work/asrdictt/lixu/sbin";
use share_hadoop;

my $jobqueue          = "asr_dictt";
my $jobname           = "selectData";
my $num_reduce        = 100;
my $in_blocksize      = 256*1024*1024;
my $block_size        = 64*1024*1024;

my @hdir_src     = (
"/workdir/asrdictt/gasrdictt/zhyou2/mlg/202311_car/french/*",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_byd_0.2",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_car_0.2",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_duodian_0.1",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_gswan_0.2",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_jiaju15s_0.1",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_music0_0.02",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_music1_0.02",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_music2_0.02",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_onenoise_music_0.02",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_tv_0.02",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_addnoise_white_0.1",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/mlefa_mix_all_amp_0.2",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/wav_mix_all_noisy_lsaDenoise_0.2_mlefa",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/wav_mix_all_noisy_maeClose_0.1_mlefa",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/wav_mix_all_noisy_maeOpen_0.1_mlefa",
"/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/wav_mix_all_speedup1.2_0.2_mlefa",
);
my $hdir_out     = "/workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/dnnfa_fixmlf_addfangzhen_all_fbnocmn40";
my $hdir_src     = join(" -input ", @hdir_src);
#my $scp = "/work/asrdictt/lixu/fangyan/share_model/cn_sc_can/cn_1wh_can_4kh_sc_4kh/gendata/unpack/out.record.scp";
my $mlf = "",

my $dir_bin           = "/work/asrdictt/lixu/bin";
my $bin_stream        = "$dir_bin/streamingAC-2.5.0.jar";
my $bin_randname      = "$dir_bin/randname";
my $bin_randnamered   = "$dir_bin/randnamered";
my $bin_selecttail    = "$dir_bin/selecttail";
my $bin_pakselect     = "$dir_bin/selectrecord";
# my $config           = "/ps/sppro/dyliu2/asr_bak/hadoop/hadoop_jobs/chn_10000h/FA/1kh_if_nocmn/raw_fea.config.fb40"; 
	my $config           = "/work/asrdictt/xiaobao/bin/htk-0.1.4/cfg/raw_fea.config.fp3";
# my $bin_raw_fea      = "/ps/sppro/dyliu2/asr_bak/bin/htk/bin/raw_fea_fbk40";#fbk40
my $bin_raw_fea		= "/train8/asrmlg/ddye2/asr/italian/italian_gongban_ubctc_20230920/5_gen_fb40/bin/raw_fea";
my $bin_addmlf        = "$dir_bin/addmlf";

#my $bin_selectdata	="/ps2/asrdictt/zhyou2/DATA/5.rand/tool/selectData";
my $scp				= "out.record.scp";
my $scp_dir			= "./lid_code/cn1wh";

my @cmd_map;
my @cmd_red;
my @files;

my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

::RemoveHadoopDirIfExist($hdir_out);

@cmd_map = (
# ::CmdToLocal("$bin_pakselect $scp"),
#::CmdToLocal("$bin_selecttail wav mlf_sy"),
::CmdToLocal("$bin_raw_fea $config fbnocmn40"),
#::CmdToLocal("$bin_selecttail fbnocmn40 mlf_sy"),
::CmdToLocal("$bin_randname"),
);

@cmd_red = (
::CmdToLocal("$bin_randnamered"),
);

@files   = (
$bin_randname,
$bin_randnamered,
$bin_selecttail,
$bin_raw_fea,
$config,
$bin_pakselect,
# "$scp_dir/$scp",
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
."-Dmapreduce.map.failures.maxpercent=20 "
."-Dmapreduce.job.queuename=$jobqueue "
."-Dmapreduce.job.name=$jobname "
."-Dmapreduce.job.reduces=$num_reduce "
."-Dmapreduce.map.memory.mb=20000 "
."-Dmapreduce.reduce.memory.mb=20000 "
."-Ddc.input.block.size=$in_blocksize "
."-Ddfs.block.size=$block_size "
."-Ddfs.replication=2 "
."-files \"$files\" "
."-input $hdir_src "
."-output $hdir_out "
;
$cmd .= "-mapper \"$cmd_map\" " if(@cmd_map > 0);
$cmd .= "-reducer \"$cmd_red\" " if(@cmd_red > 0);
::PR($cmd);
::SuccessOrDie("$hdir_out");
