use strict;

use lib "/work/asrdictt/lixu/sbin";
use share_hadoop;

my $jobqueue          = "asr_trans";
my $jobname           = "selectData";
my $num_reduce        = 1;
my $in_blocksize      = 256*1024*1024;
my $block_size        = 64*1024*1024;

my @hdir_src     = (
"/workdir/asrmlg/dasrmlg/ddye2/noise_huanxing/onenoise_music/onenoise_music.scp.1.pak",  ##单点噪声-音乐
"/workdir/asrmlg/dasrmlg/ddye2/noise_huanxing/onenoise_tv/onenoise_tv.scp.1.pak",        ##单点噪声-电视
"/workdir/asrdictt/dasrdictt/taoyu/noisedata/16k/2.700h_pure_music_data.pak",  #### 音乐噪声, from转写
"/workdir/asrmlg/dasrmlg/ddye2/noise_huanxing/music_1/music_1.scp.1.pak",                ## 音乐 from 多通道代码
"/workdir/asrmlg/dasrmlg/ddye2/noise_huanxing/music_2/music_2.scp.1.pak",                ## 音乐 from 多通道代码
);
my $hdir_out     = "/workdir/asrdictt/gasrdictt/zhyou2/mlg/202311_car/noises/noise_music_tv_mix_all.pak";
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
my $bin_raw_fea		= "/train8/asrmlg/ddye2/tools_new/htk-0.1.4/bin/raw_fea";
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
# ::CmdToLocal("$bin_raw_fea $config fbnocmn40"),
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
."-Dmapreduce.map.memory.mb=40000 "
."-Dmapreduce.reduce.memory.mb=40000 "
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
