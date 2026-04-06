use strict;

use lib "/work/asrdictt/lixu/sbin";
use share_hadoop;

my $jobqueue          = "chime";
my $jobname           = "wavPakCut";
my $num_reduce        = 1;
my $in_blocksize      = 256*1024*1024;
my $block_size        = 64*1024*1024;

my @hdir_src     = (
"/workdir/asrdictt/dasrdictt/taoyu/noisedata/chezai/noise_dz_x3x4_4Car10Scene/iflytek-20180521-part-00000",
);
my $hdir_out     = "/workdir/asrdictt/gasrdictt/zhyou2/mlg/202311_car/noises/noise_dz_x3x4_4Car10Scene_cut180";
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

my $bin_rmdata		= "/raw15/asrdictt/permanent/zhyou2/ps2/DATA/5.rand/tool/rmsent";
#my $bin_selectdata	="/ps2/asrdictt/zhyou2/DATA/5.rand/tool/selectData";
my $scp				= "out.record.scp.only3-4";
my $scp_dir			= "/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/french/0.data_prepare/merge_noise/car";

my $bin_wavSplitFixedLen	= "/work1/asrdictt/taoyu/bin/wavSplitFixedLen";

my @cmd_map;
my @cmd_red;
my @files;

my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

::RemoveHadoopDirIfExist($hdir_out);

@cmd_map = (
::CmdToLocal("$bin_wavSplitFixedLen wav 180 32000"),

# ::CmdToLocal("$bin_rmdata $scp"),

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
"$scp_dir/$scp",
$bin_rmdata,
$bin_wavSplitFixedLen,
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
."-Dmapreduce.map.memory.mb=4000 "
."-Dmapreduce.reduce.memory.mb=4000 "
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
