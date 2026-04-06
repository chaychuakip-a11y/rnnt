use strict;

use lib "/work1/asrdictt/hjwang11/sbin";
use share_hadoop;

my $jobname           = "6.0.LsaDenoise.pl";#SET
my $jobqueue          = "asr_dictt";#SET
my $num_reduce        = 100;
my $in_blocksize      = 256*1024*1024;
my $block_size        = 64*1024*1024;
my $replication       = 2;

my @hdir_src     = (
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise_rm_head_44bytes/*[6-7]",
                   );
my $hdir_out     = ("/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise_0.2_lsa");
my $hdir_src     = join(" -input ", @hdir_src);

my $dir_bin            = "/work1/asrdictt/hjwang11/bin";
my $bin_stream         = "$dir_bin/streamingAC-2.5.0.jar";

my $bin_denoise        = "$dir_bin/wav_lsa"; #/ps/asr/taoyu/code/wav_lsa/wav_lsa/Release/wav_lsa

my $bin_selecttail     = "$dir_bin/selecttail";
my $bin_renametail     = "$dir_bin/renametail";
my $bin_randname       = "$dir_bin/randname";
my $bin_randnamered    = "$dir_bin/randnamered";

my @cmd_map;
my @cmd_red;
my @files;
my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

@cmd_map = (
::CmdToLocal("$bin_selecttail wav mlf_sy"),
::CmdToLocal("$bin_denoise"),
::CmdToLocal("$bin_randname"),
);

@cmd_red = (
::CmdToLocal("$bin_randnamered"),
);

@files   = (
$bin_denoise,
$bin_selecttail,
#$bin_renametail,
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
#."-Dmapreduce.map.cpu.vcores=$thread_nums "
."-Dmapreduce.map.failures.maxpercent=20 "
."-Dmapreduce.job.queuename=$jobqueue "
."-Dmapreduce.job.name=$jobname "
."-Dmapreduce.job.reduces=$num_reduce "
."-Dmapreduce.map.memory.mb=3000 "
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

