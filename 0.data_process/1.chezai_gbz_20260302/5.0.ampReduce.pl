use strict;

use lib "/work1/asrdictt/hjwang11/sbin";
use share_hadoop;

my $jobname           = "5.0.ampReduce";#SET
my $jobqueue          = "asr_dictt";#SET
my $num_reduce        = 100;
my $in_blocksize      = 256*1024*1024;
my $block_size        = 64*1024*1024;
my $replication       = 2;

my @hdir_src     = (
                    "/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa/*[6-7]",
                   );
my $hdir_out     = ("/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_0.2_ampReduce");
my $hdir_src     = join(" -input ", @hdir_src);

##tools
my $dir_bin            = "/work1/asrdictt/taoyu/bin";
my $bin_stream         = "$dir_bin/streamingAC-2.5.0.jar";

my $bin_selecttail     = "$dir_bin/selecttail";
my $bin_renametail     = "$dir_bin/renametail";
my $bin_randname       = "$dir_bin/randname";
my $bin_randnamered    = "$dir_bin/randnamered";
my $bin_amprand        = "$dir_bin/wavAmplify_random"; ##/work/asrtrans/qrwang2/bins/2.fea_fa/wavAmplify_random

my @cmd_map;
my @cmd_red;
my @files;
my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

@cmd_map = (
::CmdToLocal("$bin_selecttail wav mlf_sy mlf_fa_ph"),
::CmdToLocal("$bin_amprand wav out 0.3 0.05"),
::CmdToLocal("$bin_selecttail out mlf_sy mlf_fa_ph"),
::CmdToLocal("$bin_renametail out wav"),
::CmdToLocal("$bin_randname"),
);

@cmd_red = (
::CmdToLocal("$bin_randnamered"),
);

@files   = (
$bin_amprand,
$bin_renametail,
$bin_selecttail,
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
."-Dmapreduce.reduce.memory.mb=3000 "
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
