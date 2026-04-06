use strict;

use lib "/work1/asrdictt/taoyu/sbin";
use share_hadoop;

my $jobname      = "reverb";
my $jobqueue     = "nlp";
my $num_reduce   = 80;
my $in_blocksize = 512*1024*1024;
my $block_size   = 64*1024*1024;
my $replication  = 2;

my @hdir_src     = (
                    "/workdir/asrdictt/tasrdictt/taoyu/mlg/arabic/13kh_wav_fb40_dnnfa/*-part-000[6-7]*", ## 1/4;
                   );
my $hdir_out     = ("/workdir/asrdictt/tasrdictt/taoyu/mlg/arabic/13kh_wav_reverb_0.25.wav_dnnfa");
my $hdir_src     = join(" -input ", @hdir_src);

my $rirdata      = "hdfs://mycluster/workdir/asrdictt/dasrdictt/taoyu/reverb/h_real_and_syn_274/all_cut_h_2parts.index.scp.tmp.pak.1";#IN   274 types h including syn h and real h  #"hdfs://mycluster/workdir/asrdictt/gasrdictt/yjjiang/yongxuDATA/h_real_and_syn_274/all_cut_h_2parts.index.scp.tmp.pak.1"
my $noisedata    = "hdfs://mycluster/workdir/asrdictt/dasrdictt/taoyu/noisedata/16k/WhiteNoise.raw.index.scp.tmp.pak.1";#IN, not used here #"hdfs://mycluster/workdir/asrdictt/gasrdictt/tiangao/Dereverb/whitenoise/whitenoise.scp.tmp.pak.1";

my @ratio        = (1);#SET  how much training data you will use, multiply ratio

my $dir_bin           = "/work1/asrdictt/taoyu/bin";
my $bin_stream        = "$dir_bin/streamingAC-2.5.0.jar";
my $bin_reverb        = "$dir_bin/Conv_Timerand"; #/work/asrdictt/yjjiang/jst-data/tools/Conv_Timerand/Debug/Conv_Timerand
my $bin_selecttail    = "$dir_bin/selecttail";

my $bin_randname      = "$dir_bin/randname";
my $bin_randnamered   = "$dir_bin/randnamered";

my @cmd_map;
my @cmd_red;
my @files;
my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

foreach my $i(0)
{
	my $j = $i + 1;
	my $jobname_cur  = $jobname;
	my $hdir_out_cur = $hdir_out;
	my $seed_cur     = 0;
	my $ratio_cur    = $ratio[$i];

	@cmd_map = (
	::CmdToLocal("$bin_selecttail wav mlf_sy mlf_fa_ph"),
	::CmdToLocal("$bin_reverb -h $rirdata -n $noisedata -r $seed_cur -multiple $ratio_cur"),
	::CmdToLocal("$bin_randname"),
	);

	@cmd_red = (
	::CmdToLocal("$bin_randnamered"),
	);

	@files   = (
	$bin_reverb,
	$bin_selecttail,
	$noisedata,
	$rirdata,
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
	."-Dmapreduce.map.java.opts=\"-Xmx72000m\" "
	."-Dmapreduce.reduce.java.opts=\"-Xmx4096m\" "
	#."-Dmapreduce.map.cpu.vcores=$thread_nums "
	."-Dmapreduce.map.failures.maxpercent=20 "
	."-Dmapreduce.job.queuename=$jobqueue "
	."-Dmapreduce.job.name=$jobname_cur "
	."-Dmapreduce.job.reduces=$num_reduce "
	."-Dmapreduce.map.memory.mb=20000 "
	."-Dmapreduce.reduce.memory.mb=3000 "
	."-Ddc.input.block.size=$in_blocksize "
	."-Ddfs.blocksize=$block_size "
	."-Ddfs.replication=$replication "
	."-files \"$files\" "
	."-input $hdir_src "
	."-output $hdir_out_cur "
	;

	$cmd .= "-mapper \"$cmd_map\" " if(@cmd_map > 0);
	$cmd .= "-reducer \"$cmd_red\" " if(@cmd_red > 0);

	::RemoveHadoopDirIfExist($hdir_out_cur);
	::PR($cmd);
	::SuccessOrDie("$hdir_out_cur");
}
