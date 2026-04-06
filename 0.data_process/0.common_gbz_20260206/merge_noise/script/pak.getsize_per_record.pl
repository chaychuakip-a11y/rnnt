use strict;

use lib "/work1/asrdictt/taoyu/sbin";
use share_hadoop;

my $bin_stream     = "/work/asrdictt/taoyu/bin/streamingAC-2.5.0.jar";
my $bin_getsize    = "/work/asrdictt/taoyu/bin/getsize";
my $bin_getsizered = "/work/asrdictt/taoyu/bin/getsizered";
my $bin_getsize_per_record = "/work1/asrdictt/taoyu/bin/getsize_per_record";
my $bin_getsize_per_record_red = "/work1/asrdictt/taoyu/bin/getsize_per_record_red";

my $jobqueue     = "asr_dictt";
my $jobname      = "getsize";

my $num_reduce   = 1;
my $in_blocksize = 256*1024*1024;
my $block_size   = 64*1024*1024;

foreach(1){

my @src_label    = (
	# "noise_pingwen_baizao_mix_all",
	# "2.700h_pure_music_data",	##my $in_blocksize = 2048*1024*1024;
	# "NoiseGS.addKTV",
	# "noise_duodian_mix_all",
	# "noise_car_byd_rm3-4",
	"noise_dz_x3x4_4Car10Scene",
);

my @hdir_src     = (
	# "/workdir/asrdictt/gasrdictt/zhyou2/mlg/202311_car/noises/noise_pingwen_baizao_mix_all.pak/iflytek-20231116-part-00000",
	# "/workdir/asrdictt/dasrdictt/taoyu/noisedata/16k/2.700h_pure_music_data.pak",
	# "/workdir/asrdictt/dasrdictt/taoyu/noisedata/16k/NoiseGS.addKTV.pcm.index.scp.tmp.pak.1",
	# "/workdir/asrdictt/gasrdictt/zhyou2/mlg/202311_car/noises/noise_duodian_mix_all.pak/iflytek-20231116-part-00000",
	# "/workdir/asrdictt/gasrdictt/zhyou2/mlg/202311_car/noises/noise_car_byd_rm3-4.pak/iflytek-20231116-part-00000",
	"/workdir/asrdictt/dasrdictt/taoyu/noisedata/chezai/noise_dz_x3x4_4Car10Scene/iflytek-20180521-part-00000",
);

@hdir_src == @src_label || die "Error: count mismatch";

my $hdir_out     = "/workdir/asrdictt/tasrdictt/zhyou2/Test/checkedMlfwithDict/src_wav_getsize";
my $hdir_src;

my @cmd_map;
my @cmd_red;
my @files;

my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

mkdir "log" if(!-e "log");
# open(OUT, ">", "getsize.log") || die $!;
# printf OUT "%30s\t%10s\t%15s\t%10s\t%s\n", "label", "nRecord", "t(second)", "t(hour)", "hdir_src";

foreach my $i(0..$#src_label)
{
	my $hdir_src_cur = $hdir_src[$i];
	my $hdir_out_cur = $hdir_out."/$src_label[$i]";
	my $jobname_cur = $jobname."_$src_label[$i]";

	if(!-e "log/getsize.$src_label[$i].pak" || !-s "log/getsize.$src_label[$i].pak")
	{
		::RemoveHadoopDirIfExist($hdir_out_cur);

		@cmd_map = (
		# ::CmdToLocal("$bin_getsize wav 0 32000"),
		#::CmdToLocal("$bin_getsize wav 0 16000"),	#8k
		#::CmdToLocal("$bin_getsize fbp3 0 30000"),
		::CmdToLocal("$bin_getsize_per_record wav 0 32000 f_size"),
		);

		@cmd_red = (
		# ::CmdToLocal("$bin_getsizered"),
		# ::CmdToLocal("$bin_getsize_per_record_red size f_size_stat"),
		);

		@files   = (
		$bin_getsize,
		$bin_getsizered,
		$bin_getsize_per_record,
		$bin_getsize_per_record_red,
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
		#."-Dmapreduce.map.java.opts=\"-Xmx36000m\" "
		#."-Dmapreduce.reduce.java.opts=\"-Xmx4096m\" "
		#."-Dmapreduce.map.cpu.vcores=$thread_nums "
		#."-Dmapreduce.map.failures.maxpercent=20 "
		."-Dmapreduce.job.queuename=$jobqueue "
		."-Dmapreduce.job.name=$jobname_cur "
		."-Dmapreduce.job.reduces=$num_reduce "
		."-Dmapreduce.map.memory.mb=1500 "
		."-Dmapreduce.reduce.memory.mb=1500 "
		."-Ddc.input.block.size=$in_blocksize "
		."-Ddfs.blocksize=$block_size "
		."-Ddfs.replication=1 "
		."-files \"$files\" "
		."-input $hdir_src_cur "
		."-output $hdir_out_cur "
		;

		$cmd .= "-mapper \"$cmd_map\" " if(@cmd_map > 0);
		$cmd .= "-reducer \"$cmd_red\" " if(@cmd_red > 0);

		::PR($cmd);
		::SuccessOrDie("$hdir_out_cur");

		$cmd = "hdfs dfs -cat $hdir_out_cur/*part* | /work1/asrdictt/taoyu/bin/fea_lab_lat_unpack_1 - log/$src_label[$i] f_size";
		::PR($cmd);
	}
	# open(IN, "log/getsize.$src_label[$i].log") || die "$!, $src_label[$i]";
	# my $in_ = <IN>; chomp $in_;
	# $in_ =~ /Count of records: (\d+)/ || die "Error: informal line: $_\n";
	# my $nRecord = $1;

	# $in_ = <IN>; chomp $in_;
	# $in_ =~ /Total size: ([\d\.]+) seconds, or ([\d\.]+) hours./ || die "Error: informal line: $_\n";
	# my ($t_second, $t_hour) = ($1, $2);
	# close IN;
	# printf OUT "% 30s\t%10d\t%15.3f\t%10.3f\t%s\n", $src_label[$i], $nRecord, $t_second, $t_hour, $hdir_src_cur;
}

}
