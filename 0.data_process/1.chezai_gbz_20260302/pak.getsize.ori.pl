use strict;

use lib "/work1/asrdictt/hjwang11/sbin";
use share_hadoop;

###配置hadoop参数
my $jobqueue   = "adapt";###提交的队列
my $jobname    = "getsize";###提交的任务名
my $num_reduce = 1;###reduce的个数
my $in_blocksize      = 64*1024*1024;###输入的块大小
my $block_size        = 64*1024*1024;###输出的块大小

###配置输入
my @hdir_src     = (
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand",
);
my $hdir_src     = join(" -input ", @hdir_src);

###配置输出
my $src_label      = "french_chezai_gbz_20260302_rand";
my $hdir_out       = "/workdir/asrdictt/dasrdictt/hjwang11/src_wav_getsize";

###配置工具
my $bin_stream     = "/work1/asrdictt/hjwang11/bin/streamingAC-2.5.0.jar";
my $bin_getsize    = "/work1/asrdictt/hjwang11/bin/getsize";
my $bin_getsizered = "/work1/asrdictt/hjwang11/bin/getsizered";

my $cmd_map;
my $cmd_red;
my $cmd;


my $hdir_src_cur = $hdir_src;
my $hdir_out_cur = $hdir_out."/$src_label";
my $jobname_cur = $jobname."_$src_label";

###检查如果输出目录存在则删除
::RemoveHadoopDirIfExist($hdir_out_cur);

###hadoop命令
$cmd_map = ::CmdToLocal("$bin_getsize wav 0 32000");
$cmd_red = ::CmdToLocal("$bin_getsizered");

$cmd = "hadoop jar $bin_stream "
."-Dmapreduce.job.queuename=$jobqueue "
."-Dmapreduce.job.name=$jobname_cur "
."-Dmapreduce.job.reduces=$num_reduce "
."-Ddc.input.block.size=$in_blocksize "
."-Ddfs.blocksize=$block_size "
."-Ddfs.replication=1 "###数据备份数
#."-partitioner com.iflytek.hadoop.streaming.mapred.BalancePatitioner256 "
."-input $hdir_src_cur "
."-output $hdir_out_cur "
."-mapper \"$cmd_map\" "
."-reducer \"$cmd_red\" "
."-file $bin_getsize "
."-file $bin_getsizered "
;

###提交运行
::PR($cmd);
###检查运行是否成功
::SuccessOrDie("$hdir_out_cur");

###将结果从hdfs down到本地
$cmd = "hdfs dfs -cat $hdir_out_cur/*part* | $bin_getsizered >getsize.$src_label.pak 2>getsize.$src_label.log";
::PR($cmd);

