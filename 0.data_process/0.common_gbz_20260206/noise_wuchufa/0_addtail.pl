use strict;
use lib "/work/asrdictt/taoyu/sbin";
use share_hadoop;
my $jobname			= "addtail"; # Here Set Job Name
my $jobqueue			= "asr_dictt"; # Here Set The Job Queue Of Your Group
my $reduces_num			= 100;
my $input_hdfs_block_size	= 256 * 1024 * 1024;
my $hdfs_block_size		= 64 * 1024 * 1024;
my @hdir_srcs     = (
"/workdir/asrdictt/dasrdictt/xxtong/DC/Dichan_2022/NOISE/noise_zong/*",
); # Here Set Your Input Path On Hdfs

my $bin_dir		  = "/train8/asrmlg/ddye2/tools_new/atom-v20151016b"; # Here Set Your Local Bin Folder
my $streaming			= "$bin_dir/streamingAC-2.5.0.jar"; # Here Set Your Streaming Tool

my $input_dir = join(" -input ", @hdir_srcs);

my $local_output = "noise_wuchufa_3states";
my $output_dir		    = "/workdir/asrdictt/gasrdictt/zhyou2/mlg/202311_car/$local_output"; # Here Set Your Output Path On Hdfs
#my $output_dir    = "/workdir/asrmlg/dasrmlg/ddye2/gmkws/japanese_local/dnnfa/$local_output"; # Here Set Your Output Path On Hdfs
system("hdfs dfs -rm -r $output_dir");

my $dir_bin 		= "/work1/asrdictt/taoyu/bin";
my $bin_renametail    = "$dir_bin/renametail";
my $bin_addtail        = "$dir_bin/addtail";
my $bin_selecttail = "$dir_bin/selecttail";
my $bin_randname    = "$dir_bin/randname";
my $bin_randnamered = "$dir_bin/randnamered";

my $file_mlf_sy		="/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/french/0.data_prepare/noise_wuchufa/out.mlf_fa_ph.3states";

my $cmd_map = join(" | ", (
::CmdToLocal("$bin_renametail mlf_fa_ph mlf_fa_s1"),
::CmdToLocal("$bin_addtail $file_mlf_sy mlf_fa_ph"),
::CmdToLocal("$bin_randname"),
));

my $cmd_red = ::CmdToLocal("$bin_randnamered");

open(OUT, ">", "./mapper.$jobname.sh") || die $!;
print OUT "$cmd_map\n";
close OUT;
$cmd_map = "bash ./mapper.$jobname.sh";
#+++++++++++++++++++++++++++++++++++++++++


my $hadoop_cmd = "hadoop jar $streaming "
		."-Dmapred.child.java.opts=\"-Xmx4096m\" "
		."-Dmapreduce.map.java.opts=\"-Xmx4096m\" "
		."-Dmapred.job.queue.name=$jobqueue "
		."-Dmapred.job.name=$jobname "
		."-Dmapreduce.map.memory.mb=40000 " # Here You Can Set Map Memory. when use multiple threads,this should be set more
		."-Ddc.input.block.size=$input_hdfs_block_size "
		."-Ddfs.block.size=$hdfs_block_size "
		."-numReduceTasks $reduces_num "
		."-input $input_dir " # Here You Can Use Multiple "-input" To Set Multiple Input Path
		."-output $output_dir " 
		#."-mapper \"./atom -c atom_hadoop.cfg -mtn $MLP_thread_nums -dtn $DEC_thread_nums\" "

		."-mapper \"$cmd_map\" "
		."-reducer \"$cmd_red\" "
		."-file \"$bin_selecttail\" "
		."-file \"$bin_randname\" "
		."-file \"$bin_randnamered\" "
		."-file \"mapper.$jobname.sh\" "

		."-file \"$bin_renametail\" "
		."-file \"$bin_addtail\" "
		."-file \"$file_mlf_sy\" ";



!system("$hadoop_cmd") or die;

#system("hdfs dfs -cat $output_dir/*part* | ./get_fa_mlf > result_600H_DNN_fa.mlf");
# system("hdfs dfs -cat $output_dir/*part* | $bin_dir/qnfiletransfer_1 - fea fea.$local_pfile_name.pfile mlf_fa_ph lab.$local_pfile_name.pfile $local_states_list lab.$local_pfile_name.scp");


###############################################################################################################
############################################Resource Matrix####################################################
#		mlp.bin		acmod.bin	fst.bin		wfst.bin        fst.bin	        hmmlist.final
#DNN/BN-FA	yes		yes		yes					
#GMM-FA				yes		yes					
#DNN/BN-lattice	yes		yes		yes		yes		yes		yes
#GMM-lattice			yes		yes		yes		yes             yes
#DNN-BN-decode	yes		yes				yes			
#GMM-decode			yes				yes
###############################################################################################################
