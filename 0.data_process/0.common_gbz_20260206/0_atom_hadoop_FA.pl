use strict;
use lib "/work/asrdictt/taoyu/sbin";
use share_hadoop;
my $jobqueue			= $ARGV[0]; # Here Set The Job Queue Of Your Group
my $jobname			= $ARGV[1]; # Here Set Job Name
my $reduces_num			= 100;
my $input_hdfs_block_size	= 256 * 1024 * 1024;
my $hdfs_block_size		= 64 * 1024 * 1024;
my @hdir_srcs     = (
$ARGV[2]
); # Here Set Your Input Path On Hdfs


my $input_dir = join(" -input ", @hdir_srcs);

my $output_dir		    = $ARGV[3]; # Here Set Your Output Path On Hdfs
system("hdfs dfs -rm -r $output_dir");
my $bin_dir		  = "/train8/asrmlg/ddye2/tools_new/atom-v20151016b"; # Here Set Your Local Bin Folder

##input resource
my $res_raw = "./mle_yingchen15";
####my $local_states_list = "/states.list";  # Set
my $src_hmmlist = "$res_raw/hmmlist.final";# Set
my $src_acmod_bin = "$res_raw/atom_acmod_dnn_c.bin"; # Set
my $mlp_mod_local = "$res_raw/mlp.9.wts"; # Set
my $src_states_count = "$res_raw/state.input.dnnfa.txt"; # Set
my $fea_norm_local = "$res_raw/fea.norm"; # Set

# my $fst_bin_local = "$res_raw/fst.bin"; # Set
my $fst_bin_local = "/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/french/data_prepare/gen_fst_bin/atom_FA_french_ly1213h_ly2k_xw696h_openSRC412h_fz1655h_ttsMusicVideoPoi_v16.word_dict_phone_nosp.add_pred_1gram/fst.bin";

my $streaming			= "$bin_dir/streamingAC-2.5.0.jar"; # Here Set Your Streaming Tool

my $MLP_thread_nums   = 1;  #advise: set 1 if DNN MLP, 1-8 if RNN MLP
my $DEC_thread_nums   = 4;  #advise: set 1-4 if running on hadoop, 1-12 if running at local
my $thread_nums = $MLP_thread_nums + $DEC_thread_nums;
my $res_dir 			= "res"; # Here Set Your Local Res Folder
my $hmmlist           = "$res_dir/hmmlist.final";
my $acmod_bin         = "$res_dir/atom_acmod_dnn_c.bin";
my $states_count      = "$res_dir/state.input.dnnfa.txt"; 
my $fst_bin           = "$res_dir/fst.bin";
my $mlp_res_bin = "$res_dir/mlp.9.wts";
my $fea_norm = "$res_dir/fea.norm";
system("mkdir -p $res_dir");
system("cp $src_hmmlist $hmmlist");
system("cp $src_acmod_bin $acmod_bin");
system("cp $fea_norm_local $fea_norm");
system("cp $mlp_mod_local $mlp_res_bin");
system("cp $fst_bin_local $fst_bin");
system("cp $src_states_count $states_count" );

#+++++++++++++++++++++++++++++++++++++++++
my $cmd_map;
my $cmd_red;
my $dir_bin      = "/train8/asrmlg/ddye2/tools_new";
my $bin_cmvn     = "$dir_bin/htk-0.1.4/bin/cmvn_simple";
my $bin_selecttail = "$dir_bin/selecttail";
my $bin_randname    = "$dir_bin/randname";
my $bin_randnamered = "$dir_bin/randnamered";
my $bin_raw_fea    = "$dir_bin/htk-0.1.4/bin/raw_fea";
my $bin_fep_decoder    = "$dir_bin/atom_ublstm/fep_decoder_lattice_test";
my $config1            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB24_0_D_A_P3";
my $config2            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerMFCC_0_D_A_P3";
my $config3            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB24_0_D_A";
my $config4            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB40";
my $config            = "./atom_hadoop.cfg";
my $config5            = "raw_fea.config";
##decode parameter
my $mlp_thread_nums   = 1;  #advise: set 1 if DNN MLP, 1-8 if RNN MLP
my $dec_thread_nums   = 1;  #advise: set 1-4 if running on hadoop, 1-12 if running at local
my $thread_nums       = $mlp_thread_nums + $dec_thread_nums;

my $bin_renametail    = "/work1/asrdictt/taoyu/bin/renametail";
my $bin_addtail        = "/work1/asrdictt/taoyu/bin/addtail";

my $file_mlf_sy		="/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/french/data_prepare/check_oov/mlf_zx/lab.mlf_sy.nopunc";


$cmd_map = join(" | ", (
::CmdToLocal("$bin_selecttail wav"),
::CmdToLocal("$bin_addtail $file_mlf_sy mlf_sy"),
# ::CmdToLocal("$bin_renametail mlf_ori mlf_sy"),
#::CmdToLocal("$checkMLF mlf_sy $dict"),
#::CmdToLocal("$bin_selecttail fbp3 fbnocmn40 fb40 mlf_sy"),
::CmdToLocal("$bin_raw_fea $config5 fea"),
::CmdToLocal("$bin_cmvn 2 24 1 fea"),
#::CmdToLocal("$bin_raw_fea $config4 fbnocmn40"),
#::CmdToLocal("$bin_raw_fea $config4 fb40"),
#::CmdToLocal("$bin_cmvn 2 40 1 fb40"),
::CmdToLocal("$bin_fep_decoder -c $config -mtn $mlp_thread_nums -dtn $dec_thread_nums"),
#::CmdToLocal("$bin_selecttail wav mlf_sy fbnocmn40 fb40 mlf_fa_ph"),
::CmdToLocal("$bin_selecttail wav mlf_sy fbnocmn40 mlf_fa_ph"),
::CmdToLocal("$bin_randname"),
));

$cmd_red = ::CmdToLocal("$bin_randnamered");

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
		."-Dmapreduce.map.cpu.vcores=$thread_nums "  
		."-Ddc.input.block.size=$input_hdfs_block_size "
		."-Ddfs.block.size=$hdfs_block_size "
		."-numReduceTasks $reduces_num "
		."-input $input_dir " # Here You Can Use Multiple "-input" To Set Multiple Input Path
		."-output $output_dir " 
		#."-mapper \"./atom -c atom_hadoop.cfg -mtn $MLP_thread_nums -dtn $DEC_thread_nums\" "

		."-mapper \"$cmd_map\" "
		."-reducer \"$cmd_red\" "
		."-file \"$config5\" "
		."-file \"$bin_raw_fea\" "
		."-file \"$bin_cmvn\" "
		."-file \"$bin_fep_decoder\" "
		."-file \"$bin_selecttail\" "
		."-file \"$bin_randname\" "
		."-file \"$bin_randnamered\" "
		."-file \"mapper.$jobname.sh\" "

		."-file \"$bin_dir/atom\" "
		."-file \"$bin_dir/lattice.so\" "
		."-file \"$bin_dir/decoder.so\" "
		."-file \"$bin_dir/libboost_thread.so.1.46.1\" "
		."-file \"atom_hadoop.cfg\" " # Res Name Must Be The Same As Those In Config
		."-file \"$fst_bin\" "
		."-file \"$hmmlist\" "
		."-file \"$mlp_res_bin\" "
		."-file \"$fea_norm\" "
		."-file \"$states_count\" "
		."-file \"$acmod_bin\" "
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
