use strict;

use lib "/work1/asrdictt/hjwang11/sbin";
use share_hadoop;

my $jobname           = "7.0.MaeDenoiseOpen";#SET
my $jobqueue          = "asr_dictt";#SET
my $num_reduce        = 100; #SET
my $in_blocksize      = 256*1024*1024;
my $block_size        = 64*1024*1024;
my $replication       = 2;

my @hdir_src = (
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise_rm_head_44bytes/*8",
);
my $hdir_out          = "/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise_0.1_mae_open";
my $hdir_src          = join(" -input ", @hdir_src);

my $dir_bin           = "/work1/asrdictt/hjwang11/bin";
my $bin_randname      = "$dir_bin/randname";
my $bin_randnamered   = "$dir_bin/randnamered";

my $bin_stream        = "$dir_bin/streamingAC-2.5.0.jar";
my $bin_selecttail    = "$dir_bin/selecttail";
my $bin_renametail    = "$dir_bin/renametail";

##### MAE process
my $bin_amp           = "$dir_bin/wavAmplify";#"/lustre2/asr/yjjiang/jst-data/tools/wavAmplify/Debug/wavAmplify"
my $noisedataL        = "/raw15/asrdictt/permanent/taoyu/ps/mae/7.5kh_srfmae/noise_L.scp.tmp.pak.1";#"/lustre2/asr/yjjiang/jst-data/jobs/for_lishang/NOCMN-model/chezai_local_ce/Project1_chnengphone/nocmn/data_process/7.5kh_srfmae/noise_L.scp.tmp.pak.1";#IN
my $noisedataR        = "/raw15/asrdictt/permanent/taoyu/ps/mae/7.5kh_srfmae/noise_R.scp.tmp.pak.1";#"/lustre2/asr/yjjiang/jst-data/jobs/for_lishang/NOCMN-model/chezai_local_ce/Project1_chnengphone/nocmn/data_process/7.5kh_srfmae/noise_R.scp.tmp.pak.1";#IN

my $RIRX              = "/raw15/asrdictt/permanent/taoyu/ps/mae/RIRs/jiashi_open.txt";#"/lustre1/embed/lishang/lishang/jst-data/code/mt_mae/mt_mae/Debug/RIRs/jiashi_open.txt";
my $bin_mt_MAE        = "$dir_bin/mt_mae";                       #"/ps/asr/taoyu/code/mt_mae/mt_mae/Debug/mt_mae";

##### feature extraction
my $bin_raw_fea       = "$dir_bin/htk-0.1.4/bin/raw_fea";
my $bin_cmvn          = "$dir_bin/htk-0.1.4/bin/cmvn_simple";
my $config1           = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB24_0_D_A_P3";
my $config2           = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerMFCC_0_D_A_P3";
my $config3           = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB24_0_D_A";
my $config4           = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB40";
my $config5            = "$dir_bin/raw_fea.config";

##### fa
my $config            = "./atom_hadoop.cfg";

my $dir_lm            = "/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/am/0.data_process/0.common_gbz_20260206/mle_yingchen15";
my $fst_bin           = "/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/am/0.data_process/1.chezai_gbz_20260302/gen_fst_bin/atom_FA_french_ly1213h_ly2k_xw696h_openSRC412h_fz1655h_ttsMusicVideoPoi_v16.word_dict_phone_nosp.add_pred_20260302_1gram/fst.bin";
#my $wfst_bin          = "$dir_lm/atom_wfst.bin";
#my $G_fst             = "$dir_lm/G.fst";

my $hmmlist           = "$dir_lm/hmmlist.final"; ###/ps/asrtrans/zpxie2/Data_from_lustre2/jobs/En_native/BN_MPE_FA/mpe_3kh_bn_NativeEN/hmmlist
my $acmod_bin         = "$dir_lm/atom_acmod_dnn_c.bin"; ###/ps/asrtrans/zpxie2/Data_from_lustre2/jobs/En_native/BN_MPE_ublstm_3k3H/atom_acmod.dnn.bin
my $wts_file          = "$dir_lm/mlp.9.wts";  ###/ps/asrtrans/zpxie2/Data_from_lustre2/jobs/preprocess_data/1_Native_English/Native_20180108_FYJ_208h/fea_FA/uBLSTMP.iter0002.9.mat
my $fea_norm          = "$dir_lm/fea.norm";  ###/ps/asrtrans/zpxie2/Data_from_lustre2/jobs/preprocess_data/1_Native_English/Native_20180108_FYJ_208h/fea_FA/fea.norm
my $states_count       = "$dir_lm/state.input.dnnfa.txt";  ###/ps/asrtrans/zpxie2/Data_from_lustre2/jobs/preprocess_data/1_Native_English/Native_20180108_FYJ_208h/fea_FA/states.count.txt


my $sbin_CreateConfig = "$dir_bin/atom_ublstm/CreateDecConfig.pl";
my $bin_fep_decoder   = "$dir_bin/atom_ublstm/fep_decoder_lattice_test";
my $bin_libboost_so   = "$dir_bin/atom_ublstm/libboost_thread.so.1.46.1";
my $bin_libboost_thread_so = "$dir_bin/atom_ublstm/libboost_thread.so.1.58.0";
my $bin_libboost_system_so = "$dir_bin/atom_ublstm/libboost_system.so.1.58.0";
my $bin_lattice_so    = "$dir_bin/atom_ublstm/lattice.so";
my $bin_decode_so     = "$dir_bin/atom_ublstm/decoder.so";
my $bin_ulm_so        = "$dir_bin/atom_ublstm/rescore/ulmc.so";
my $bin_lm_so         = "$dir_bin/atom_ublstm/rescore/LMTrie_F.so";
my $bin_res_so        = "$dir_bin/atom_ublstm/rescore/ResMgr.so";

##decode parameter
my $mlp_thread_nums   = 1;  #advise: set 1 if DNN MLP, 1-8 if RNN MLP
my $dec_thread_nums   = 1;  #advise: set 1-4 if running on hadoop, 1-12 if running at local
my $thread_nums       = $mlp_thread_nums + $dec_thread_nums;

my @cmd_map;
my @cmd_red;
my @files;
my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

@cmd_map = (
::CmdToLocal("$bin_selecttail mlf_sy wav"),
::CmdToLocal("$bin_mt_MAE $RIRX 1.0 $noisedataL $noisedataR"),
# ::CmdToLocal("$bin_raw_fea $config5 fea"),
# ::CmdToLocal("$bin_cmvn 2 24 1 fea"),
# ::CmdToLocal("$bin_raw_fea $config4 fbnocmn40"),
# ::CmdToLocal("$bin_raw_fea $config4 fb40"),
# ::CmdToLocal("$bin_cmvn 2 40 1 fb40"),
# ::CmdToLocal("$bin_fep_decoder -c $config -mtn $mlp_thread_nums -dtn $dec_thread_nums"),
# ::CmdToLocal("$bin_selecttail fbnocmn40 fb40 mlf_sy mlf_fa_ph"),
::CmdToLocal("$bin_randname"),
);

@cmd_red = (
::CmdToLocal("$bin_randnamered"),
);

@files   = (
$bin_mt_MAE,
$RIRX,
$noisedataL,
$noisedataR,
$bin_amp,

$bin_raw_fea,
$bin_cmvn,
$config1,
$config2,
$config3,
$config4,
$config5,
$config,
$dir_lm,
$fst_bin,
#$wfst_bin,
#$G_fst,
$hmmlist,
$acmod_bin,
$wts_file,
$fea_norm,
$states_count,
$bin_fep_decoder,
$bin_libboost_so,
$bin_libboost_thread_so,
$bin_libboost_system_so,
$bin_lattice_so,
$bin_decode_so,
$bin_ulm_so,
$bin_lm_so,
$bin_res_so,

$bin_selecttail,
$bin_renametail,
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
."-Dmapreduce.map.memory.mb=12000 "
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
