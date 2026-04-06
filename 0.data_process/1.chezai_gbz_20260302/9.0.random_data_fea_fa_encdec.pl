use strict;

use lib "/work1/asrdictt/hjwang11/sbin";
use share_hadoop;

my $jobname           = "9.0.random_data_fea_fa";#SET
my $jobqueue          = "asr_dictt";#SET
my $num_reduce        = 100;
my $in_blocksize      = 256*1024*1024;
my $block_size        = 64*1024*1024;
my $replication       = 2;

my @hdir_src = (
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand", ###原始数据
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_addnoise_rm_head_44bytes/*", ###加噪数据
"/workdir/asrdictt/tasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_0.2_speedup1.2_fa", ###原始数据变速
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_0.2_ampReduce", ###原始数据小音量
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_addnoise_0.2_lsa_fa", ###加噪数据lsa降噪
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_addnoise_0.1_mae_open_fa", ###加噪数据mae_open降噪
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_addnoise_0.1_mae_close_fa", ###加噪数据mae_close降噪
);
my $hdir_out          = "/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_speedup1.2_ampReduce_lsa_mae_fea_fa_encdec";
my $hdir_src          = join(" -input ", @hdir_src);

##input resource  fa
my $config            = "./atom_hadoop.cfg";

my $dir_lm            = "/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/am/0.data_process/0.common_gbz_20260206/mle_yingchen15";
my $fst_bin           = "/yrfs4/asrdictt/hjwang11/multilingual/indonesian/rnnt/dict/indonesian_20260113.dict.add_trainset_oov_pred_1gram/fst.bin";
#my $wfst_bin          = "$dir_lm/atom_wfst.bin";
#my $G_fst             = "$dir_lm/G.fst";

my $hmmlist           = "$dir_lm/hmmlist.final"; ###/ps/asrtrans/zpxie2/Data_from_lustre2/jobs/En_native/BN_MPE_FA/mpe_3kh_bn_NativeEN/hmmlist
my $acmod_bin         = "$dir_lm/atom_acmod_indonesian"; ###/ps/asrtrans/zpxie2/Data_from_lustre2/jobs/En_native/BN_MPE_ublstm_3k3H/atom_acmod.dnn.bin

my $wts_file          = "$dir_lm/mlp.9.wts";  ###/ps/asrtrans/zpxie2/Data_from_lustre2/jobs/preprocess_data/1_Native_English/Native_20180108_FYJ_208h/fea_FA/uBLSTMP.iter0002.9.mat
my $fea_norm          = "$dir_lm/fea_hmm_fa_indonesian_txwg_300h.norm";  ###/ps/asrtrans/zpxie2/Data_from_lustre2/jobs/preprocess_data/1_Native_English/Native_20180108_FYJ_208h/fea_FA/fea.norm
my $state_count       = "$dir_lm/state.indonesian_hmm.txt";  ###/ps/asrtrans/zpxie2/Data_from_lustre2/jobs/preprocess_data/1_Native_English/Native_20180108_FYJ_208h/fea_FA/states.count.txt

##decode parameter
my $mlp_thread_nums   = 1;  #advise: set 1 if DNN MLP, 1-8 if RNN MLP
my $dec_thread_nums   = 1;  #advise: set 1-4 if running on hadoop, 1-12 if running at local
my $thread_nums       = $mlp_thread_nums + $dec_thread_nums;

##tools
my $dir_bin            = "/work1/asrdictt/hjwang11/bin";
my $bin_stream         = "$dir_bin/streamingAC-2.5.0.jar";

my $bin_selecttail     = "$dir_bin/selecttail";
my $bin_renametail     = "$dir_bin/renametail";
my $bin_randname       = "$dir_bin/randname";
my $bin_randnamered    = "$dir_bin/randnamered";
my $checkMLF           = "$dir_bin/checkMlfWithDict";

my $sbin_CreateConfig  = "$dir_bin/atom_ublstm/CreateDecConfig.pl";
my $bin_fep_decoder    = "$dir_bin/atom_ublstm/fep_decoder_lattice_test";
my $bin_libboost_so    = "$dir_bin/atom_ublstm/libboost_thread.so.1.46.1";
my $bin_libboost_thread_so = "$dir_bin/atom_ublstm/libboost_thread.so.1.58.0";
my $bin_libboost_system_so = "$dir_bin/atom_ublstm/libboost_system.so.1.58.0";
my $bin_lattice_so     = "$dir_bin/atom_ublstm/lattice.so";
my $bin_decode_so      = "$dir_bin/atom_ublstm/decoder.so";
my $bin_ulm_so         = "$dir_bin/atom_ublstm/rescore/ulmc.so";
my $bin_lm_so          = "$dir_bin/atom_ublstm/rescore/LMTrie_F.so";
my $bin_res_so         = "$dir_bin/atom_ublstm/rescore/ResMgr.so";

my $bin_cmvn           = "$dir_bin/htk-0.1.4/bin/cmvn_simple";
my $bin_raw_fea        = "$dir_bin/htk-0.1.4/bin/raw_fea";
my $config1            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB24_0_D_A_P3";
my $config2            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerMFCC_0_D_A_P3";
my $config3            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB24_0_D_A";
my $config4            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB40";

my $bin_enc            = "$dir_bin/speexenc";
my $bin_enc_res0       = "$dir_bin/libspeex.so.1.5.0";
my $bin_enc_res1       = "$dir_bin/libspeex.so.1";

my @cmd_map;
my @cmd_red;
my @files;
my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

@cmd_map = (
::CmdToLocal("$bin_selecttail wav mlf_sy mlf_fa_ph"),
::CmdToLocal("$bin_enc wav encdec 10"),
::CmdToLocal("$bin_selecttail encdec mlf_sy mlf_fa_ph"),
::CmdToLocal("$bin_renametail encdec wav"),
#::CmdToLocal("$bin_raw_fea $config4 fbnocmn40"),
::CmdToLocal("$bin_raw_fea $config4 fb40"),
::CmdToLocal("$bin_cmvn 2 40 1 fb40"),
::CmdToLocal("$bin_selecttail fb40 mlf_sy mlf_fa_ph"),
::CmdToLocal("$bin_randname"),
);

@cmd_red = (
::CmdToLocal("$bin_randnamered"),
);

@files   = (
$bin_enc,
$bin_enc_res0,
$bin_enc_res1,
$config,
$acmod_bin,
$wts_file,
$fea_norm,
$state_count,
$bin_fep_decoder,
$bin_libboost_so,
$bin_libboost_thread_so,
$bin_libboost_system_so,
$bin_lattice_so,
$bin_decode_so,
$bin_ulm_so,
$bin_lm_so,
$bin_res_so,
$fst_bin,
#$wfst_bin,
#$G_fst,
$hmmlist,
$bin_cmvn,
$bin_raw_fea,
$config1,
$config2,
$config3,
$config4,
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
."-Dmapreduce.map.cpu.vcores=$thread_nums "
."-Dmapreduce.map.failures.maxpercent=20 "
."-Dmapreduce.job.queuename=$jobqueue "
."-Dmapreduce.job.name=$jobname "
."-Dmapreduce.job.reduces=$num_reduce "
."-Dmapreduce.map.memory.mb=12000 "
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

