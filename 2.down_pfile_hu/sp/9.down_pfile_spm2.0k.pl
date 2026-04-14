use strict;
use lib "/work1/asrdictt/hjwang11/aBak/clwu4/sbin";
use share_hadoop;

@ARGV >= 1 || die "usage: perl 9.down_pfile_spm2.0k.pl <split_id>\n";

for (my $ii = 0; $ii < 1; $ii = $ii + 1)
{
    my $num = $ARGV[0];

    ## 输出目录  ### set
    my $dir_lib = "PLACEHOLDER_HU_PFILE_OUT_DIR";
    system("mkdir -p $dir_lib") if (!-e $dir_lib);

    # 工具路径
    my $bin_qnfiletransfer = "/work1/asrdictt/hjwang11/aBak/clwu4/qnfiletranser_subword_encdec.bin";
    my $bin_qnnorm         = "/work1/asrdictt/hjwang11/aBak/clwu4/qnnorm";
    my $bin_cmvn           = "/work1/asrdictt/hjwang11/aBak/clwu4/cmvn_simple";
    my $bin_rm_bad         = "/work1/asrdictt/hjwang11/aBak/clwu4/pak_rm_bad_label.bin";
    my $selecttail         = "/work1/asrdictt/taoyu/bin/selecttail";
    my $addtail            = "/work1/asrdictt/taoyu/bin/addtail";

    # SPM MLF 文件：经过 1.get_mlf_sp_hu/spm_encode.sh 生成的 out.mlf_sy.mlf_sp
    # 单文件供所有 part 使用（addtail 按 utterance ID 匹配，无需分 part）  ### set
    my $mlf_sp_file = "PLACEHOLDER_HU_SPM_MLF";
    # !! 指向 4.cat_mlf_sp.sh 合并后的总文件，不是 per-part 的 sent.mlf_sp !!
    # 示例：/raw15/asrdictt/permanent/tyliu23/hu/rnnt/1.get_mlf_sp_hu/out.mlf_sy.mlf.danzi.mlf_sp

    # SPM 词表文件（由 spm_hu_bpe_2000.vocab 生成的 states_list 格式）  ### set
    my $states_list = "PLACEHOLDER_HU_SPM_STATES_LIST";
    # 示例：/raw15/asrdictt/permanent/tyliu23/hu/rnnt/spm/states_list.hu.spm2.0k

    # SPM map 文件（piece -> id 映射，.map 格式）  ### set
    my $map_list = "PLACEHOLDER_HU_SPM_MAP_LIST";
    # 示例：/raw15/asrdictt/permanent/tyliu23/hu/rnnt/spm/states_list.hu.spm2.0k.map

    my $descript_fea     = "fbnocmn40";
    my $descript_lab     = "mlf_sy";
    my $descript_lab_sp  = "mlf_sp";

    my $split_id  = $num;
    my $file_norm = "$dir_lib/fea.norm.$split_id";
    my $pfile_fea = "$dir_lib/fea.pfile.$split_id";
    my $pfile_lab = "$dir_lib/lab.pfile.$split_id";
    my $scp_lab   = "$dir_lib/lab.scp.$split_id";
    my $trans_log = "$dir_lib/trans.log.$split_id";

    # FA HDFS 输出（与 1_dnnfa.pl 中 $hdir_out 一致，按 split_id 取对应 part）
    my $inputdir = "/workdir/asrdictt/dasrdictt/tyliu23/hu/dnnfa/tongyong/hungarian_wav_dnnfa/*$split_id";

    if (!-e "$pfile_fea.done.finish")
    {
        system("touch $pfile_fea.done.start");
        my $cmd = "hdfs dfs -cat $inputdir"
                . " |$selecttail $descript_fea"
                . " | $addtail $mlf_sp_file $descript_lab_sp"
                . " | $bin_qnfiletransfer $map_list $states_list $descript_lab_sp"
                . " $pfile_lab $descript_fea 40 - $scp_lab 0"
                . " >$trans_log 2>&1";

        print $cmd . "\n";
        system($cmd);
        !system("$bin_qnnorm norm_ftrfile=$pfile_fea output_normfile=$file_norm")
            || die "qnnorm failed: $file_norm.\n";
        system("touch $pfile_fea.done.finish");
    }
}
