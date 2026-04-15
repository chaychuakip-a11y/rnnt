use strict;

# 用法：perl 15.pfile_paste.pl <sp_dir> <ce_dir> <split_id>
# 将 SP/BPE 标签（列1）和 CE phone 标签（列2）合并为 2列 pfile，输出到 ce_dir/mix/

my ($dir_lib_sp, $dir_lib_ce, $split_id) = @ARGV;

my $nSplit  = 100;
my $dir_mix = "$dir_lib_ce/mix/";

my $bin_pfile_paste = "/work2/asrdictt/xxtong/workbak_xxtong/tools/bin/pfile_paste";

$split_id < $nSplit || die "Error: split_id $split_id out of range\n";

system("mkdir -p $dir_mix") if (!-e $dir_mix);

my $i = $split_id;
{
    my $out = "$dir_mix/lab.pfile$i";
    my $in1 = "$dir_lib_sp/lab.pfile$i";    # SP BPE 标签（经 rand_by_index 对齐后）
    my $in2 = "$dir_lib_ce/lab.pfile$i";    # CE phone 标签

    warn("$bin_pfile_paste $out $in1 $in2");
    system("$bin_pfile_paste $out $in1 $in2");
}
