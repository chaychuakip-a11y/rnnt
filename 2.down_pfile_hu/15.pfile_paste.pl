use strict;

# 用法：perl 15.pfile_paste.pl <dir_pwd> <split_id>
# 将 CE 标签（列1）和 SP/BPE 标签（列2）合并为 2列 pfile，输出到 mix/

my ($dir, $split_id) = @ARGV;

my $nSplit = 100;
my $dir_lib_ce  = "$dir/lib_fb40";          # CE pfile 目录
my $dir_lib_sp  = "$dir";                   # SP pfile 目录（lab.pfile.N）
my $dir_mix     = "$dir/lib_fb40/mix/";     # 合并输出目录

my $bin_pfile_paste = "/work2/asrdictt/xxtong/workbak_xxtong/tools/bin/pfile_paste";

foreach my $split_id ($split_id)
{
    $split_id < $nSplit || die "Error: split_id $split_id out of range\n";
}

system("mkdir -p $dir_mix") if (!-e $dir_mix);

my $i = $split_id;
{
    my $out  = "$dir_mix/lab.pfile$i";
    my $in1  = "$dir_lib_sp/lab.pfile$i";    # SP BPE 标签（经 rand_by_index 对齐后）
    my $in2  = "$dir_lib_ce/lab.pfile$i";    # CE phone 标签

    warn("$bin_pfile_paste $out $in1 $in2");
    system("$bin_pfile_paste $out $in1 $in2");
}
