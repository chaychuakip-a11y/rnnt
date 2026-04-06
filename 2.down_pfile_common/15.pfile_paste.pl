use strict;

#@ARGV >= 0 || die "usage: pl split_id\n";

my ($dir, $split_id)   = @ARGV;

my $nSplit = 100;
my $dir_lib_supervised_ctcTriphone_ctc = "$dir/lib_fb40";
my $dir_lib_supervised_ctcTriphone_ed  = "$dir";

my $dir_lib_supervised_ctcTriphone_ed_ctc_mix = "$dir/lib_fb40/mix/";

#tool
my $bin_pfile_paste  = "/work2/asrdictt/xxtong/workbak_xxtong/tools/bin/pfile_paste";
my $cmd;

foreach my $split_id($split_id)
{
    $split_id        < $nSplit || die "Error: not support";
}

system("mkdir -p $dir_lib_supervised_ctcTriphone_ed_ctc_mix") if(!-e $dir_lib_supervised_ctcTriphone_ed_ctc_mix);

my $split=$split_id;
my $i=$split;
{
	warn("$bin_pfile_paste $dir_lib_supervised_ctcTriphone_ed_ctc_mix/lab.pfile$i $dir_lib_supervised_ctcTriphone_ed/lab.pfile$i $dir_lib_supervised_ctcTriphone_ctc/lab.pfile$i");
	system("$bin_pfile_paste $dir_lib_supervised_ctcTriphone_ed_ctc_mix/lab.pfile$i $dir_lib_supervised_ctcTriphone_ed/lab.pfile$i $dir_lib_supervised_ctcTriphone_ctc/lab.pfile$i");
}
