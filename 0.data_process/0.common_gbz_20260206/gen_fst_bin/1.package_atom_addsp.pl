use strict;

my $name = "french_ly1213h_ly2k_xw696h_openSRC412h_fz1655h_ttsMusicVideoPoi_v16.word_dict_phone_nosp.add_pred";

my $dir_root           = ".";
my $lm                 = "./$name.1gram";
my $dict               = "./$name";
my $hmmlist            = "/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/french/data_prepare/mle_yingchen15/hmmlist.final";
my $out_key            = "$name";
my $dir_out            = "$dir_root/atom_FA_$name\_1gram";

my $floor_value        = 1000000;

my $dir_bin            = "/work1/asrdictt/taoyu/bin_atom";
my $bin_package        = "$dir_bin/package_ori";
my $bin_wfst_res       = "$dir_bin/BuildWFSTResTools";
my $bin_fst_res        = "$dir_bin/iFlyBuildFstRes";
my $bin_fstcompile     = "$dir_bin/fstcompile";

if(!-e $dir_out)
{
	mkdir($dir_out) || die "Error: mkdir $dir_out\n";
}

!system("$bin_package $dict $hmmlist $lm $floor_value $dir_out") || die "Error: fail to package, $!\n";
chdir("$dir_out") || die "Cannot chdir to $dir_out\n";
!system("$bin_wfst_res output.wfst.mvrd.txt atom_wfst.bin $out_key 0 1") || die "Error: fail to BuildWFSTRes\n";
!system("$bin_fst_res atom_fst.bin L C A2 L_W triphones.syms words.syms") || die "Error: fail to BuildFstRes";
!system("mv atom_fst.bin fst.bin") || die "Error: fail to rename fst.bin";
!system("$bin_fstcompile -isymbols=triphones.syms -osymbols=words.syms output.wfst.mvrd.txt G.fst") || die "Error: fail to create G.fst";
