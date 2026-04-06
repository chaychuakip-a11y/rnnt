use strict;

use lib "/work1/asrdictt/hjwang11/sbin";
use share_hadoop;

my $jobname           = "fea_rm_head";#SET
my $jobqueue          = "adapt";#SET
my $num_reduce        = 10;
my $in_blocksize      = 512*1024*1024;
my $block_size        = 64*1024*1024;
my $replication       = 2;

my @hdir_src          = (
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise/mlefa_mix_all_addnoise_byd_0.2",
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise/mlefa_mix_all_addnoise_car_0.2",
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise/mlefa_mix_all_addnoise_duodian_0.1",
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise/mlefa_mix_all_addnoise_gswan_0.2",
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise/mlefa_mix_all_addnoise_jiaju15s_0.1",
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise/mlefa_mix_all_addnoise_music0_0.02",
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise/mlefa_mix_all_addnoise_music1_0.02",
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise/mlefa_mix_all_addnoise_music2_0.02",
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise/mlefa_mix_all_addnoise_onenoise_music_0.02",
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise/mlefa_mix_all_addnoise_tv_0.02",
"/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise/mlefa_mix_all_addnoise_white_0.1",
                        );
my $hdir_out          = ("/workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_fa_addnoise_rm_head_44bytes");
my $hdir_src          = join(" -input ", @hdir_src);
my $dir_tmp           = "tmp"; mkdir $dir_tmp if !-e $dir_tmp;
foreach my $hdir_cur (@hdir_src)
{
	if($hdir_cur =~ /\*/ && $hdir_cur =~ /part/)
	{
		$hdir_cur =~ s#/[^/]+$##;
	}
	$hdir_cur .= '/_SUCCESS';
	system("/work1/asrdictt/hjwang11/sbin/wait_dir_hdfs.pl $hdir_cur");
}

##tools
my $dir_bin            = "/work1/asrdictt/taoyu/bin";
my $bin_stream         = "$dir_bin/streamingAC-2.5.0.jar";
my $bin_selecttail     = "$dir_bin/selecttail";
my $bin_renametail     = "$dir_bin/renametail";
my $bin_randname       = "$dir_bin/randname";
my $bin_randnamered    = "$dir_bin/randnamered";
my $checkMLF           = "$dir_bin/checkMlfWithDict";
my $bin_cmvn           = "$dir_bin/htk-0.1.4/bin/cmvn_simple";
my $bin_raw_fea        = "$dir_bin/htk-0.1.4/bin/raw_fea";
my $config1            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerMFCC_0_D_A";
my $config2            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB24_0_D_A";
my $config3            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB40";
my $bin_rm_head        = "$dir_bin/rmDataHead";

my @cmd_map;
my @cmd_red;
my @files;
my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

@cmd_map = (
#::CmdToLocal("$checkMLF mlf_sy $dict"),
::CmdToLocal("$bin_selecttail wav mlf_sy mlf_fa_ph"),
#::CmdToLocal("$bin_rm_head wav 44"),
# ::CmdToLocal("$bin_raw_fea $config2 fb72"),
# ::CmdToLocal("$bin_cmvn 2 24 1 fb72"),
# ::CmdToLocal("$bin_raw_fea $config3 fb40"),
# ::CmdToLocal("$bin_selecttail mlf_sy fb40 mlf_fa_ph"),
# ::CmdToLocal("$bin_randname"),
);

@cmd_red = (
# ::CmdToLocal("$bin_randnamered"),
);

@files   = (
$bin_cmvn,
$bin_raw_fea,
$config1,
$config2,
$config3,
$bin_renametail,
$bin_selecttail,
$bin_rm_head,
$bin_randname,
$bin_randnamered,
);

$cmd_map = join(" | ", @cmd_map);
$cmd_red = join(" | ", @cmd_red);

if(@cmd_map > 1)
{
	open(OUT, ">", "$dir_tmp/mapper.$jobname.sh") || die $!;
	print OUT "#!/bin/bash\n";
	print OUT "$cmd_map\n";
	close OUT;
	push(@files, "$dir_tmp/mapper.$jobname.sh");
	$cmd_map = "bash ./mapper.$jobname.sh";
}

if(@cmd_red > 1)
{
	open(OUT, ">", "$dir_tmp/reducer.$jobname.sh") || die $!;
	print OUT "#!/bin/bash\n";
	print OUT "$cmd_red\n";
	close OUT;
	push(@files, "$dir_tmp/reducer.$jobname.sh");
	$cmd_red = "bash ./reducer.$jobname.sh";
}

$files = join(",", @files);

$cmd =  "hadoop jar $bin_stream "
."-Dmapreduce.map.java.opts=\"-Xmx72000m\" "
."-Dmapreduce.reduce.java.opts=\"-Xmx4096m\" "
# ."-Dmapreduce.map.cpu.vcores=$thread_nums "
."-Dmapreduce.map.failures.maxpercent=20 "
."-Dmapreduce.job.queuename=$jobqueue "
."-Dmapreduce.job.name=$jobname "
."-Dmapreduce.job.reduces=$num_reduce "
."-Dmapreduce.map.memory.mb=2000 "
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

sub ChangePara
{
    my ($cfg,$paraname,$para)=@_;
    if ($para=~/\//)
    {
        if ($para!~/workdir/ && !-e $para)
        {
            die "Error: can't find $para\n";
        }
        $para=~s/.*\//\.\//;
    }
    open(IN,$cfg)||die"can't open $cfg";
    my @cfgcon=();
    while(<IN>)
    {
        push @cfgcon,"$_";
    }
    close IN;
    open(OUT,">","$cfg") || die"can't open $cfg\n";
    my $find=0;
    foreach (@cfgcon)
    {
        next if(/^\s*\#/);
        if(/^\s*$paraname\s*=/)
        {
            print OUT "$paraname = $para\n";
            $find=1;
        }
        else
        {
            print OUT "$_";
        }
    }
    die "can't find $paraname\n" if($find==0);
    close OUT;
}
