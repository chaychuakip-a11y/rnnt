use strict;
my $bin_qnnorm     = "/work1/asrdictt/taoyu/tools/QN/bin/qnnorm";
my $bin_pfile_info = "/work1/asrdictt/taoyu/tools/QN/bin/pfile_info";

my $dir_0	= "/yrfs4/asrdictt/zhyou2/traindata/mlg20231115/french/lib_fb40";
my $file_norm   = "$dir_0/fea.norm";
my $pfile_fea   = "";
my $pfile_lab   = "";
my $nSplit      = 10;

my @pfile_fea;
my @file_norm;
foreach my $i(0..$nSplit-1)
{
	$pfile_fea[$i] = "$dir_0/fea.pfile$i";
	$file_norm[$i]   = "$dir_0/fea.norm$i";
#	print "$i\_$_\n";
}

my $device_id   = 0; ### set GPUID
my $gpu_count   = 4; ### 4 or 8
my $gpu_arrays  = join(",", (0..$gpu_count-1));

my @nSent;
my @nFrame;
my @train_bp_range;
my @train_cv_range;
my $pfile_fea_cur;
my $pfile_lab_cur;
my $file_norm_cur;
my $pfile_fea_cv;
my $pfile_lab_cv;
my $epoch_cur;
my $i;
my $j;
my $cmd;




foreach $j(0..$nSplit-1)
{
#	$pfile_fea_cur = $nSplit > 1 ? $pfile_fea.$j : $pfile_fea;
	$pfile_fea_cur = $pfile_fea[$j];
	my $tmp = `$bin_pfile_info -i $pfile_fea_cur`;
	if($tmp && $tmp =~ /(\d+)\s*sentences,\s*(\d+)\s*frames/)
	{
		$nSent[$j] = $1;
		$nFrame[$j] = $2;
	}
	else
	{
		die "Error: fail to get sentence count, from pfile: $pfile_fea_cur\n";
	}

}

if(!-e $file_norm)
{
	foreach $j(0..$nSplit-1)
	{
#		$pfile_fea_cur = $nSplit > 1 ? $pfile_fea.$j : $pfile_fea;
#		$file_norm_cur = $nSplit > 1 ? $file_norm.$j : $file_norm;
		$pfile_fea_cur = $pfile_fea[$j];
		$file_norm_cur = $file_norm[$j];
		next if(-e $file_norm_cur);
		!system("$bin_qnnorm norm_ftrfile=$pfile_fea_cur output_normfile=$file_norm_cur") || die "qnnorm failed: $file_norm_cur.\n";
	}

	if($nSplit > 1)
	{
		my @mean_all;
		my @var_all;

		foreach $j(0..$nSplit-1)
		{
			my $nDim;
			my @mean;
			my @var;
			$file_norm_cur = $file_norm[$j];
			open(IN, $file_norm_cur) || die $!;
			$_ = <IN>;chomp;
			/^vec\s+(\d+)/ || die "Informal line: $_";
			$nDim = $1;
			while(<IN>)
			{
				chomp;
				push(@mean, $_);
				$nDim--;
				last if($nDim == 0);
			}
			$_ = <IN>;chomp;
			/^vec\s+(\d+)/ || die "Informal line: $_";
			$nDim = $1;
			while(<IN>)
			{
				chomp;
				push(@var, $_);
				$nDim--;
				last if($nDim == 0);
			}
			close IN;
			push(@mean_all, [@mean]);
			push(@var_all, [@var]);

			@mean == @var || die;
			@mean == @{$mean_all[0]} || die;
		}

		my @mean;
		my @var;
		my $nFrameTotal = 0;
		foreach $j(0..$nSplit-1)
		{
			$nFrameTotal += $nFrame[$j];
		}
		foreach $i(0..$#{$mean_all[0]})
		{
			foreach $j(0..$nSplit-1)
			{
				$mean[$i] += $nFrame[$j]/$nFrameTotal*${$mean_all[$j]}[$i];
				$var[$i] += $nFrame[$j]/$nFrameTotal*(1.0/${$var_all[$j]}[$i]/${$var_all[$j]}[$i] + ${$mean_all[$j]}[$i]*${$mean_all[$j]}[$i]);
			}
			$var[$i] -= $mean[$i]*$mean[$i];
			$var[$i] = 1/sqrt($var[$i]);
		}

		open(OUT, ">", $file_norm) || die;
		print OUT "vec ".scalar(@mean)."\n";
		foreach (@mean)
		{
			printf OUT "%.5e\n", $_;
		}
		print OUT "vec ".scalar(@var)."\n";
		foreach (@var)
		{
			printf OUT "%f\n", $_;
		}
		close OUT;
	}
}

print "Done!";
