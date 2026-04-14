use strict;
my $bin_qnnorm     = "/work1/asrdictt/taoyu/tools/QN/bin/qnnorm";
my $bin_pfile_info = "/work1/asrdictt/taoyu/tools/QN/bin/pfile_info";

# 匈牙利语 CE pfile 输出目录  ### set
my $dir_0      = "PLACEHOLDER_HU_PFILE_OUT_DIR/lib_fb40";
my $file_norm  = "$dir_0/fea.norm";    # 合并后的全局 norm 文件
my $nSplit     = 10;

my (@pfile_fea, @file_norm);
foreach my $i (0 .. $nSplit - 1)
{
    $pfile_fea[$i]  = "$dir_0/fea.pfile$i";
    $file_norm[$i]  = "$dir_0/fea.norm$i";
}

my (@nSent, @nFrame);

foreach my $j (0 .. $nSplit - 1)
{
    my $tmp = `$bin_pfile_info -i $pfile_fea[$j]`;
    if ($tmp && $tmp =~ /(\d+)\s*sentences,\s*(\d+)\s*frames/)
    {
        $nSent[$j]  = $1;
        $nFrame[$j] = $2;
    }
    else
    {
        die "Error: fail to get sentence count from pfile: $pfile_fea[$j]\n";
    }
}

if (!-e $file_norm)
{
    foreach my $j (0 .. $nSplit - 1)
    {
        my $pfile_fea_cur = $pfile_fea[$j];
        my $file_norm_cur = $file_norm[$j];
        next if (-e $file_norm_cur);
        !system("$bin_qnnorm norm_ftrfile=$pfile_fea_cur output_normfile=$file_norm_cur")
            || die "qnnorm failed: $file_norm_cur.\n";
    }

    # 加权合并 10 个 part 的 norm 为全局 fea.norm
    my (@mean_all, @var_all);
    foreach my $j (0 .. $nSplit - 1)
    {
        my $file_norm_cur = $file_norm[$j];
        my (@mean, @var, $nDim);
        open(IN, $file_norm_cur) || die $!;
        $_ = <IN>; chomp;
        /^vec\s+(\d+)/ || die "Informal line: $_";
        $nDim = $1;
        while (<IN>) { chomp; push @mean, $_; $nDim--; last if $nDim == 0; }
        $_ = <IN>; chomp;
        /^vec\s+(\d+)/ || die "Informal line: $_";
        $nDim = $1;
        while (<IN>) { chomp; push @var, $_; $nDim--; last if $nDim == 0; }
        close IN;
        push @mean_all, [@mean];
        push @var_all,  [@var];
        @mean == @var           || die;
        @mean == @{$mean_all[0]}|| die;
    }

    my (@mean, @var);
    my $nFrameTotal = 0;
    $nFrameTotal += $_ for @nFrame;

    foreach my $i (0 .. $#{$mean_all[0]})
    {
        foreach my $j (0 .. $nSplit - 1)
        {
            $mean[$i] += $nFrame[$j] / $nFrameTotal * ${$mean_all[$j]}[$i];
            $var[$i]  += $nFrame[$j] / $nFrameTotal
                       * (1.0 / ${$var_all[$j]}[$i] / ${$var_all[$j]}[$i]
                          + ${$mean_all[$j]}[$i] * ${$mean_all[$j]}[$i]);
        }
        $var[$i] -= $mean[$i] * $mean[$i];
        $var[$i]  = 1 / sqrt($var[$i]);
    }

    open(OUT, ">", $file_norm) || die;
    print OUT "vec " . scalar(@mean) . "\n";
    printf OUT "%.5e\n", $_ for @mean;
    print OUT "vec " . scalar(@var) . "\n";
    printf OUT "%f\n",  $_ for @var;
    close OUT;
}

print "Done! fea.norm written to $file_norm\n";
