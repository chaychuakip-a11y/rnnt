
my @scps;

# 用法：perl 11.get_same_labpfile.pl <ce_len_file> <ed_len_file> <split_id>
# 生成 <split_id>.index.scp，供 12.run_pfile_rand_by_index.sh 使用

open IN, "$ARGV[0]";
while (<IN>) {
    chomp;
    if (/(\d+) (\d+)/) {
        push @scps, $2;
    }
}
close IN;

open OUT, ">$ARGV[2].index.scp";
open IN2, "$ARGV[1]";
my $idx    = 0;
my $offset = 1;
while (<IN2>) {
    chomp;
    if (/(\d+) (\d+)/) {
        my $id  = $1;
        my $len = $2;
        if ($scps[$idx] == $len) {
            print OUT $id + $offset, "\n";
            $idx += 1;
        }
        else {
            warn;
        }
    }
}
