
my @scps;

# open IN, "/yrfs4/asrdictt/zhyou2/traindata/mlg20231115/english/car_from_hjwang20240530_notongyong_nonoise_fa/lib_fb40/lab.len.0";
open IN, "$ARGV[0]";
while(<IN>){
    chomp;
    if(/(\d+) (\d+)/){
        push @scps, $2;
    }
}
close IN;

open OUT, ">$ARGV[2].index.scp";
# open IN2, "/yrfs4/asrdictt/zhyou2/traindata/mlg20231115/english/car_from_hjwang20240530_notongyong_nonoise/lab.len.0";
open IN2, "$ARGV[1]";
my $idx=0;
my $offset=1;
while(<IN2>){
    chomp;
    if(/(\d+) (\d+)/){
        my $id=$1; my $len=$2;
        if($scps[$idx] == $len){
            # die "$scps[$idx], $id, $len, $_";
            print OUT $id+$offset,"\n";
            $idx+=1;
        }
        else{
            # $offset+=1;
            warn;
        }
    }
}

# /work1/asrdictt/taoyu/tools/bin/pfile_rand_by_index