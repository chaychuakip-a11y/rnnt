open IN, "out.record.scp.bak";

my %hash;
my $count=0;
while(<IN>){
    chomp;
    $hash{$_}=1;
}
close IN;

warn "scp load done";

open IN, "/raw22/asrmlg/permanent/taoyu/data/mlg/car_data/pack20231031/fr20231031.scp";
open OUT, ">check.log";
while(<IN>){
    chomp;
    if(/(.*)\=/){
        # die $1, ' =================== ', $_;
        print OUT "$1\n" if(defined $hash{$_} );
        $count+=1;
    }
}
close IN;
close OUT;

warn "$count found\n";