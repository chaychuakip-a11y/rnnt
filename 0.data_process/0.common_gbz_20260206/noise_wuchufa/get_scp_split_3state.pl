
open IN, "out.mlf_fa_ph";
open OUT, ">out.mlf_fa_ph.3states";

my $name;

while(<IN>){
    if(/^(\d+)\s+(\d+)/){
        my $begin_frame = $1;
        my $total_frames = $2;
        my $first_frame = int($total_frames/3);
        my $second_frame = int($total_frames/3*2);
        die if($first_frame <= $begin_frame);
        die if($second_frame <= $first_frame);
        die if($total_frames <= $second_frame);
        print OUT "$begin_frame $first_frame sil_s2\n";
        print OUT "$first_frame $second_frame sil_s3\n";
        print OUT "$second_frame $total_frames sil_s4\n";
    }
    else{
        print OUT $_;
    }
}