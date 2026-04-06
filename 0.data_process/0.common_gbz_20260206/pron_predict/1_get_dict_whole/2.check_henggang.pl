use strict;

open IN, "./output/new_word.txt.out.dict";

my %hash;
while(<IN>){
    chomp;
    if(/(.*)\t(.*)/){
        $hash{$1} = $2;
        # die "$1, $2, $hash{$1}, ", $hash{"rechercheret"};
    }
}
close IN;

open IN, "./input/new_word.txt.todo";
open OUT, ">./output/new_word.txt.out.dict.check";
while(<IN>){
    chomp;
    my $word_ori = $_;
    my $word = $_;
    $word =~ s/[\-\;\(\)]//g;
    # die "$word_ori, $word, $hash{$word}, ", $hash{"rechercheret"};
    if(defined $hash{$word}){
        print OUT "$word_ori\t$hash{$word}\n";
    }
    else{
        # print "$_ cant pred pron\n";sleep 1;
    }
}
close IN; close OUT;