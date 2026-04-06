
# my $keyword = $ARGV[0];
my $keyword = 'out.mlf_sy';

open IN, "$keyword.scp";
my @scps = <IN>;
chomp @scps;
close IN;

open IN, "$keyword.sent.mlf_sp";
my @mlfs = <IN>;
chomp @mlfs;
close IN;


open OUT, ">$keyword.mlf.danzi.mlf_sp";
print OUT "#!MLF!#\n";

for(my $indx=0; $indx<@scps; $indx+=1){
	print OUT "\"*\/", $scps[$indx], ".lab\"\n";
	my @words = split(/ /, $mlfs[$indx]);
	print OUT join("\n", @words), "\n";
	print OUT ".\n";
}
