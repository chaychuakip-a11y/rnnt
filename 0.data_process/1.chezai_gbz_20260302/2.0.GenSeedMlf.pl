use strict;
use List::Util 'shuffle';

@ARGV == 2 || @ARGV == 3 || die "Usage: pl scp_in scp_out [seed]\n";

my ($scp_in, $scp_out, $seed) = @ARGV;

my (@list, @idx, $num);

srand($seed) if(defined($seed));

open(IN, "$scp_in") || die $!;
@list = <IN>;
map chomp, @list;
@idx = (0..$#list);
close(IN);

@idx = shuffle(@idx);

open(OUT, ">", $scp_out) || die $!;
my $i = 0;
foreach my $list (@list)
{
	$list =~ s/.*\///;
	$list =~ s/\.lab//;
	#my($id, $file) = split(/=/, $list);
	#my $id = join("-", $list, $idx[$i]);
	print OUT "\"*\/$list.lab\"\n";
	print OUT $idx[$i]."\n";
	print OUT "."."\n";
	$i ++;
}
close OUT;