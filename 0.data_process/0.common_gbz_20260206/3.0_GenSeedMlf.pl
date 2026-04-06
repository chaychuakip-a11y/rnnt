use strict;
use List::Util 'shuffle';

# @ARGV == 2 || @ARGV == 3 || die "Usage: pl scp_in scp_out [seed]\n";

# my ($scp_in, $scp_out, $seed) = @ARGV;

my $hdir_src = "/workdir/asrdictt/gasrdictt/zhyou2/mlg/202311_car/french/french_zx8053h_wav_fb75_mlefa/*-part-* /workdir/asrdictt/gasrdictt/zhyou2/mlg/202311_car/french/french_tx13362h_wav_fb75_mlefa/*part-*";

system("hdfs dfs -cat $hdir_src | /work1/asrdictt/taoyu/bin/fea_lab_lat_unpack_1 - .");

my $scp_in  = "./out.record.scp";
my $scp_out = "./seed.mlf";
my $seed    = 100;

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