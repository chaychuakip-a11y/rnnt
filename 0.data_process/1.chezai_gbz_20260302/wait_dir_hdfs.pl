#!/usr/bin/perl -w
use strict;

@ARGV == 1 || @ARGV == 2 || die "Usage: pl dstpath [fs_namenode(hdfs://192.168.89.100:9040)]\n";

my ($dstpath, $dfs) = @ARGV;
if(@ARGV == 1)
{
	$dfs = "";
}
else
{
	$dfs = "-fs $dfs";
}

my $bNotExist = system("hdfs dfs $dfs -test -e $dstpath");

while($bNotExist)
{
	print "wait once, for $dstpath\n";
	sleep(60);
	$bNotExist = system("hdfs dfs $dfs -test -e $dstpath");
}
