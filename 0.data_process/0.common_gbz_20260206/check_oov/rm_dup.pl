
my %hash;

# open IN, "/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/french/data_prepare/gen_fst_bin/dict/french_ly1213h_ly2k_xw696h_openSRC412h_fz1655h_ttsMusicVideoPoi_v16.wlist";
# while(<IN>){
#     chomp;
#     $hash{$_}=1;
# }

open OUT, ">/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/french/data_prepare/pron_predict/1_get_dict_whole/input/new_word.txt.todo";

my @paths=(
    "/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/french/data_prepare/check_oov/mlf_tx/oov.list",
    "/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/french/data_prepare/check_oov/mlf_zx/oov.list",
);

foreach my $path(@paths){
    # die $path;
    open IN, "$path";
    while(<IN>){
        chomp;
        $hash{$_}=1;
    }
    close IN;
}

foreach my $key (sort{$hash{$b} <=> $hash{$a} } keys %hash)
{
        # my $value = $hash{$key};
        print OUT "$key\n";
}