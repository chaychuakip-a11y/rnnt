cat /yrfs4/asrdictt/hjwang11/traindata/french/data/all.txt /yrfs4/asrdictt/hjwang11/traindata/french/data/gongban/all.txt >tmp.txt

dict=/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/am/0.data_process/0.common_gbz_20260206/gen_fst_bin/dict/french_ly1213h_ly2k_xw696h_openSRC412h_fz1655h_ttsMusicVideoPoi_v16.word_dict_phone_nosp.checked
for j in tmp
do
	perl /work1/asrdictt/hjwang11/sbin/GetOOVWordFromCorpus_v2.pl $dict $j.txt $j.oov $j.with_oov $j.no_oov
done

cat /raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/am/0.data_process/0.common_gbz_20260206/check_oov/*/oov.list tmp.oov >all.oov


###手动去重，清洗开头或者结尾的-