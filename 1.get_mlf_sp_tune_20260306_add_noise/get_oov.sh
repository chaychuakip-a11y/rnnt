### 集外词过滤
dict=/yrfs4/asrdictt/hjwang11/multilingual/indonesian/rnnt/spm_2000/states_list.indonesian_onlyen.spm2.0k.map
for i in out.mlf_sy.sent.mlf_sp
do
	perl /work1/asrdictt/hjwang11/sbin/GetOOVWordFromCorpus_v2.pl $dict $i $i.oov $i.with_oov $i.no_oov
done

