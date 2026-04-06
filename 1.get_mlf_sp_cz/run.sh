source /work1/asrdictt/zhyou2/.bashrc_20210525
source activate theta20230329_fairseq0122

for i in {0..9}; do
	echo $i
	hdfs dfs -cat /workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand_speedup1.2_ampReduce_lsa_mae_fea_fa/*$i | /work1/asrdictt/hjwang11/bin/fea_lab_lat_unpack_1 - ./$i mlf_sy &
done
