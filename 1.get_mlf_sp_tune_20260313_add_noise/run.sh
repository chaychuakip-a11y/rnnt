source /work1/asrdictt/zhyou2/.bashrc_20210525
source activate theta20230329_fairseq0122

for i in {0..9}; do
	echo $i
	hdfs dfs -cat /workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_common_1000h_merge_cz_400h_add_cn_noise5.7h_pure_noise95h_20260313/*$i | /work1/asrdictt/hjwang11/bin/fea_lab_lat_unpack_1 - ./$i mlf_sy &
done
