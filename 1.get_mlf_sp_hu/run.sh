source /work1/asrdictt/zhyou2/.bashrc_20210525
source activate theta20230329_fairseq0122

for i in {0..9}; do
	echo $i
	hdfs dfs -cat /workdir/asrdictt/tasrdictt/zhyou2/mlg/202311_car/french/dnnfa_fixmlf_addfangzhen_all_fbnocmn40/*$i | /work1/asrdictt/hjwang11/bin/fea_lab_lat_unpack_1 - ./$i mlf_sy &
done
