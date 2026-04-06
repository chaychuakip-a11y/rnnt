source /work1/asrdictt/zhyou2/.bashrc_20210525
source activate theta20230329_fairseq0122

part=($(seq 0 9))
for p in ${part[*]}
do
cp ./get_mlf_sp.pl ./${p}
cd ./${p}
echo 'cd ./', $p
# perl get_mlf_sp.pl > get_mlf_sp.log.$p 2>&1 &
cat out.mlf_sy.mlf.danzi.mlf_sp /work1/asrdictt/hjwang11/multilingual/english/rnnt/am/20250508/0.data_process/empty.mlf.all > out.mlf_sy.mlf.danzi.mlf_sp.add_empty &
cd ..
echo $p 
done
