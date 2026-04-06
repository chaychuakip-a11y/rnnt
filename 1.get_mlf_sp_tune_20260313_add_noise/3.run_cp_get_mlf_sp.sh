source /work1/asrdictt/zhyou2/.bashrc_20210525
source activate theta20230329_fairseq0122

part=($(seq 0 9))
for p in ${part[*]}
do
cp ./get_mlf_sp.pl ./${p}
cd ./${p}
echo 'cd ./', $p
perl get_mlf_sp.pl > get_mlf_sp.log.$p 2>&1 &
cd ..
echo $p 
done
