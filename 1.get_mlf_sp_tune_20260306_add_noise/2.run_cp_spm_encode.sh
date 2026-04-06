source /work1/asrdictt/zhyou2/.bashrc_20210525
source activate theta20230329_fairseq0122

part=($(seq 0 9))
for p in ${part[*]}
do
cp ./spm_encode.sh ./${p}
cd ./${p}
echo 'cd ./', $p
sh spm_encode.sh > spm_encode.log.$p 2>&1 &
cd ..
echo $p 
done
