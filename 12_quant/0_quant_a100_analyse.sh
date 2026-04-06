####调试机 99.238执行，99.156执行需要减低bunchsize
# 多机训练需加上
#export NCCL_SOCKET_IFNAME=eno2.100

# export CUDA_VISIBLE_DEVICES=$1
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1

if [ -f ./asr/c.so.102 ]
then
    echo "switch to cuda 10.2"
    mv ./asr/c.so ./asr/c.so.111
    cp ./asr/c.so.102 ./asr/c.so
else
    echo "Now already cuda 10.2"
fi

# source /home/asrprg/qhxu3/rnnt.bashrc
#source ~/.bashrc_20210525
#source /train8/asrmlg/ddye2/RNNT/norway_zhy_20240712/4_rnnt/bashrc_20240716
# source activate fastemit-k2
#source /train8/asrmlg/ddye2/RNNT/norway_zhy_20240712/4_rnnt/bashrc_20210525
#source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191
#source /work1/asrdictt/zhyou2/.bashrc_20210525_cuda111
#source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191
#echo fastemit-k2-a
source /work1/asrdictt/zhyou2/.bashrc_20210525
#source /train8/asrmlg/ddye2/RNNT/danish/am/5_rnnt/4_train_ddye2_500_end_l1/bashrc_20210525_cuda111
source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191
export CUDA_VISIBLE_DEVICES=$1


# python train.py config_nooov_lr0.08.ini
#model=/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/am/11_train_2000_cectc_clamp/train_v0/model.init
#model=/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/am/11_train_2000_cectc_clamp/train_v0/model.iter0.part5
model=/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/am/11_train_2000_cectc_clamp_2/train_v8/model.iter0.part6

python quant_train_search_final2_savetmp.py $model
python quant_train_search_final2_analyse.py
