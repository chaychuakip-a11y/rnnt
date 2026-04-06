# 多机训练需加上
#export NCCL_SOCKET_IFNAME=eno2.100

# export CUDA_VISIBLE_DEVICES=$1
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1

# if [ -f ./asr/c.so.111 ]
# then
#     echo "switch to cuda 11.1"
#     mv ./asr/c.so ./asr/c.so.102
#     mv ./asr/c.so.111 ./asr/c.so
# else
#     echo "Now already cuda 11.1"
# fi

# source /home/asrprg/qhxu3/a100.bashrc
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.


source /work1/asrdictt/zhyou2/.bashrc_20210525_cuda111
source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191
echo fastemit-k2-a

# ulimit -u 10240

init_pt=/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/am/10_train_2000_p1_ctc_russianinitmodel/train_onlywordCTC_v0/model.iter4.part9
init_param=./train_v0/model.init

python get_init_pt_rand.py $init_pt $init_param
