# source /work/asr/dyliu2/workspace/pytorch/train/example/hybcnn_ED/pytorch-asr-extension-master-cuda9.2/bashrc

# while [ ! -f '/yrfs4/asrdictt/zhyou2/traindata/mlg20231115/english/all/fea.pfile.0.done.finish' ]
# do
# echo waiting..
# sleep 60
# done

########### export PATH="/work/asr/dyliu2/AutoML/anaconda3/bin:$PATH"
########### 多机训练需加上
##########export NCCL_SOCKET_IFNAME=eno2.100
##########
##########export MASTER_ADDR=172.20.99.165
##########export MASTER_PORT=29564
##########export RANK=$1
##########export WORLD_SIZE=2
##########export NCCL_DEBUG=INFO
##########export NCCL_IB_DISABLE=1

if [ -f ./asr/c.so.111 ]
then
    echo "switch to cuda 11.1"
    #mv ./asr/c.so ./asr/c-v.so
    #mv ./asr/c-a.so ./asr/c.so
else
    echo "Now already cuda 11.1"
fi

#export CUDA_VISIBLE_DEVICES=$2
# kill -9 $(ps -ef |grep 'python'|grep -v grep|awk '{print $2}')

#source /home/asr/zhyou2/.bashrc_20201019
#source activate pytorch

#cp asr/c-a.so asr/c.so
# # source /home1/asrdictt/zhyou2/.bashrc_20220225_A100
# # source activate pytorch
# source ./pt190cu111.bashrc

# python train.py config.ini

# source /work/asrprg/whzhang9/pytorch.bashrc

# # export CUDA_VISIBLE_DEVICES=0

# source activate v5mocha
# # source activate mocha_python_A100

#source /work1/asrdictt/zhyou2/.bashrc_20210525_cuda111
#source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191
#echo fastemit-k2-a
source /train8/asrmlg/ddye2/RNNT/danish/am/5_rnnt/4_train_ddye2_500_end_l1/bashrc_20210525_cuda111
source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191
python3 w_analyst_new.py
#out_dir=out_train_004
#cmd=config_all_lr0.08_fix_init-en_phone.ini
#DESCRIPTION=russian_l1_1
#python run_process.py ${out_dir} ${cmd} ${DESCRIPTION}

#python train.py config_all_lr0.08_fix_init-en_phone.ini
#python train.py config_clamp_V1_exp2.ini
