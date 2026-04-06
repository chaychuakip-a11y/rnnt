# source /work/asr/dyliu2/workspace/pytorch/train/example/hybcnn_ED/pytorch-asr-extension-master-cuda9.2/bashrc

# while [ ! -f '/yrfs4/asrdictt/zhyou2/traindata/mlg20231115/english/all/fea.pfile.0.done.finish' ]
# do
# echo waiting..
# sleep 60
# done
 
# # 设置文件路径
# file="/yrfs4/asrdictt/zhyou2/traindata/jiaofu2024_mlg/russian/rnnt_pfile/fea.pfile00" 
# # 设置时间限制，例如：5分钟前
# max_age=300 
# # 获取当前时间戳
# current_time=$(date +%s) 
# # 获取文件最后修改时间戳
# mod_time=$(stat -c %Y "$file") 
# # 计算文件最后修改时间与当前时间的差值
# age=$((current_time - mod_time)) 
# # 判断文件是否未更新（即时间差是否小于设定的最大时间）
# while [ $age -lt $max_age ]; do
#     echo waiting..
#     sleep 60
# done

# export PATH="/work/asr/dyliu2/AutoML/anaconda3/bin:$PATH"
# 多机训练需加上
export NCCL_SOCKET_IFNAME=eno2.100

export MASTER_ADDR=172.20.98.153
export MASTER_PORT=29563
export RANK=$1
export WORLD_SIZE=2
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

#export CUDA_VISIBLE_DEVICES=$2
# kill -9 $(ps -ef |grep 'python'|grep -v grep|awk '{print $2}')

#source /home/asr/zhyou2/.bashrc_20201019
#source activate pytorch

# cp asr/c-a.so asr/c.so
# # source /home1/asrdictt/zhyou2/.bashrc_20220225_A100
# # source activate pytorch
# source ./pt190cu111.bashrc

# python train.py config.ini

# source /work/asrprg/whzhang9/pytorch.bashrc

# # export CUDA_VISIBLE_DEVICES=0

# source activate v5mocha
# # source activate mocha_python_A100

source /work1/asrdictt/zhyou2/.bashrc_20210525
source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191
echo fastemit-k2-v

# python train.py config_nooov_lr0.08.ini

python train.py config_all_lr0.08_fix_init-en_phone.ini
