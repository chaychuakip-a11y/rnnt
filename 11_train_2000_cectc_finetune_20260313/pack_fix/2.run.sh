
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

if [ -f /etc/profile ]; then
    . /etc/profile
fi
# cuda & cudnn
export PATH=$PATH:/opt/lib/cuda-10.2/bin:/opt/lib/cudnn/cudnn-10.2-v7.6.5.32/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/lib/cuda-10.2/lib64:/opt/lib/cudnn/cudnn-10.2-v7.6.5.32/lib64                                             
export LIBRARY_PATH=$LIBRARY_PATH:/opt/lib/cuda-10.2/lib64:/opt/lib/cudnn/cudnn-10.2-v7.6.5.32/lib64
# gcc
export PATH=$PATH:/opt/compiler/gcc-7.3.0-os7.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/compiler/gcc-7.3.0-os7.2/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/opt/compiler/gcc-7.3.0-os7.2/lib64
# local bin
export PATH=$PATH:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin
# hadoop, dlp and java
export PATH=$PATH:/home/hadoop/hadoop2/bin:/opt/dls_cli:/opt/lib/jdk1.7.0_55/bin

#module load cuda/11.1
source /home3/asrprg/sjguo6/anaconda3_home/bin/activate rnnt
which conda
#source /train8/asrmlg/ddye2/RNNT/danish/am/5_rnnt/4_train_ddye2_500_end_l1/bashrc_20210525_cuda111
#source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191
#source ../rnnt_train/.bashrc

export CUDA_VISIBLE_DEVICES=1
module load gcc/7.3.0-os7.2

# source /home3/asrprg/qhxu3/rnnt.bashrc

#python export_onnx_new.py
# python export_onnx_8bit.py
#python export_onnx_8bit_test.py

# python export_onnx_biaoshi.py

source /home3/asrtrans/wszhang9/anaconda3/bin/activate /home3/asrtrans/wszhang9/anaconda3/envs/XLite
python3 simOnnx.py
