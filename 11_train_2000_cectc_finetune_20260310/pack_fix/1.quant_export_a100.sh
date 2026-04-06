
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


source /home3/asrprg/sjguo6/anaconda3_home/bin/activate rnnt
which conda
#source /train8/asrmlg/ddye2/RNNT/danish/am/5_rnnt/4_train_ddye2_500_end_l1/bashrc_20210525_cuda111
#source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191
#which conda
#
export CUDA_VISIBLE_DEVICES=1
module load gcc/7.3.0-os7.2

# source /home3/asrprg/qhxu3/rnnt.bashrc

# source /train8/asrmlg/ddye2/RNNT/danish/am/5_rnnt/4_train_ddye2_500_end_l1/bashrc_20210525_cuda111
# source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191

cp -rp /raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/am/11_train_2000_cectc_finetune_20260310/quant_fix/trq_output/model_3e-07_240_cv93.38_ph93.79-enc54.65-dec21.97.pt ./
key=model_3e-07_240_cv93.38_ph93.79-enc54.65-dec21.97.pt
python trans_model_relu_quant.py $key

python quant_export_onnx_8bit.py $key.convert.relu.blank
