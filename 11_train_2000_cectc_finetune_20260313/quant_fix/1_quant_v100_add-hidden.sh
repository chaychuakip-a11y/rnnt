###99.238执行
# 多机训练需加上
#export NCCL_SOCKET_IFNAME=eno2.100

export CUDA_VISIBLE_DEVICES=$1
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1

if [ -f ./asr/c.so.111 ]
then
    echo "switch to cuda 11.1"
    #mv ./asr/c.so ./asr/c.so.102
    #mv ./asr/c.so.111 ./asr/c.so
else
    echo "Now already cuda 11.1"
fi

# source /home/asrprg/qhxu3/a100.bashrc
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.


#a100
#source /work1/asrdictt/zhyou2/.bashrc_20210525_cuda111
#source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191
#echo fastemit-k2-a
#v100
source /work1/asrdictt/zhyou2/.bashrc_20210525
#source /train8/asrmlg/ddye2/RNNT/danish/am/5_rnnt/4_train_ddye2_500_end_l1/bashrc_20210525_cuda111
source activate /work1/asrdictt/zhyou2/anaconda3/envs/pth39_cd111_tch191

# python train.py config_nooov_lr0.08.ini

# python quant_train_search_final2_mblank_20240220_retrain.py

# python quant_train_search_final2_mblank.py
path=trq_output
model=/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/am/11_train_2000_cectc_finetune_20260313/train_v0_finetune/model.iter4.part6
mkdir $path
python quant_train_search_final2_skip_add-hidden.py $path $model


#part=($(seq 0 9))
#for p in ${part[*]}
#    do
#    for i in ${part[*]}
#    do
#        echo $p 
#		echo $i
#        save_path=save_model/trq-debug_$p\_$i
#        #mkdir $save_path
#        
#        model_path=/train8/asrmlg/ddye2/RNNT/russian/russian_gongban_finetune_gbcz_20241126/4_train_rnnt_add_2fenlei_ctc_0001_clamp_xddatatrain_czdatatrain_0005/out_train_005_ctcclamp/model.iter$p.part$i
#        python quant_train_search_final2_skip_add-hidden.py $save_path $model_path
#    done
#done

#save_path=save_model_10/trq-debug_2_1
#model_path=/train8/asrmlg/ddye2/RNNT/russian/russian_gongban_finetune_gbcz_20241126/4_train_rnnt_add_2fenlei_ctc_0001_clamp_xddatatrain_czdatatrain_0005/out_train_005_ctcclamp/model.iter2.part1
#python quant_train_search_final2_skip_add-hidden.py $save_path $model_path