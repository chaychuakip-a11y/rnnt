#!/home3/asrdictt/taoyu/anaconda3/envs/pytorch1.7/bin/python
#encoding=utf-8
import torch
import os,sys,re
from collections import OrderedDict
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if __name__ == "__main__":
    
    #if len(sys.argv) != 3:
    #    print('Usage: py in_torchmodel_no_stream out_torchmodel_stream')
    #    print('Function: no stream model, to stream model...')
    #    exit(-1)
    #in_torchmodel_init="/train8/asrmlg/ddye2/RNNT/russian/russian_gongban_initxdctc_czclamp_train_cz20250315/4_train_cztts_initts2x_2_8_trainctc/out_train_0006_ctcclamp/initmodel_raw/model.iter4.part5"
    in_torchmodel_ctc="/train8/asrmlg/ddye2/RNNT/russian/russian_gongban_xd_20250107/8_quant/tmp.pt"
    #in_torchmodel_ed="/train8/asrmlg/ddye2/RNNT/russian/russian_gongban_xd_20250107_finetune_v2_250425/11_train_xd_kaiyuan_afterv1_1_fixdecoder_1/out_train_0016/model.iter5.part8"
    #out_torchmodel="fixencoder_5_8_add_ctc_7_5_model2.model"

    
    #states_new = OrderedDict()
    #in_torchmodel, out_torchmodel = sys.argv[1:]
    #states_init = torch.load(in_torchmodel_init)
    states_ctc = torch.load(in_torchmodel_ctc)
    #states_ed = torch.load(in_torchmodel_ed)
            
    for p1 in states_ctc:
        print(p1)
        #if "phone_ce" in p1:
        #    print("p1",p1)
        #    states_ed["state_dict"][p1].copy_(states_ctc["state_dict"][p1])
        #elif "encoder" in p1 and "dnn_skip" not in p1:
        #    print("p1--",p1)
        #    states_ed["state_dict"][p1].copy_(states_init["state_dict"][p1])
 
    #torch.save(states_ed, out_torchmodel)
    '''
    states_new = OrderedDict()
    print('Loading base network...')
    
    for k, v in states["state_dict"].items():
        if "encoder.net" in k :
            states_new[k] = v
            k_new = "encoder.net_online." + k.split('.',2)[-1]
            print(k, '=>')
            print(k_new)
            states_new[k_new] = v.clone()
        elif "densenet.conv1" in k:
            k_new = re.sub("conv1","conv1.conv1",k)
            print(k, '=>')
            print(k_new)
            states_new[k_new] = v
        elif "conv2_" in k:
            conv = k.split('.')[4]
            conv_new = conv + ".conv1"
            k_new = re.sub(conv,conv_new,k)
            print(k, '=>')
            print(k_new)
            states_new[k_new] = v
        elif "ce_layer" in k:
            states_new[k] = v
            k_new = "ce_layer_chunk." + k.split('.',1)[-1]
            print(k, '=>')
            print(k_new)
            states_new[k_new] = v.clone()
        elif "encoder.enc_conv1" in k:
            states_new[k] = v
            k_new = "encoder.enc_conv1_online." + k.split('.',2)[-1]
            print(k, '=>')
            print(k_new)
            states_new[k_new] = v.clone()
        elif "decoder" in k:
            states_new[k] = v
            k_new = "decoder_online." + k.split('.',1)[-1]
            print(k, '=>')
            print(k_new)
            states_new[k_new] = v.clone()
        elif "classification" in k:
            states_new[k] = v
            k_new = k.split('.')[0] + "_online." + k.split('.',1)[-1]
            print(k, '=>')
            print(k_new)
            states_new[k_new] = v.clone()
        else:
            states_new[k] = v
            # print(k)
    torch.save({"state_dict": states_new}, out_torchmodel)
    '''