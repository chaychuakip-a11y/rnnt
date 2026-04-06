#encoding=utf-8
import os
import sys
import torch

#src_model_path = "../rnnt_train/train_bpe2k_4wh_half10/model.iter14.part0.trans"
src_model_path = sys.argv[1]
model = torch.load(src_model_path)

#model
state_dict = model
#print(state_dict['encoder.conv2.bn._ifly_bitbrain_ignore_parameter'])
for key, value in state_dict.items():
    print(key, " ",  torch.max(value))
