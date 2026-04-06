#encoding=utf-8
import os
import sys
import torch

#src_model_path = "../rnnt_train/train_bpe2k_4wh_half10/model.iter17.part2"
#dst_model_path = "../rnnt_train/train_bpe2k_4wh_half10/model.iter17.part2.trans"

src_model_path=sys.argv[1]
dst_model_path=src_model_path+".convert.relu.blank"

model = torch.load(src_model_path)
# print(model.keys())

#model
state_dict = model#['state_dict']
cls_weight = state_dict["joint.project_layer.weight"]
for key,value in state_dict.items():
    print(key,",",value.shape)
#cls_weight = cls_weight[:-1]
#tmp = cls_weight[-1]
#print(cls_weight[2:].shape)
#print(cls_weight[1:-1].shape)

tmp =  cls_weight[1:4].clone()
# b = 3
b=1
cls_weight[1:-b-3] = cls_weight[1+3:-b].clone()
cls_weight[-(3+b):-b] = tmp
#cls_weight[1] = tmp

bias = state_dict["joint.project_layer.bias"]

#bias = bias[:-1]
#tmp = bias[-1]
tmp = bias[1:4].clone()
bias[1:-b-3] = bias[1+3:-b].clone()
bias[-(3+b):-b] = tmp
#bias[1] = tmp


emb = state_dict['decoder.embedding.embedding.weight']

tmp = emb[1:4].clone()
emb[1:-3] = emb[1+3:].clone()
emb[-3:] = tmp

#emb[1] = tmp


state_dict["joint.project_layer.weight"] = cls_weight
state_dict["joint.project_layer.bias"] = bias
state_dict['decoder.embedding.embedding.weight'] = emb
# model["state_dict"] = state_dict
# torch.save(model, dst_model_path)
torch.save(state_dict, dst_model_path)
