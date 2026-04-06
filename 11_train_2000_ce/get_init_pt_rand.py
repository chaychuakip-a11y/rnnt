import torch
import os
import sys
import torch.nn as nn
import math

model_asr_path = sys.argv[1]
model_asr = torch.load(model_asr_path)


# print(model_asr['state_dict'].keys())
# for key in model_asr['state_dict'].keys():
#     print(f"{key}: {model_asr['state_dict'][key].shape}")
# exit()

# model_param_path = sys.argv[2]
# model_param = torch.load(model_param_path)
# for key in model_param['state_dict'].keys():
#     print(f"{key}: {model_param['state_dict'][key].shape}")
# exit()

def xavier(x, p=6):
    shape = x.shape
    count = 1
    for s in shape:
        count = count * s
    fan_in = count / shape[0]
    scale = math.sqrt(p/fan_in)
    nn.init.uniform_(x.data, -scale, scale)

# noblank_weight = torch.empty(2001, 256)
# noblank_bias = torch.empty(2001)
# xavier(noblank_weight)
# xavier(noblank_bias)
# print(noblank_weight.shape, noblank_bias.shape)
# model_asr['state_dict']['joint.project_layer.weight'] = torch.cat( (model_asr['state_dict']['joint.project_layer.weight'][:1,:], noblank_weight), dim=0 )
# model_asr['state_dict']['joint.project_layer.bias'] = torch.cat( (model_asr['state_dict']['joint.project_layer.bias'][:1], noblank_bias), dim=0 )

# blank_weight = torch.empty(1, 256)
# blank_bias = torch.empty(1)
# xavier(blank_weight)
# xavier(blank_bias)
# model_asr['state_dict']['joint.project_layer.weight'] = torch.cat( (blank_weight, model_asr['state_dict']['joint.project_layer.weight'][1:,:]), dim=0 )
# model_asr['state_dict']['joint.project_layer.bias'] = torch.cat( (blank_bias, model_asr['state_dict']['joint.project_layer.bias'][1:]), dim=0 )


# model_asr['state_dict']['joint.project_layer.weight'] = model_asr['state_dict']['encoder.dnn_skip_out.conv.weight'].squeeze()
# model_asr['state_dict']['joint.project_layer.bias'] = model_asr['state_dict']['encoder.dnn_skip_out.conv.bias']


#del model_asr['state_dict']['encoder.dnn_skip_out.conv.weight']
#del model_asr['state_dict']['encoder.dnn_skip_out.conv.bias']
#del model_asr['state_dict']['phone_ce.ctc_out.weight']
#del model_asr['state_dict']['phone_ce.ctc_out.bias']
#del model_asr['state_dict']['decoder.embedding.embedding.weight']
#del model_asr['state_dict']['joint.project_layer.weight']
#del model_asr['state_dict']['joint.project_layer.bias']

del model_asr['state_dict']['decoder.lstm.weight_ih_l0']
del model_asr['state_dict']['decoder.lstm.weight_hh_l0']
del model_asr['state_dict']['decoder.lstm.bias_ih_l0']
del model_asr['state_dict']['decoder.lstm.bias_hh_l0']
del model_asr['state_dict']['decoder.lstm2.weight_ih_l0']
del model_asr['state_dict']['decoder.lstm2.weight_hh_l0']
del model_asr['state_dict']['decoder.lstm2.bias_ih_l0']
del model_asr['state_dict']['decoder.lstm2.bias_hh_l0']
#
#del model_asr['state_dict']['decoder.output_proj.weight']
#del model_asr['state_dict']['decoder.output_proj.bias']
#del model_asr['state_dict']['decoder.embedding.embedding.weight']
#del model_asr['state_dict']['encoder.conv4.conv.conv.bias']
#del model_asr['state_dict']['encoder.conv04.conv.weight']
#del model_asr['state_dict']['encoder.conv04.conv.bias']


model_param_path = sys.argv[2]
with open(model_param_path, 'wb') as wf:
    torch.save(model_asr, wf)    
