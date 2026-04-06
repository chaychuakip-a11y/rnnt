# by pcli2 2019 Dec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import numpy as np
import time
import math
import random
from asr.layers import *
# from asrops import LSTMP,UBLSTMP
from asr.data import clip_mask, cnn2rnn, rnn2cnn
from typing import List, Tuple, Dict
from asr.functions import xavier
from asr.train import Trainer
from warp_rnnt_fa import rnnt_loss
# import warp_mblank_rnnt_fa as mblank_rnnt_fa
# from mblankrnnt_pytorch import  mbalnk_rnnt_loss 
# from asr
# from warprnnt_pytorch import RNNTLoss
# import k2 
# from asr.layers.mochat_decoder import MoChADecoder
import struct

word_dict = "/work1/asrdictt/zhyou2/workspace/yewu/chezai_syllable_allnodes/states_list.all"
voc_size = 2000+1

class ACC0(nn.Module):
    def __init__(self):
        super(ACC0, self).__init__()


    def forward(self, logit, label):
        target = self.getlabel(label)
        lprobs = torch.nn.functional.log_softmax(logit, dim=1)
        acc = self.getacc(lprobs, target)
        return acc

    @jit.ignore
    def getlabel(self, label):
        target = label.clone()
        target = target.flatten()
        target[target <= 0] = -1
        target = target.long()
        target = target.unsqueeze(-1)
        return target

    @jit.ignore
    def getacc(self, lprob, target):
        num_class = lprob.size()[1]
        _, new_target = torch.broadcast_tensors(lprob, target)

        remove_pad_mask = new_target.ne(-1)
        lprob = lprob[remove_pad_mask]

        target = target[target != -1]
        target = target.unsqueeze(-1)

        lprob = lprob.reshape((-1, num_class))

        preds = torch.argmax(lprob, dim=1)
        correct_holder = torch.eq(preds.squeeze(), target.squeeze()).float()

        num_corr = correct_holder.sum()
        num_sample = torch.numel(correct_holder)
        acc = num_corr / num_sample
        return acc


class CeLoss0(nn.Module):
    def __init__(self):
        super(CeLoss0, self).__init__()
        self.celoss = nn.CrossEntropyLoss(ignore_index=-1,reduction='mean')

    #@jit.script_method
    def forward(self, x, frame_label, num_classes, nmod):
        target = frame_label.clone()
        target = target.flatten()
        target[target<=0] = -1
        target = target.long()
        target = target.view(-1,nmod)[:,0]
        target = target.unsqueeze(-1)
        target = target.flatten().long()
        
        x = x.reshape((-1, num_classes))
        
        ce_loss = self.celoss(x, target)
        return ce_loss

class Vocabulary(object):
    def __init__(self,f_dic):
        self.idx2word = []
        self.word2idx = {}
        with open(f_dic, "r", encoding="GBK") as f:
            for idx, line in enumerate(f):
                x = line.strip()
                self.idx2word.append(x)
                self.word2idx[x] = idx

    def list_to_sent(self, li):
        words = [self.idx2word[i] for i in li]
        return " ".join(words)

def write_in_htk(out_file, f_out_fea):
    # f_out_fea: [T, 15003]
    # print("***************start write*********************")
    with open(out_file, "wb") as o:
        num_frame2 = f_out_fea.shape[0]  # 32佝
        step_in_ns = 100000  # 32佝
        dim_frame2 = f_out_fea.shape[1] * 4  # 16佝
        fea_format = 9  # 16佝
        s = struct.pack(">iiHh", num_frame2, step_in_ns, dim_frame2, fea_format)
        o.write(s)
        flat_fea = f_out_fea.cpu().detach().numpy().reshape(-1).tolist()
        fmt = ">%df"%(len(flat_fea))
        s = struct.pack(fmt, *flat_fea)
        o.write(s)

def mask_data(data):
    b, c, h, w = data.size()
    for i in range(b):
        torch.random.seed()
        index = random.randint(0, w)
        torch.random.seed()
        width = random.randint(0, 100)
        end = min(index + width, w)
        data.data[i, :, :, index:end].fill_(0)
    return data

def mask_data_f(data):
    b, c, h, w = data.size()
    for i in range(b):
        torch.random.seed()
        index = random.randint(0, h)
        torch.random.seed()
        width = random.randint(0, 13)
        end = min(index + width, h)
        data.data[i, :, index:end, :].fill_(0)
    return data

def mask_data_y(data, len, _st):
    b,_ = data.size()
    for i in range(b):
        torch.random.seed()
        if(len[i]<2):continue
        index = random.randint(2, len[i])
        torch.random.seed()
        width = random.randint(0, (len[i]*0.15).int())
        end = min(index + width, len[i].int()-1)
        data.data[i, index:end].fill_(_st)
    return data

def edit_dist(labs, recs):
    # print(labs.shape(),recs.shape());exit()
    n_lab = len(labs)
    n_rec = len(recs)
    dist_mat = np.zeros((n_lab+1, n_rec+1))
    for j in range(n_rec + 1):
        dist_mat[0, j] = j

    for i in range(n_lab + 1):
        dist_mat[i, 0] = i

    for i in range(1, n_lab+1):
        for j in range(1, n_rec+1):
            hit_score = dist_mat[i-1,j-1] + (labs[i-1]!=recs[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1,j] + 1

            err = hit_score
            if err > ins_score:
                err = ins_score
            if err > del_score:
                err = del_score
            dist_mat[i, j]=err

    return dist_mat[n_lab, n_rec]

def computer_cer(preds, labels):
    dist = sum(edit_dist.eval(label, pred) for label, pred in zip(labels, preds))
    total = sum(len(l) for l in labels)
    return dist, total

def remove_negative(prob, plen, data):
    b, t, u, d = prob.size()
    u0 = 1
    while(1):
        if(torch.all(data[:,u-1-u0] == -1)):
            u0 = u0 + 1
        else:
            break
    if(u0>1):
        prob = prob[:,:,:u-(u0-1),:]
        data = data[:,:u-(u0-1)-1]
    t0 = torch.max(plen)
    if(t0<t):
        prob = prob[:,:t0,:,:]
    return prob.contiguous(), data.contiguous()

def check_negative(data):
    b, h, w = data.size()
    for i in range(b):
        for j in reversed(range(w)):
            if(data[i,0,j] == -1):
                data[i,0,j] = data[i,0,0]
            else:
                break
    return data

class conv2dBNRelu(nn.Module):
    def __init__(self,inchannels, outchannels, kernel=1, stride=1, padding=1, groups=1, dilation=1):
        super(conv2dBNRelu,self).__init__()
        self.conv = nn.Conv2d(inchannels, outchannels, kernel, stride, padding, groups = groups, dilation=dilation)
        self.bn = nn.BatchNorm2d(outchannels, affine=False)
        self.relu = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.reset()

    def reset(self):
        init.xavier_uniform_(self.conv.weight)
        init.constant_(self.conv.bias, 0.1)
    
    def forward(self, x, mask=None, relu=0):
        x = self.conv(x)
        if mask is not None:
            nmod = mask.shape[3] // x.shape[3]
            mask = mask[:,:,:,::nmod]
            if mask.shape[1]!=x.shape[1]:
                mask = mask.repeat([1, x.shape[1], x.shape[2], 1])
            #print("%d %d" % (mask.shape[3],x.shape[3]))
            x *= mask
        x = self.bn(x)
        if relu ==1:
            x = self.relu(x)
        return x

class NullModule(nn.Module):
    def __init__(self):
        super(NullModule, self).__init__()
    def forward(self, x):
        return x

class ConvBN(nn.Module):
    def __init__(self, input_channel, output_channel, kernel, stride, pad, bias_value, relu_value, act_type="None"):
        super(ConvBN, self).__init__()
        self.conv=nn.Conv2d(input_channel, output_channel, kernel, stride, pad)
        self.bn=nn.BatchNorm2d(output_channel, momentum=0.01)
        xavier(self.conv.weight)
        nn.init.constant_(self.conv.bias.data, bias_value)
        nn.init.constant_(self.bn.weight.data, 1)
        nn.init.constant_(self.bn.bias.data, 0)
        self.act_type = act_type
        if act_type == "ReLU":
            self.activation = nn.ReLU()
        if act_type == "Tanh":
            self.activation = nn.Tanh()
        if act_type == "None":
            self.activation = NullModule()

    def forward(self, x:torch.Tensor) -> torch.Tensor:  
        x = self.conv(x)
        # x = x.clamp(-8, 8)
        x = self.bn(x)
        # x = x.clamp(-8, 8)
        x = self.activation(x)
        return x

class ConvNoBN(nn.Module):
    def __init__(self, input_channel, output_channel, kernel, stride, pad, bias_value, relu_value, act_type="None"):
        super(ConvNoBN, self).__init__()
        self.conv=nn.Conv2d(input_channel, output_channel, kernel, stride, pad)
        xavier(self.conv.weight)
        nn.init.constant_(self.conv.bias.data, bias_value)
        self.act_type = act_type
        if act_type == "ReLU":
            self.activation = nn.ReLU()
        if act_type == "Tanh":
            self.activation = nn.Tanh()
        if act_type == "None":
            self.activation = NullModule()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = ConvBN(1, 32, (7,3), (1,1), (0,1), 0, 0.1, "ReLU")
        self.pool1 = nn.MaxPool2d((2,2), (2,2), (0,0))#17
        self.conv2 = ConvBN(32, 64, (5,3), (1,1), (0,1), 0, 0.1, "ReLU")
        self.conv3 = ConvBN(64, 64, (3,3), (1,1), (0,1), 0, 0.1, "ReLU")
        self.conv4 = ConvBN(1408, 192, (1,1), (1,1), (0,0), 0, 0.1, "None")

        hidden_size = 512
        n_layers =1
        dropout = 0
        bidirectional = False

        self.lstm = nn.LSTM(
            input_size=192,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.lstm3 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.conv01 = nn.Conv2d(512, 256, (1,3), (1,1), (0,2), groups =1, dilation= (1,2))
        xavier(self.conv01.weight)
        nn.init.constant_(self.conv01.bias.data, 0)
        self.bn01=nn.BatchNorm2d(256, momentum=0.01)
        self.relu01 = nn.ReLU() 
        self.conv02 = nn.Conv2d(256, 256, (1,3), (1,1), (0,2), groups =1, dilation= (1,2))
        xavier(self.conv02.weight)
        nn.init.constant_(self.conv02.bias.data, 0)
        self.bn02=nn.BatchNorm2d(256, momentum=0.01)
        self.relu02 = nn.ReLU()
        self.conv03 = nn.Conv2d(256, 256, (1,3), (1,1), (0,2), groups =1, dilation= (1,2))
        xavier(self.conv03.weight)
        nn.init.constant_(self.conv03.bias.data, 0)
        self.bn03=nn.BatchNorm2d(256, momentum=0.01)
        self.relu03 = nn.ReLU()
        self.conv04 = nn.Conv2d(256, 256, (1,3), (1,1), (0,2), groups =1, dilation= (1,2))
        xavier(self.conv04.weight)
        nn.init.constant_(self.conv04.bias.data, 0)
        self.bn04=nn.BatchNorm2d(256, momentum=0.01)
        self.relu04 = nn.ReLU() 

        self.dnn_skip1 = ConvNoBN(256, 256, (1,1), (1,1), (0,0), 0, 0.1, "ReLU")
        self.dnn_skip2 = ConvNoBN(256, 256, (1,1), (1,1), (0,0), 0, 0.1, "Tanh")
        self.dnn_skip_out = ConvNoBN(256, voc_size+1, (1,1), (1,1), (0,0), 0, 0.1, "None")
        xavier(self.dnn_skip1.conv.weight)
        nn.init.constant_(self.dnn_skip1.conv.bias.data, 0)
        xavier(self.dnn_skip2.conv.weight)
        nn.init.constant_(self.dnn_skip2.conv.bias.data, 0)
        xavier(self.dnn_skip_out.conv.weight)
        nn.init.constant_(self.dnn_skip_out.conv.bias.data, 0)

    def forward(self, xx: torch.Tensor, meta: Dict[str, torch.Tensor]=None) -> torch.Tensor:

        x = self.conv1(xx)
        # x = x.clamp(-256, 256)
        x = self.pool1(x)
        x = self.conv2(x)
        # x = x.clamp(-256, 256)
        
        x = self.conv3(x)
        # x = x.clamp(-256, 256)

        x = x.permute(0,1,3,2)
        x = x.reshape(x.shape[0],x.shape[1],-1,x.shape[3]*2)
        x = x.permute(0,1,3,2)
        x = x.reshape(x.shape[0],-1,1,x.shape[3])
        x = self.conv4(x)

        inputs = x.permute(0,3,2,1).reshape(x.shape[0],x.shape[3],x.shape[1]) #nchw -> ntd

        input_lengths = None
        if meta is not None:
            input_lengths = meta["inputs_length"]

        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths.cpu(), batch_first=True)

        self.lstm.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
        outputs1, hidden1 = self.lstm(inputs)
        if(len(outputs1)==2):
            output1_data, output1_len = outputs1
            temp_inputs = tuple((output1_data, output1_len, True, True))
            outputs2, hidden2 = self.lstm2(temp_inputs)
            output2_data, output2_len = outputs2
            temp_inputs = tuple((output2_data, output2_len, True, True))
            outputs, hidden = self.lstm3(temp_inputs)
        else:
            outputs2, hidden2 = self.lstm2(outputs1)
            outputs, hidden = self.lstm3(outputs2)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        x = outputs
        x = x.permute(0,2,1).reshape(x.shape[0],x.shape[2],1,x.shape[1]) #ntd->nchw		
        
        x = self.conv01(x)
        # x = x.clamp(-256, 256)
        x = self.bn01(x)
        # x = x.clamp(-8, 8)
        x = self.relu01(x)
        x = self.conv02(x)
        # x = x.clamp(-256, 256)
        x = self.bn02(x)
        # x = x.clamp(-8, 8)
        x = self.relu02(x)
        x = self.conv03(x)
        # x = x.clamp(-256, 256)
        x = self.bn03(x)
        # x = x.clamp(-8, 8)
        x = self.relu03(x)
        x = self.conv04(x)
        # x = x.clamp(-256, 256)
        x = self.bn04(x)
        # x = x.clamp(-1, 1)
        x = self.relu04(x)

        skip_out = self.dnn_skip1(x)
        skip_out = self.dnn_skip2(skip_out)
        skip_out = self.dnn_skip_out(skip_out)
        skip_out = skip_out.permute(0,3,2,1).reshape(skip_out.shape[0],skip_out.shape[3],skip_out.shape[1]) #nchw -> ntd

        x = x.permute(0,3,2,1).reshape(x.shape[0],x.shape[3],x.shape[1]) #nchw -> ntd

        return x, skip_out

class MaskEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(MaskEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight.data, -0.05, 0.05)

    def forward(self, input_):
        mask, input_ = self.mask(input_)
        out = self.embedding(input_)
        mask = mask.unsqueeze(-1)
        out = out * mask.float()
        return out

    @jit.ignore
    def mask(self, input_) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = input_.clone()
        input_[input_ < 0] = 0
        mask[mask >= 0] = 1
        mask[mask < 0] = 0
        return mask, input_

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        vocab_size  = voc_size
        hidden_size = 320
        output_size = 256
        n_layers    = 1
        dropout     = 0
        share_weight= False

        self.embedding = MaskEmbedding(num_embeddings=vocab_size, embedding_dim=hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
		
        self.output_proj = nn.Linear(hidden_size, output_size)
        if share_weight:
            self.embedding.weight = self.output_proj.weight

    def forward(self, x, length=None, hidden=None):
        embed_inputs = self.embedding(x)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths.cpu(), batch_first=True)

        if(hidden is not None):
            p_hidden0 = hidden[0]
            p_hidden1 = hidden[1]
        else:
            p_hidden0 = None
            p_hidden1 = None

        self.lstm.flatten_parameters()
        outputs1, c_hidden1= self.lstm(embed_inputs, p_hidden0)

        self.lstm2.flatten_parameters()
        if(len(outputs1)==2):
            output1_data, output1_len = outputs1
            temp_inputs = tuple((output1_data, output1_len, True, True))
            outputs, c_hidden = self.lstm2(temp_inputs, p_hidden1)
        elif(len(outputs1)==1):
            temp_inputs = torch.nn.utils.rnn.pack_padded_sequence(outputs1, torch.tensor(np.array([1])).long().cpu(), batch_first=True)   #for quant.py
            if(p_hidden1 is None):
                outputs, c_hidden = self.lstm2(outputs1)   #for quant.py
            else:
                outputs, c_hidden = self.lstm2(temp_inputs, p_hidden1)
            outputs = outputs[0]
        else:
            outputs, c_hidden = self.lstm2(outputs1, p_hidden1)

        hiddens = (c_hidden1, c_hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        outputs = self.output_proj(outputs)

        return outputs, hiddens

class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)

        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            enc_state = enc_state.unsqueeze(2)# N T D
            dec_state = dec_state.unsqueeze(1)# N U D

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])#N T U D
            dec_state = dec_state.repeat([1, t, 1, 1])#N T U D
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)

        outputs = self.project_layer(outputs)

        return outputs

class phone_ce(nn.Module):
    def __init__(self, input_size, vocab_size):
        super(phone_ce, self).__init__()

        self.ctc_dnn = nn.Linear(input_size, input_size)
        xavier(self.ctc_dnn.weight)
        nn.init.zeros_(self.ctc_dnn.bias.data)
        self.relu = nn.ReLU()

        self.ctc_out = nn.Linear(input_size, vocab_size)
        xavier(self.ctc_out.weight)
        nn.init.zeros_(self.ctc_out.bias.data)


    def forward(self, enc_state):
        
        phone_ctc_logit = self.ctc_dnn(enc_state) # BTD 
        phone_ctc_logit = self.relu(phone_ctc_logit) # BTD 
        phone_ctc_logit = self.ctc_out(phone_ctc_logit) # BTD 

        return phone_ctc_logit


class Transducer(nn.Module):
    def __init__(self):
        super(Transducer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.joint   = JointNet(
            input_size=512,
            inner_dim=256,
            vocab_size=voc_size+1
            )
            
        self.phone_dim = 41 ###set，phones.list数量

        self.phone_ce = phone_ce(256, self.phone_dim)

        self.blank_id = 0
        # self.crit = RNNTLoss()
        self.accuracy = ACC()

        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean',zero_infinity=True)
        self.acc_ctc = AccCtc(blank=0)

        self.loss = CeLoss()
        self.loss0 = CeLoss0()
        self.accuracy0 = ACC0()

        self.total_word = 0
        self.total_dist = 0

        self.total_dist_skip = 0
        self.total_skip_count_enc = 0
        self.total_skip_count_dec = 0
        self.total_t_length = 0

        self.num_samples = 0
        self.sum_acc = 0
        
    def forward(self, x: torch.Tensor, meta:Dict[str, torch.Tensor]):        
        
        # for name,param in self.named_parameters():
            # print(name, param.requires_grad)
        # self.encoder.eval()
        
        # print("slef.training:", self.training)

        if self.training:
            x = mask_data_f(x)
        (b, c, f, t) = x.size()

        targets        = meta["att_label"].permute(1,0).contiguous()
        ###downpfile数据states.list添加了blank，则下面两行注释掉
        # targets[targets == -1] = -2
        # targets = targets.add(1)
        # targets[targets == 7028] = 0
        # targets[targets == 1] = 5

        att_mask = meta["att_mask"].permute(1,0).contiguous()
        targets_length = att_mask.sum(1)
        # import pdb;pdb.set_trace()
        # meta["label_ctc_mask_noise"] 

        ### enc
        enc_state, logits_enc_ctc = self.encoder(x, meta)

        if "label_ctc_ph" in meta:
            label_ctc = meta["label_ctc_ph"].permute(1,0).contiguous()
            label_ctc_mask = meta["label_ctc_ph_mask"].permute(1,0).contiguous()
            label_ctc_length = label_ctc_mask.sum(1)
            # print('111: ', targets.shape, att_mask.shape, label_ctc.shape, label_ctc_mask.shape, enc_state.shape)
            try:
                label_ctc = clip_mask(label_ctc, enc_state.size(1), 1)
            except Exception as e:
                print(e, label_ctc.shape, enc_state.shape);exit()   #mask dim 216 must be a integer multiple of clip_length 51 torch.Size([4, 216]) torch.Size([4, 51, 256])
        # print(label_ctc_mask.shape, label_ctc_mask[0]);exit()

        # print(targets[0], label_ctc[0])
        # print(targets.shape,label_ctc.shape);exit()    #torch.Size([200, 40]) torch.Size([14, 40]) torch.Size([40, 200])
            
        data_mask = clip_mask(meta["rnn_mask"], enc_state.size(1), 0)
        data_mask = data_mask.squeeze(2).permute(1,0)
        inputs_length =  data_mask.sum(1)

        # phone_ctc_logit = self.ctc_dnn(enc_state) # BTD 
        # phone_ctc_logit = self.relu(phone_ctc_logit) # BTD 
        # phone_ctc_logit = self.ctc_out(phone_ctc_logit) # BTD 
        # print(enc_state.shape);exit()   # torch.Size([8, 228, 256])
        phone_ctc_logit = self.phone_ce(enc_state) # BTD 

        logits_enc_ctc_post = logits_enc_ctc[:,:,:-1]  
        (_, logits_enc_idx_by_ctc) = torch.max(logits_enc_ctc_post, dim=2)
        logits_enc_idx_by_ctc[logits_enc_idx_by_ctc>0]=1  

        logits_enc_binary = torch.cat((logits_enc_ctc[:,:,0:1], logits_enc_ctc[:,:,-1:]), dim=-1)
        # # logits_enc_binary = F.linear(skip_state, b_weight, b_bias)
        logits_enc_binary = logits_enc_binary.reshape(-1,logits_enc_binary.shape[-1])
        # # print(x.shape, logits_enc_binary.shape, logits_enc_post.shape, logits_enc_idx.detach().shape)#;exit()   #torch.Size([8, 1, 40, 912]) torch.Size([8, 228, 2]) torch.Size([8, 228, 1601]) torch.Size([8, 228])      

        ### dec 
        targets = F.pad(targets, pad=(1, 0, 0, 0), value=2) #add start label zero
        targets_length = targets_length.add(1)
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0) #add blank label zero
        concat_targets_length = targets_length.add(1)
        if self.training:
            concat_targets = mask_data_y(concat_targets, concat_targets_length, 1)

        dec_state,_ = self.decoder(concat_targets, concat_targets_length)

        # logits_tmp  = self.joint(enc_state, dec_state) #BT(U+1)D

        # logits_post = logits_tmp[:,:,:,:-1]  
        # logits_post_lab = logits_post.reshape(logits_post.shape[0],-1,logits_post.shape[-1])
        # print(logits_post_lab.shape)    #torch.Size([8, 13680, 1601])
        # (logits_pred, logits_idx) = torch.max(logits_post_lab, dim=2)
        # logits_idx[logits_idx>0]=1

        # logits_final = torch.cat((logits_tmp[:,:,:,0:1], logits_tmp[:,:,:,-1:]), dim=-1)  
        # logits_final = logits_final.reshape(-1,logits_final.shape[-1])      
                    
        if self.training:  
            # print("train: ", targets)
            # modified by qhxu3 - start.
            # use warp-rnnt instead of warp-transducer, which is faster theoretically and has fastemit built-in.
            # logits_rnnt = torch.log_softmax(logits_post, -1)
            # loss = rnnt_loss(
            #     logits_rnnt.contiguous().cuda(),
            #     targets.cuda().int(),
            #     inputs_length.int(), 
            #     targets_length.int(),
            #     reduction='mean',
            #     fastemit_lambda=0.001
            # )
            # # modified by qhxu3 - end.
            # # print(targets)
            
            ### ctc loss
            log_probs = F.log_softmax(logits_enc_ctc_post, dim=-1)
            ## log_probs torch.Size([204, 15, 1601]) torch.Size([204, 8]) torch.Size([204, 1, 40, 60]) torch.Size([204]) torch.Size([204])
            # print('log_probs', log_probs.shape, targets.shape, x.shape,inputs_length.shape,targets_length.shape)
            #### assert shape: log_probs:(t,b,d), ctc_label:(b,t), input_lengths:(b), target_lengths:(b)
            loss_ctc = self.ctc_loss(log_probs.permute(1,0,2),targets.cuda().long(),inputs_length.long(),targets_length.long())
            
            ##ctc binary
            # loss_enc_binary_ctc = self.loss(logits_enc_binary, logits_enc_idx_by_ctc.detach())
            # if((logits_enc_idx_by_ctc==0).all() ):
            #     loss_enc_binary_ctc0 = torch.Tensor([0]).cuda()
            # else:
            #     loss_enc_binary_ctc0 = self.loss0(logits_enc_binary, logits_enc_idx_by_ctc.detach(), 2, 1)

            ### dec binary
            # loss_binary = self.loss(logits_final, logits_idx.detach())
            # if((logits_idx==0).all() ):
            #     loss_binary0 = torch.Tensor([0]).cuda()
            # else:
            #     loss_binary0 = self.loss0(logits_final, logits_idx.detach(), 2, 1)


            # # ### for enc binary, get dec preds
            # zero_token = torch.LongTensor([[0]])
            # if x.is_cuda:
            #     zero_token = zero_token.cuda()
            # def decode(enc_state, lengths):
            #     token_list = []
            #     token_list_b = []
            #     dec_state, hidden = self.decoder(zero_token)
            #     for t in range(lengths):
            #         logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))

            #         logits_b = torch.cat((logits[:1], logits[-1:]))
            #         # print(logits_b)
            #         out_b = F.softmax(logits_b, dim=0).detach()
            #         pred_b = torch.argmax(out_b, dim=0)
            #         pred_b = int(pred_b.item())
            #         token_list_b.append(pred_b)

            #         logits = logits[:-1]
            #         out = F.softmax(logits, dim=0).detach()
            #         pred = torch.argmax(out, dim=0)
            #         pred = int(pred.item())
            #         token_list.append(pred)

            #         if pred != 0:
            #             token = torch.LongTensor([[pred]])

            #             if enc_state.is_cuda:
            #                 token = token.cuda()

            #             dec_state, hidden = self.decoder(token, hidden=hidden)

            #     return token_list, token_list_b

            # # print(enc_state.shape, inputs_length)
            # preds_rnnt = -1.0*torch.ones((enc_state.shape[0], enc_state.shape[1]), device=enc_state.device)
            # preds_rnnt_b = preds_rnnt.clone()
            # for j in range(x.shape[0]):
            #     # print(enc_state.shape, inputs_length[i])
            #     pred, pred_b = decode(enc_state[j].clone().detach(), inputs_length[j].int())
            #     # print(j, len(pred), len(pred_b))
            #     preds_rnnt[j, :inputs_length[j].int()] = torch.tensor(pred)
            #     preds_rnnt_b[j, :inputs_length[j].int()]= torch.tensor(pred_b)
            #     # print(targets[j], preds, inputs_length[j], targets.shape, enc_state.shape);exit()
            # # print(x.shape, (preds_rnnt).shape, (preds_rnnt_b).shape)
            # # print((preds_rnnt).int()[1], (preds_rnnt_b)[1])

            # # rnnt binary pred lab
            # logits_enc_idx_by_dec = preds_rnnt_b#torch.tensor(preds_rnnt_b, device=enc_state.device)
            
            # # rnnt pred lab
            # logits_enc_idx_all = preds_rnnt#torch.tensor(preds_rnnt, device=enc_state.device)
            # logits_enc_idx = logits_enc_idx_all.clone()
            # logits_enc_idx[logits_enc_idx>0]=1

            # ##dec pred zhidao enc
            # loss_enc_binary = self.loss(logits_enc_binary, logits_enc_idx.detach())
            # if((logits_enc_idx==0).all() ):
            #     loss_enc_binary0 = torch.Tensor([0]).cuda()
            # else:
            #     loss_enc_binary0 = self.loss0(logits_enc_binary, logits_enc_idx.detach(), 2, 1)

            # ##dec zhidao enc
            # loss_enc_binary_tea = self.loss(logits_enc_binary, logits_enc_idx_by_dec.detach())
            # if((logits_enc_idx_by_dec==0).all() ):
            #     loss_enc_binary0_tea = torch.Tensor([0]).cuda()
            # else:
            #     loss_enc_binary0_tea = self.loss0(logits_enc_binary, logits_enc_idx_by_dec.detach(), 2, 1)

            # ##xjb zhidao
            # # print("logits_enc_idx_all: ", log_probs.shape, logits_enc_idx_all.shape, logits_final.shape, logits_idx.detach().shape)
            # loss_enc_ctc_zhidao = self.loss(log_probs.reshape(-1,log_probs.shape[-1]), logits_enc_idx_all.detach())


            loss_dict = {}
            # loss_dict["rnnt_loss"]=loss
            # if "label_ctc_ph" in meta:
            #     phone_ctc_logit = F.log_softmax(phone_ctc_logit,dim=-1)
            #     # print(phone_ctc_logit.shape, label_ctc.shape);exit()    # torch.Size([6, 319, 55]) torch.Size([6, 319])
            #     # print(label_ctc.shape, phone_ctc_logit.shape, inputs_length, label_ctc_length)  #torch.Size([18, 212]) torch.Size([18, 53, 49]) torch.Size([18]) torch.Size([18])                
            #     phone_ctc_loss = self.loss(phone_ctc_logit.reshape(-1, self.phone_dim).contiguous(),     #[T,B,C]
            #                             label_ctc.contiguous(),    #[B,S]
            #                         )
            #     loss_dict["phone_ctc_loss"]=phone_ctc_loss  #*0.1
            # loss_dict["loss_binary"]=loss_binary
            # loss_dict["loss_binary0"]=loss_binary0
            loss_dict["loss_ctc"]=loss_ctc  #*0.1
            # loss_dict["loss_enc_binary_ctc"]=loss_enc_binary_ctc    *0.1
            # loss_dict["loss_enc_binary_ctc0"]=loss_enc_binary_ctc0  *0.1

            # loss_dict["loss_enc_binary"]=loss_enc_binary #*0.1
            # loss_dict["loss_enc_binary0"]=loss_enc_binary0 #*0.1
            # loss_dict["loss_enc_binary_tea"]=loss_enc_binary_tea #*0.1
            # loss_dict["loss_enc_binary0_tea"]=loss_enc_binary0_tea #*0.1
            # loss_dict["loss_enc_ctc_zhidao"]=loss_enc_ctc_zhidao #*0.1

            return loss_dict
        else:
            # print('valid pprocessing.........')
            batch_size = b
            zero_token = torch.LongTensor([[0]])
            if x.is_cuda:
                zero_token = zero_token.cuda()

            def decode(enc_state, lengths):
                token_list = []
                dec_state, hidden = self.decoder(zero_token)
                for t in range(lengths):
                    logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                    logits = logits[:-1]
                    out = F.softmax(logits, dim=0).detach()
                    pred = torch.argmax(out, dim=0)
                    pred = int(pred.item())

                    if pred != 0:
                        token_list.append(pred)
                        token = torch.LongTensor([[pred]])

                        if enc_state.is_cuda:
                            token = token.cuda()

                        dec_state, hidden = self.decoder(token, hidden=hidden)
                return token_list

            def decode_skip(enc_state, lengths, enc_blank_logit):
                count_enc = 0
                count_dec = 0
                token_list = []
                dec_state, hidden = self.decoder(zero_token)

                # print(enc_blank_logit.shape);exit()
                enc_blank_logit = enc_blank_logit.softmax(-1)

                for t in range(lengths):
                    if enc_blank_logit[t][0] >float(0.9):
                        count_enc+=1
                        continue
                    logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))

                    out_b = F.softmax(torch.cat((logits[:1],logits[-1:])), dim=0).detach()        
                    if out_b[0] >float(0.9):
                        count_dec+=1
                        continue

                    logits = logits[:-1]                    
                        
                    out = F.softmax(logits, dim=0).detach()
                    pred = torch.argmax(out, dim=0)
                    pred = int(pred.item())

                    if pred != 0:
                        token_list.append(pred)
                        token = torch.LongTensor([[pred]])

                        if enc_state.is_cuda:
                            token = token.cuda()

                        dec_state, hidden = self.decoder(token, hidden=hidden)
                return token_list, count_enc, count_dec

            results = []
            results_skip = []
            skip_counts_enc = []
            skip_counts_dec = []
            total_length = []
            for i in range(batch_size):
                decoded_seq = decode(enc_state[i], inputs_length[i].int())
                results.append(decoded_seq)
                
                decoded_skip_seq, skip_count_enc, skip_count_dec = decode_skip(enc_state[i], inputs_length[i].int(), logits_enc_binary)
                results_skip.append(decoded_skip_seq)
                skip_counts_enc.append(skip_count_enc)
                skip_counts_dec.append(skip_count_dec)
                total_length.append(inputs_length[0])

            targets[targets<-1] = -1
            targets = targets.flatten()
            targets = targets[targets != -1]
            num_words = len(targets)-1
            self.total_word += num_words
 
            # print("cv: ", targets)
            ##rnnt acc
            dist = edit_dist(targets, results[0])

            self.total_dist += dist

            cer = dist / num_words * 100
            acc = 100 - cer

            ##rnnt skip acc
            dist_skip = edit_dist(targets, results_skip[0])

            self.total_dist_skip += dist_skip

            cer_skip = dist_skip / num_words * 100
            acc_skip = 100 - cer_skip

            ##rnnt skip rate
            self.total_skip_count_enc += skip_counts_enc[0]
            self.total_skip_count_dec += skip_counts_dec[0]
            self.total_t_length += total_length[0]
            skip_rate_enc = skip_counts_enc[0]/total_length[0]
            skip_rate_dec = skip_counts_dec[0]/total_length[0]


            ##dec 2class acc
            # acc_blank = self.accuracy(logits_final, logits_idx)
            # if((logits_idx==0).all()):
            #     acc_blank0 = torch.Tensor([0])
            # else:
            #     acc_blank0 = self.accuracy0(logits_final, logits_idx)

            ##ctc acc
            ctc_acc, _, _ = self.acc_ctc(logits_enc_ctc_post.squeeze(), targets)
            ctc_acc = ctc_acc*100

            ##enc 2class acc
            # enc_acc_blank = self.accuracy(logits_enc_binary, logits_enc_idx_by_ctc)
            # if((logits_enc_idx_by_ctc==0).all()):
            #     enc_acc_blank0 = torch.Tensor([0])
            # else:
            #     enc_acc_blank0 = self.accuracy0(logits_enc_binary, logits_enc_idx_by_ctc)

            acc_dict = {}
            # acc_dict['rnnt_acc'] =acc
            # acc_dict['rnnt_skip_acc'] =acc_skip
            # acc_dict['skip_rate_enc'] =skip_rate_enc
            # acc_dict['skip_rate_dec'] =skip_rate_dec
            # if "label_ctc_ph" in meta:
            #     phctc_acc = self.accuracy(phone_ctc_logit.permute(1,0,2).reshape(-1, self.phone_dim).contiguous(), label_ctc)
            #     # print(phctc_acc); exit()
            #     acc_dict['phctc_acc'] =phctc_acc*100

            # acc_dict['dec_acc_blank'] =acc_blank
            # acc_dict['dec_acc_blank0'] =acc_blank0
            acc_dict['ctc_acc'] =ctc_acc
            # acc_dict['enc_acc_blank'] =enc_acc_blank
            # acc_dict['enc_acc_blank0'] =enc_acc_blank0

            return acc_dict

class trainer(Trainer):
    def forward(self, index, data_element, model):
        data, meta = data_element
        data = data.cuda()
        for key in meta:
            if isinstance(meta[key], int) or meta[key]==None:
                continue
            meta[key] = meta[key].cuda()
        celoss = model(data, meta)
        print("hjwang11-celoss:", celoss)
        return celoss
