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
from asr.data import clip_mask, cnn2rnn, rnn2cnn
from typing import List, Tuple, Dict
from asr.functions import xavier
from asr.train import Trainer
import struct
from warp_rnnt import rnnt_loss

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
        num_frame2 = f_out_fea.shape[0]  # 32位
        step_in_ns = 100000  # 32位
        dim_frame2 = f_out_fea.shape[1] * 4  # 16位
        fea_format = 9  # 16位
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
        self.relu = nn.LeakyReLU(negative_slope=0.1)
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
            self.activation = nn.LeakyReLU(negative_slope=relu_value)
        if act_type == "Tanh":
            self.activation = nn.Tanh()
        if act_type == "None":
            self.activation = NullModule()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
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
        self.relu01 = nn.LeakyReLU(negative_slope=0.1)
        self.conv02 = nn.Conv2d(256, 256, (1,3), (1,1), (0,2), groups =1, dilation= (1,2))
        xavier(self.conv02.weight)
        nn.init.constant_(self.conv02.bias.data, 0)
        self.bn02=nn.BatchNorm2d(256, momentum=0.01)
        self.relu02 = nn.LeakyReLU(negative_slope=0.1)
        self.conv03 = nn.Conv2d(256, 256, (1,3), (1,1), (0,2), groups =1, dilation= (1,2))
        xavier(self.conv03.weight)
        nn.init.constant_(self.conv03.bias.data, 0)
        self.bn03=nn.BatchNorm2d(256, momentum=0.01)
        self.relu03 = nn.LeakyReLU(negative_slope=0.1)
        self.conv04 = nn.Conv2d(256, 256, (1,3), (1,1), (0,2), groups =1, dilation= (1,2))
        xavier(self.conv04.weight)
        nn.init.constant_(self.conv04.bias.data, 0)
        self.bn04=nn.BatchNorm2d(256, momentum=0.01)
        self.relu04 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, xx: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:

        x = self.conv1(xx)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)

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
        x = self.bn01(x)
        x = self.relu01(x)
        x = self.conv02(x)
        x = self.bn02(x)
        x = self.relu02(x)
        x = self.conv03(x)
        x = self.bn03(x)
        x = self.relu03(x)
        x = self.conv04(x)
        x = self.bn04(x)
        x = self.relu04(x)

        x = x.permute(0,3,2,1).reshape(x.shape[0],x.shape[3],x.shape[1]) #nchw -> ntd

        return x

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
        vocab_size  = 14840
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

class Transducer(nn.Module):
    def __init__(self):
        super(Transducer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.joint   = JointNet(
            input_size=512,
            inner_dim=256,
            vocab_size=14841
            )

        self.accuracy = ACC()
        self.loss = CeLoss()
        self.loss0 = CeLoss0()
        self.accuracy0 = ACC0()

        self.total_dist = 0
        self.total_word = 0

        self.num_samples = 0
        self.sum_acc = 0
        
    def forward(self, x: torch.Tensor, meta:Dict[str, torch.Tensor]):        
        if self.training:
            x = mask_data_f(x)
        (b, c, f, t) = x.size()

        targets        = meta["att_label"].permute(1,0).contiguous()
        targets[targets == -1] = -2
        targets = targets.add(1)

        enc_state = self.encoder(x, meta)

        att_mask = meta["att_mask"].permute(1,0).contiguous()
        targets_length = att_mask.sum(1)

        data_mask = clip_mask(meta["rnn_mask"], enc_state.size(1), 0)
        data_mask = data_mask.squeeze(2).permute(1,0)
        inputs_length =  data_mask.sum(1)

        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0) #add start label zero
        concat_targets_length = targets_length.add(1)

        dec_state,_ = self.decoder(concat_targets, concat_targets_length)

        logits_tmp  = self.joint(enc_state, dec_state) #BT(U+1)D

        logits_post = logits_tmp[:,:,:,:-1]    
        logits_post_lab = logits_post.reshape(logits_post.shape[0],-1,logits_post.shape[-1])
        
        (logits_pred, logits_idx) = torch.max(logits_post_lab, dim=2)
        logits_idx[logits_idx>0]=1
        logits_final = torch.cat((logits_tmp[:,:,:,0:1], logits_tmp[:,:,:,-1:]), dim=-1)
        logits_final = logits_final.reshape(-1,logits_final.shape[-1])

        if self.training:   
            
            # modified by qhxu3 - start.
            # use warp-rnnt instead of warp-transducer, which is faster theoretically and has fastemit built-in.
            logits_rnnt = torch.log_softmax(logits_post, -1)
            loss = rnnt_loss(
                logits_rnnt.contiguous().cuda(),
                targets.cuda().int(),
                inputs_length.int(), 
                targets_length.int(),
                reduction='mean',
                fastemit_lambda=0
            )
            # modified by qhxu3 - end.

            loss_binary = self.loss(logits_final, logits_idx.detach())
            if((logits_idx==0).all() ):
                loss_binary0 = torch.Tensor([0]).cuda()
            else:
                loss_binary0 = self.loss0(logits_final, logits_idx.detach(), 2, 1)

            return loss, loss_binary0, loss_binary
        else:
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

            results = []
            for i in range(batch_size):
                decoded_seq = decode(enc_state[i], inputs_length[i].int())
                results.append(decoded_seq)

            targets[targets<-1] = -1
            targets = targets.flatten()
            targets = targets[targets != -1]
            dist = edit_dist(targets, results[0])
            num_words = len(targets)-1

            self.total_dist += dist
            self.total_word += num_words

            cer = self.total_dist / self.total_word * 100
            acc = 100 - cer

            return acc, torch.Tensor([0]), torch.Tensor([0])

class trainer(Trainer):
    def forward(self, index, data_element, model):
        data, meta = data_element
        data = data.cuda()
        for key in meta:
            if isinstance(meta[key], int) or meta[key]==None:
                continue
            meta[key] = meta[key].cuda()
        celoss = model(data, meta)
        return celoss