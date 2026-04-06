# by pcli2, innovated by dlp/jieding's source code
import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader, Dataset
from asr.data import PfileInfo


class Normfile():
    def __init__(self, filename):
        assert os.path.isfile(filename), "file not exists {0}".format(filename)
        fp = open(filename, 'r')
        content = fp.readlines()
        fp.close()
        self.mean_dim = int(content[0].split()[-1])
        self.var_dim = int(content[self.mean_dim+1].split()[-1])
        assert self.mean_dim == self.var_dim, "{0}: mean dim {1} != var dim {2}".format(filename, self.mean_dim, self.var_dim)
        assert self.mean_dim + self.var_dim + 2 == len(content), "norm file error: {0}".format(filename)
        self.mean = np.array([float(line[:-1]) for line in content[1:self.mean_dim+1]], dtype=np.float32)
        self.var = np.array([float(line[:-1]) for line in content[self.mean_dim+2:]], dtype=np.float32)


def TestPfileDataLoader(file_fea, file_lab, file_norm, batch_num, nmod_pad=64, start_sent=0, end_sent=1):
    fea_pfileinfo = PfileInfo(file_fea)
    lab_pfileinfo = PfileInfo(file_lab) if file_lab is not None else None

    if file_lab is not None:
        assert fea_pfileinfo.num_sentences == lab_pfileinfo.num_sentences, "number of sentences in feature and label are not equal"

    pfile_dataset = PfileDataset(file_fea, file_lab, file_norm, batch_num, nmod_pad, start_sent, end_sent)
    
    return DataLoader(pfile_dataset, batch_size=1, num_workers=0, collate_fn=collate_fn)



def get_length(data, nmod=1):
    t= data.size(0)
    s = data.size(1)
    lengths= torch.zeros(( s), dtype=torch.float32, device=data.device)
    for j in range(s):
        for i in range(t):
            if((data[t-1-i,j] == 1)):
                lengths[j]=int( (t-i+nmod-1)/nmod )
                break
    return lengths



class PfileDataset(Dataset):
    def __init__(self, file_fea, file_lab, file_norm, batch_num, nmod_pad=64, start_sent=0, end_sent=1):
        fea_pfileinfo = PfileInfo(file_fea)
        if file_lab is not None:
            lab_pfileinfo = PfileInfo(file_lab)
        norm = Normfile(file_norm)
        mean = torch.from_numpy(norm.mean)
        mean = mean.reshape(1, 1, -1, 1)
        var = torch.from_numpy(norm.var)
        var = var.reshape(1, 1, -1, 1)
        self.file_fea = file_fea
        self.file_lab = file_lab
        self.mean = mean
        self.var = var
        self.batch_num = batch_num
        self.nmod_pad = nmod_pad
        self.feature_cache = []
        if file_lab is not None:
            self.label_cache = []
        self.start_sent = start_sent
        self.end_sent = end_sent

        binary_start = 0
        binary_end = 0
        dtype = np.dtype("float32")
        fp = open(file_fea, "rb")
        size = fea_pfileinfo.seq_info[end_sent-1][1] + fea_pfileinfo.seq_info[end_sent-1][2] - fea_pfileinfo.seq_info[start_sent][1]
        start = fea_pfileinfo.seq_info[start_sent][1]
        fp.seek(fea_pfileinfo.header_size + len(fea_pfileinfo.data_format)*dtype.itemsize*start)
        raw_binary = fp.read(len(fea_pfileinfo.data_format)*dtype.itemsize*size)
        for i in range(start_sent, end_sent):
            repeat_format = fea_pfileinfo.seq_info[i][2] * fea_pfileinfo.data_format
            binary_start = binary_end
            binary_end = binary_end + len(fea_pfileinfo.data_format) * dtype.itemsize * fea_pfileinfo.seq_info[i][2]
            value = np.frombuffer(raw_binary[binary_start: binary_end], dtype=dtype)
            value = value.byteswap()
            value = np.reshape(value, [-1, fea_pfileinfo.frame_length])
            value = value[:, fea_pfileinfo.real_data_start:]
            self.feature_cache.append(value)
        fp.close()

        if file_lab is not None:
            binary_start = 0
            binary_end = 0
            dtype = np.dtype("int32")
            fp = open(file_lab, "rb")
            size = lab_pfileinfo.seq_info[end_sent-1][1] + lab_pfileinfo.seq_info[end_sent-1][2] - lab_pfileinfo.seq_info[start_sent][1]
            start = lab_pfileinfo.seq_info[start_sent][1]
            fp.seek(lab_pfileinfo.header_size + len(lab_pfileinfo.data_format)*dtype.itemsize*start)
            raw_binary = fp.read(len(lab_pfileinfo.data_format)*dtype.itemsize*size)
            for i in range(start_sent, end_sent):
                repeat_format = lab_pfileinfo.seq_info[i][2] * lab_pfileinfo.data_format
                binary_start = binary_end
                binary_end = binary_end + len(lab_pfileinfo.data_format) * dtype.itemsize * lab_pfileinfo.seq_info[i][2]
                value = np.frombuffer(raw_binary[binary_start: binary_end], dtype=dtype)
                value = value.byteswap()
                value = np.reshape(value, [-1, lab_pfileinfo.frame_length])
                value = value[:, lab_pfileinfo.real_data_start:]
                self.label_cache.append(value)
            fp.close()


    def __getitem__(self, index):
        label_ctc = []
        data = [self.feature_cache.pop(0)]
        if self.file_lab is not None:
            label = [self.label_cache.pop(0)]
            
        label_length = label[0].shape[1]
            
        if label[0].shape[1] == 2:
            for idx, element in enumerate(label):
                e1, e2 = np.split(element, 2, axis=1)
                label_ctc.append(e2)
                label[idx] = e1 # xiaobao
                
        data, data_mask = self.__pad_nmod(data, self.nmod_pad, 0)
        data = data - self.mean
        data = data * self.var
        if self.file_lab is not None:
            label, _ = self.__pad_nmod(label, None, -1)
            
            if label_length == 2:
                label_ctc, _ = self.__pad_nmod(label_ctc, self.nmod_pad, -1)

            label_mask = label.clone()
            label_mask[label_mask >= 0] = 1
            label_mask[label_mask < 0] = 0
            maxlen = label_mask.sum(3).max()
            label = label[:, :, :, :maxlen]
            label_mask = label_mask[:, :, :, :maxlen]
            label = label.squeeze(1).squeeze(1)
            label = label.transpose(1, 0)
            label_mask = label_mask.squeeze(1).squeeze(1)
            label_mask = label_mask.transpose(1, 0)
        meta = {}
        meta["mask"] = data_mask.contiguous()
        if self.file_lab is not None:
            meta["att_label"] = label.long().contiguous()
            meta["att_mask"] = label_mask.float().contiguous()  
            meta["targets_length"] = get_length(meta["att_mask"], self.nmod_pad).contiguous()     
        meta['w'] = torch.Tensor([data.shape[3]])
        meta["rnn_mask"] = data_mask.transpose(1, 0).unsqueeze(2).contiguous()
        meta["inputs_length"] = get_length(meta["rnn_mask"], self.nmod_pad).contiguous()    
        
        if label_length == 2:
            # torch.set_printoptions(threshold=100000000)
            label_ctc = label_ctc.squeeze()
            if label_ctc.dim() == 1:
                label_ctc = label_ctc.unsqueeze(0)
            # label_ctc_ph = torch.full_like(label_ctc, -1)
            # # 遍历tensor的每一行
            # for i in range(label_ctc.size(0)):
            #     # 初始化一个索引列表，用于记录非重复元素的插入位置
            #     indices = []
            #     # 初始化一个变量来存储上一个非重复元素的值
            #     prev_value = None
                
            #     # 遍历tensor的当前行
            #     for j in range(label_ctc.size(1)):
            #         if label_ctc[i, j] == 53 or label_ctc[i, j] == -1:
            #             continue
            #         # 如果当前元素和上一个非重复元素不同
            #         if label_ctc[i, j] != prev_value:
            #             # 将当前元素的索引添加到列表中
            #             indices.append(j)
            #             # 更新上一个非重复元素的值
            #             prev_value = label_ctc[i, j]
                
            #     # 复制非重复元素到结果tensor的相应位置
            #     label_ctc_ph[i, 0] = 55
            #     for k, idx in enumerate(indices):
            #         label_ctc_ph[i, k+1] = label_ctc[i, idx]
            #     label_ctc_ph[i, len(indices)+1] = 56
            # # print(data.shape,label_ctc.shape, label_ctc_ph.shape, label_ctc[0]);            print(label_ctc_ph[0]);            import time;time.sleep(1)
            # #exit()
            label_ctc_ph = label_ctc

            label_ctc_ph = label_ctc_ph.permute(1,0)
            label_ctc_ph_mask = label_ctc_ph.clone()
            label_ctc_ph_mask[label_ctc_ph_mask >= 0] = 1
            label_ctc_ph_mask[label_ctc_ph_mask < 0] = 0
            # print(data.shape,label.shape,label_ctc.shape);exit()    #torch.Size([24, 18]) torch.Size([18, 1, 40, 400])

            meta["label_ctc_ph"] = label_ctc_ph.long().contiguous()
            meta["label_ctc_ph_mask"] = label_ctc_ph_mask.long().contiguous()

        return data, meta
            
        

    def __pad_nmod(self, sequence, nmod, val):
        maxlen = 0
        pad_list = []
        mask_list = []
        for nparray in sequence:
            if nparray.shape[0] > maxlen:
                maxlen = nparray.shape[0]
        if nmod is not None:
            maxlen = maxlen if maxlen % nmod == 0 else maxlen + nmod - maxlen % nmod
        for nparray in sequence:
            padlen = maxlen - nparray.shape[0]
            nparray = np.pad(nparray, ((0, padlen), (0, 0)), mode="constant", constant_values=(val,))
            nparray = nparray.transpose(1, 0)
            torcharray = torch.from_numpy(nparray)
            torchmask = torch.ones(1, torcharray.size()[1])
            if padlen > 0:
                torchmask[:, -padlen:] = 0
            torcharray = torcharray.reshape(1, 1, torcharray.size()[0], torcharray.size()[1])
            pad_list.append(torcharray)
            mask_list.append(torchmask)

        batch_array = torch.cat(pad_list, dim=0)
        batch_mask = torch.cat(mask_list, dim=0)
        return batch_array, batch_mask

    def __len__(self):
        return self.batch_num

def collate_fn(batch):
    data, meta = batch[0]
    return data, meta
