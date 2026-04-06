# by pcli2, innovated by dlp/jieding's source code
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from .pfile import Pfileinfo, PfileChunkReader, Normfile

# add by mzwang7, for read txt
def collate_fn_txt(batch):
    return batch[0]

# add by mzwang7, for read txt
def txtDataLoader(fp_txt, fp_dict, batch_size, start_sent, end_sent, ndivide=1, divide_index=0, shuffle_batch=True, num_workers=1):
    pfile_dataset = txtDataset(fp_txt, fp_dict, batch_size, start_sent, end_sent, ndivide, divide_index, shuffle_batch)
    return DataLoader(pfile_dataset, batch_size=1, num_workers=num_workers, multiprocessing_context="spawn", collate_fn=collate_fn_txt)

# add by mzwang7, for read txt
class txtDataset(Dataset):
    def __init__(self, fp_txt, fp_dict, batch_size, start_sent, end_sent, ndivide, divide_index, shuffle_batch=True):
        self.batch_size=batch_size
        self.w2i = {}
        
        with open(fp_dict, "r", encoding="GBK") as f:
            for i, l in enumerate(f):
                self.w2i[l.strip()] = i
                
        with open(fp_txt) as f:
            self.data=list(f.readlines())
        
        end_sent = min(len(self.data), end_sent)
        num_sentences = (end_sent - start_sent)// ndivide
        self.batch_num=num_sentences//batch_size
        self.read_startindex = num_sentences  * divide_index + start_sent
        self.read_endindex = self.read_startindex + num_sentences if divide_index < ndivide - 1 else end_sent
        if shuffle_batch:random.shuffle(self.data[self.read_startindex:self.read_endindex])
        
        
    def __getitem__(self, index):
        x=[]
        y=[]
        maxl=0
        sos=self.w2i['<s>']
        eos=self.w2i['</s>']
        unk=self.w2i['<unk>']
        for i in range(self.read_startindex+index*self.batch_size,min(self.read_startindex+(index+1)*self.batch_size,self.read_endindex)):
            wi=[]
            for w in self.data[i].strip().split():
                if w in self.w2i:
                    wi.append(self.w2i[w])
                else:
                    wi.append(unk)
            xi=wi.copy()
            yi=wi.copy()
            xi.insert(0,sos)
            yi.append(eos)
            x.append(xi)
            y.append(yi)
            if len(xi)>maxl:maxl=len(xi)
        x_pad=np.full((self.batch_size,maxl),-1)
        y_pad=np.full((self.batch_size,maxl),-1)
        mask=np.full((self.batch_size,maxl),0)
        for i in range(len(x)):
            l=len(x[i])
            x_pad[i,:l]=x[i]
            mask[i,:l]=1
            y_pad[i,:l]=y[i]
        return torch.from_numpy(x_pad.T).float().contiguous(), torch.from_numpy(mask.T).unsqueeze(2).float().contiguous(), torch.from_numpy(y_pad.T).float().contiguous()

    def __len__(self):
        return self.batch_num

def PfileDataLoader(file_fea, file_lab, file_norm, batch_num, bunchsize, maxsentframe, maxnumsent, start_sent, end_sent, ndivide=1, divide_index=0, nmod_pad=64, 
                    shuffle_batch=True, num_workers=1, cachesize=10000, random_seed=0):
    fea_pfileinfo = Pfileinfo(file_fea)
    lab_pfileinfo = Pfileinfo(file_lab) if file_lab is not None else None
    if file_lab is not None:
        assert fea_pfileinfo.num_sentences == lab_pfileinfo.num_sentences, "number of sentences in feature and label are not equal"
    num_sentences = fea_pfileinfo.num_sentences
    assert start_sent < num_sentences, "start_sent must smaller than total sentences {0}".format(num_sentences)
    end_sent = min(num_sentences, end_sent)

    num_sentences = end_sent - start_sent
    read_startindex = int(num_sentences / ndivide * divide_index) + start_sent
    read_endindex = read_startindex + int(num_sentences / ndivide) if divide_index < ndivide - 1 else end_sent
    pfile_dataset = PfileDataset(file_fea, file_lab, file_norm, read_startindex, read_endindex, batch_num, bunchsize, maxsentframe, 
                                 maxnumsent, nmod_pad, shuffle_batch, cachesize, random_seed)
    # pfile_dataset.pfile_chunk_reader = PfileChunkReader(fea_pfileinfo, lab_pfileinfo, start_sent, end_sent, 
    #                                               bunchsize, maxsentframe, maxnumsent, 
    #                                               nmod_pad, cachesize, shuffle_batch, random_seed)

    return DataLoader(pfile_dataset, batch_size=1, num_workers=num_workers, multiprocessing_context="spawn", collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    #return DataLoader(pfile_dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_fn)

def DebugPfileDataLoader(file_fea, file_lab, file_norm, batch_num, bunchsize, maxsentframe, maxnumsent, start_sent, end_sent, ndivide=1, divide_index=0, nmod_pad=64, 
                    shuffle_batch=True, num_workers=0, cachesize=10000, random_seed=0):
    fea_pfileinfo = Pfileinfo(file_fea)
    lab_pfileinfo = Pfileinfo(file_lab) if file_lab is not None else None
    if file_lab is not None:
        assert fea_pfileinfo.num_sentences == lab_pfileinfo.num_sentences, "number of sentences in feature and label are not equal"
    num_sentences = fea_pfileinfo.num_sentences
    assert start_sent < num_sentences, "start_sent must smaller than total sentences {0}".format(num_sentences)
    end_sent = min(num_sentences, end_sent)

    num_sentences = end_sent - start_sent
    read_startindex = int(num_sentences / ndivide * divide_index) + start_sent
    read_endindex = read_startindex + int(num_sentences / ndivide) if divide_index < ndivide - 1 else end_sent
    pfile_dataset = PfileDataset(file_fea, file_lab, file_norm, read_startindex, read_endindex, batch_num, bunchsize, maxsentframe, 
                                 maxnumsent, nmod_pad, shuffle_batch, cachesize, random_seed)
    pfile_dataset.lab_pfileinfo = lab_pfileinfo
    pfile_dataset.pfile_chunk_reader = PfileChunkReader(fea_pfileinfo, lab_pfileinfo, start_sent, end_sent, 
                                                  bunchsize, maxsentframe, maxnumsent, 
                                                  nmod_pad, cachesize, shuffle_batch, random_seed)

    return DataLoader(pfile_dataset, batch_size=1, num_workers=0, collate_fn=collate_fn)

def TestPfileDataLoader(file_fea, file_lab, file_norm, batch_num, bunchsize, maxsentframe, maxnumsent, start_sent, end_sent, ndivide=1, divide_index=0, nmod_pad=64, 
                    shuffle_batch=False, num_workers=1, cachesize=10000, random_seed=0):
    fea_pfileinfo = Pfileinfo(file_fea)
    lab_pfileinfo = Pfileinfo(file_lab) if file_lab is not None else None
    if file_lab is not None:
        assert fea_pfileinfo.num_sentences == lab_pfileinfo.num_sentences, "number of sentences in feature and label are not equal"
    num_sentences = fea_pfileinfo.num_sentences
    assert start_sent < num_sentences, "start_sent must smaller than total sentences {0}".format(num_sentences)
    end_sent = min(num_sentences, end_sent)

    num_sentences = end_sent - start_sent
    read_startindex = int(num_sentences / ndivide * divide_index) + start_sent
    read_endindex = read_startindex + int(num_sentences / ndivide) if divide_index < ndivide - 1 else end_sent
    pfile_dataset = TestPfileDataset(file_fea, file_lab, file_norm, read_startindex, read_endindex, batch_num, bunchsize, maxsentframe, 
                                 maxnumsent, nmod_pad, shuffle_batch, cachesize, random_seed)

    return DataLoader(pfile_dataset, batch_size=1, num_workers=num_workers, multiprocessing_context="spawn", collate_fn=collate_fn, worker_init_fn=worker_init_fn)

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
    def __init__(self, file_fea, file_lab, file_norm, start_index, end_index, batch_num, bunchsize, maxsentframe, maxnumsent, 
                 nmod_pad=64, shuffle_batch=True, cachesize=10000, random_seed=0):
        self.file_fea = file_fea
        self.file_lab = file_lab
        norm = Normfile(file_norm)
        mean = torch.from_numpy(norm.mean)
        mean = mean.reshape(1, 1, -1, 1)
        var = torch.from_numpy(norm.var)
        var = var.reshape(1, 1, -1, 1)
        self.mean = mean
        self.var = var
        self.bunchsize = bunchsize
        self.start_index = start_index
        self.end_index = end_index
        self.batch_num = batch_num
        self.maxsentframe = maxsentframe
        self.maxnumsent = maxnumsent
        self.nmod_pad = nmod_pad
        self.shuffle_batch = shuffle_batch
        self.cachesize = cachesize
        self.pfile_chunk_reader = None
        self.random_seed = random_seed
    def __getitem__(self, index):
        if self.lab_pfileinfo is None:
            data = self.pfile_chunk_reader.getbatch()
            data, data_mask = self.__pad_nmod(data, self.nmod_pad, 0)
            meta = {}
            meta["mask"] = data_mask
            return data, meta
        else:
            data, label = self.pfile_chunk_reader.getbatch()

            if data[0].shape[1] == 160:
                data = [data[0].reshape(-1, 40)]

            for idx, element in enumerate(label):
                if element[0,0] != 14837:
                    label[idx] = np.insert(element[:-1,:], 0, 14837).reshape(-1, 1)

            data, data_mask = self.__pad_nmod(data, self.nmod_pad, 0)
            data = data - self.mean
            data = data * self.var
            label, _ = self.__pad_nmod(label, self.nmod_pad, -1)

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
            meta["att_label"] = label.long().contiguous()
            meta["att_mask"] = label_mask.float().contiguous()
            meta['w'] = torch.Tensor([data.shape[3]])
            meta["rnn_mask"] = data_mask.transpose(1, 0).unsqueeze(2).contiguous()
            meta["inputs_length"] = get_length(meta["rnn_mask"], self.nmod_pad).contiguous()
            meta["targets_length"] = get_length(meta["att_mask"]).contiguous()
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

class TestPfileDataset(Dataset):
    def __init__(self, file_fea, file_lab, file_norm, start_index, end_index, batch_num, bunchsize, maxsentframe, maxnumsent, 
                 nmod_pad=64, shuffle_batch=False, cachesize=10000, random_seed=0):
        self.file_fea = file_fea
        self.file_lab = file_lab
        norm = Normfile(file_norm)
        mean = torch.from_numpy(norm.mean)
        mean = mean.reshape(1, 1, -1, 1)
        var = torch.from_numpy(norm.var)
        var = var.reshape(1, 1, -1, 1)
        self.mean = mean
        self.var = var
        self.bunchsize = bunchsize
        self.start_index = start_index
        self.end_index = end_index
        self.batch_num = batch_num
        self.maxsentframe = maxsentframe
        self.maxnumsent = maxnumsent
        self.nmod_pad = nmod_pad
        self.shuffle_batch = shuffle_batch
        self.cachesize = cachesize
        self.pfile_chunk_reader = None
        self.random_seed = random_seed
    def __getitem__(self, index):
        if self.lab_pfileinfo is None:
            data = self.pfile_chunk_reader.getbatch()
            data, data_mask = self.__pad_nmod(data, self.nmod_pad, 0)
            meta = {}
            meta["mask"] = data_mask
            return data, meta
        else:
            data, label = self.pfile_chunk_reader.getbatch()

            if data[0].shape[1] == 160:
                data = [data[0].reshape(-1, 40)]

            data, data_mask = self.__pad_nmod(data, self.nmod_pad, 0)
            data = data - self.mean
            data = data * self.var
            label, _ = self.__pad_nmod(label, self.nmod_pad, -1)

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
            meta["att_label"] = label.long().contiguous()
            meta["att_mask"] = label_mask.float().contiguous()
            meta['w'] = torch.Tensor([data.shape[3]])
            meta["rnn_mask"] = data_mask.transpose(1, 0).unsqueeze(2).contiguous()
            meta["inputs_length"] = get_length(meta["rnn_mask"], self.nmod_pad).contiguous()
            meta["targets_length"] = get_length(meta["att_mask"]).contiguous()
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
    
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers
    worker_idx = worker_info.id
    dataset = worker_info.dataset
    dataset.fea_pfileinfo = Pfileinfo(dataset.file_fea)
    dataset.lab_pfileinfo = Pfileinfo(dataset.file_lab) if dataset.file_lab is not None else None
    start_index = dataset.start_index
    end_index = dataset.end_index
    sentence_per_worker = int((end_index - start_index) / num_workers)
    worker_start = start_index + sentence_per_worker * worker_idx
    worker_end = min(worker_start + sentence_per_worker, end_index)
    dataset.pfile_chunk_reader = PfileChunkReader(dataset.fea_pfileinfo, dataset.lab_pfileinfo, worker_start, worker_end, 
                                                  dataset.bunchsize, dataset.maxsentframe, dataset.maxnumsent, 
                                                  dataset.nmod_pad, dataset.cachesize, dataset.shuffle_batch, dataset.random_seed)

def collate_fn(batch):
    data, meta = batch[0]
    return data, meta
