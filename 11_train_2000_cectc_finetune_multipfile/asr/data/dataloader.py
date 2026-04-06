# by pcli2, innovated by dlp/jieding's source code
import torch
import numpy as np
import random
import math
from torch.utils.data import DataLoader, Dataset
from .union_reader import PfileInfo, Normfile
from itertools import groupby, repeat


class PfileChunkReader():
    def __init__(self, pfileinfo, pfilelabinfo, start_sent, end_sent, bunchsize, maxsentframe=1000, maxnumsent=1000, nmod_pad=64, cachesize=10000, shuffle=True, random_seed=0):
        # pcli2: assume pfileinfo checking is done
        self.pfileinfo = pfileinfo
        self.pfilelabinfo = pfilelabinfo
        self.batch_list = []
        self.pfile_cache = None
        self.pfile_name = pfileinfo.filename
        self.bunchsize = bunchsize
        self.maxsentframe = maxsentframe
        self.maxnumsent = maxnumsent
        self.nmod_pad = nmod_pad
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.seq_info = pfileinfo.seq_info[start_sent:end_sent]
        self.seq_length = len(self.seq_info)
        self.cachesize = cachesize
        self.cache_start_index = 0
        self.cache_end_index = cachesize if cachesize <= self.seq_length else self.seq_length - 1
        self.fp = open(self.pfile_name, "rb")

        if self.pfilelabinfo is not None:
            self.pfilelab_cache = None
            self.pfilelab_name = pfilelabinfo.filename
            self.labseq_info = pfilelabinfo.seq_info[start_sent:end_sent]
            self.lab_fp = open(self.pfilelab_name, "rb")

        
    def __make_cache(self):
        self.pfile_cache = self.__cache_into_dict(self.pfileinfo, self.seq_info, self.cache_start_index, self.cache_end_index, self.fp)
        if self.pfilelabinfo is not None:
            self.labpfile_cache = self.__cache_into_dict(self.pfilelabinfo, self.labseq_info, self.cache_start_index, self.cache_end_index, self.lab_fp)
        

    def __cache_into_dict(self, pfileinfo, seq_info, cache_start_index, cache_end_index, fp):
        if pfileinfo.data_format[2] == 'f':
            dtype = np.dtype("float32")
        if pfileinfo.data_format[2] == 'i':
            dtype = np.dtype("int32")
        size = seq_info[cache_end_index][1] - seq_info[cache_start_index][1]
        start = seq_info[cache_start_index][1]
        fp.seek(pfileinfo.header_size + len(pfileinfo.data_format)*dtype.itemsize*start)
        raw_binary = fp.read(len(pfileinfo.data_format)*dtype.itemsize*size)
        binary_start = 0
        binary_end = 0
        pfilecache = {}
        for i in range(cache_start_index, cache_end_index):
            repeat_format = seq_info[i][2] * pfileinfo.data_format
            binary_start = binary_end
            binary_end = binary_end + len(pfileinfo.data_format) * dtype.itemsize * seq_info[i][2]
            value = np.frombuffer(raw_binary[binary_start: binary_end], dtype=dtype)
            value = value.byteswap()
            value = np.reshape(value, [-1, pfileinfo.frame_length])
            value = value[:, pfileinfo.real_data_start:]
            pfilecache[seq_info[i][0]] = value
        return pfilecache


    def __shuffle_and_batch(self):
        self.__make_cache()
        self.cache_start_index = self.cache_end_index if self.cache_end_index < self.seq_length - 1 else 0
        self.cache_end_index = min(self.cache_start_index + self.cachesize, self.seq_length - 1)
        seqlen_list = []
        for key in self.pfile_cache:
            seqlen_list.append([key, self.pfile_cache[key].shape[0]])
        if self.maxnumsent > 1:
            seqlen_list = sorted(seqlen_list, key=lambda e: e[1])

        current_batch = []
        current_batch_sent = 0
        current_maxsentframe = 0
        for seqid, seqlen in seqlen_list:
            seqlen = seqlen if seqlen % self.nmod_pad == 0 else seqlen + self.nmod_pad - seqlen % self.nmod_pad
            if self.maxsentframe is not None:
                if seqlen > self.maxsentframe:
                    continue
            if seqlen > self.bunchsize:
                continue
            if seqlen > current_maxsentframe:
                total_frame = seqlen * (1 + current_batch_sent)
                current_maxsentframe = seqlen
            else:
                total_frame = current_maxsentframe * (1 + current_batch_sent)
            if total_frame <= self.bunchsize:
                current_batch_sent = current_batch_sent + 1
                current_batch.append(seqid)
            else:
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch.append(seqid)
                current_batch_sent = 1
            if self.maxnumsent <= current_batch_sent:
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch_sent = 0
                current_maxsentframe = 0
        if self.shuffle:
            random.seed(self.random_seed)
            random.shuffle(self.batch_list)

    def getbatch(self):
        if not self.batch_list:
            self.__shuffle_and_batch()
        batch = self.batch_list.pop(0)
        batch_array = []
        lab_batch_array = []
        for seqid in batch:
            batch_array.append(self.pfile_cache[seqid])
            if self.pfilelabinfo is not None:
                lab_batch_array.append(self.labpfile_cache[seqid])
        if self.pfilelabinfo is not None:
            return batch_array, lab_batch_array
        else:
            return batch_array
        
    def __del__(self):
        self.fp.close()
        if self.pfilelabinfo is not None:
            self.lab_fp.close()


def PfileDataLoaderSingle(file_fea, file_lab, file_norm, batch_num, bunchsize, maxsentframe, maxnumsent, start_sent, end_sent, ndivide=1, divide_index=0, nmod_pad=64, 
                    shuffle_batch=True, num_workers=1, cachesize=10000, random_seed=0):
    fea_PfileInfo = PfileInfo(file_fea)
    lab_PfileInfo = PfileInfo(file_lab) if file_lab is not None else None
    if file_lab is not None:
        assert fea_PfileInfo.num_sentences == lab_PfileInfo.num_sentences, "number of sentences in feature and label are not equal"
    num_sentences = fea_PfileInfo.num_sentences
    assert start_sent < num_sentences, "start_sent must smaller than total sentences {0}".format(num_sentences)
    end_sent = min(num_sentences, end_sent)

    num_sentences = end_sent - start_sent
    read_startindex = int(num_sentences / ndivide * divide_index) + start_sent
    read_endindex = read_startindex + int(num_sentences / ndivide) if divide_index < ndivide - 1 else end_sent
    pfile_dataset = PfileDatasetSingle(file_fea, file_lab, file_norm, read_startindex, read_endindex, batch_num, bunchsize, maxsentframe, 
                                 maxnumsent, nmod_pad, shuffle_batch, cachesize, random_seed)

    return DataLoader(pfile_dataset, batch_size=1, num_workers=num_workers, multiprocessing_context="spawn", collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    #return DataLoader(pfile_dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_fn)

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



class PfileDatasetSingle(Dataset):
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
    
    def __norm_data(self, sequence, label, mean, var):
        data_norm=[]
        label_correct=[]
        label_new = []
        data_new = []
        # print('len of sequence: ', len(sequence))
        # print('len of label: ', len(label))
        for nparray in sequence:
            nparray = (nparray - mean[0,0,:,0].detach().clone().numpy())*var[0,0,:,0].detach().clone().numpy()
            data_norm.append(nparray)
            
        if label is not None:
            for nparray in label:
                # print("org: ",nparray)
                tmp_set = []
                tmp_count = []
                nparray = np.squeeze(nparray)
                start = 0
                # print("squeeze: ", type(nparray));exit()
                if(nparray.ndim == 0):  ## err: iteration over a 0-d array
                    nparray = nparray.reshape(1)
                for key, value in groupby(list(nparray)):
                    # print("key:",key)
                    # print("value: ",list(value))
                    tmp_set.append(key)
                    value_len = len(list(value))
                    tmp_count.append(value_len)

                if len(tmp_count) > 3:
                    if tmp_count[-2] + tmp_count[-3] > 7:  ########### final word frame<=8 low frame
                        initial_start = nparray.shape[0] - tmp_count[-1] - tmp_count[-2] - tmp_count[-3] ###tmp_count[-1] is 15002
                        initial_end   = initial_start + 3
                        vowel_start   = initial_end
                        vowel_end     = vowel_start + 4
                        nparray[initial_start:initial_end]  = tmp_set[-3]
                        nparray[vowel_start:vowel_end]      = tmp_set[-2]
                        nparray[vowel_end:nparray.shape[0]] = tmp_set[-1]

                    if tmp_count[1] + tmp_count[2] > 7:  ########### first word frame<=8 low frame
                        vowel_end = tmp_count[0] + tmp_count[1] + tmp_count[2]
                        vowel_start = vowel_end - 4
                        initial_end = vowel_start
                        initial_start = initial_end - 3
                        nparray[0:initial_start] = tmp_set[0]
                        nparray[initial_start:initial_end] = tmp_set[1]
                        nparray[vowel_start:vowel_end] = tmp_set[2]
                                                                            
                label_correct.append(nparray.reshape(-1,1))
            
            for nparray, nparray_data in zip(label_correct, data_norm):            
            # for nparray, nparray_data in zip(label, data_norm):
                tmp_count = []
                nparray = np.squeeze(nparray)
                start = 0
                if(nparray.ndim == 0):  ## err: iteration over a 0-d array
                    nparray = nparray.reshape(1)
                for key, value in groupby(list(nparray)):
                    value_len = len(list(value))
                    tmp_count.append(value_len)
                remain = random.randint(3, min(tmp_count[0], 10)) if tmp_count[0] > 3 else tmp_count[0]
                start = tmp_count[0] - remain
                # print('tmp_count',tmp_count[0])
                # print('start',start)
                label_new.append(nparray[start:].reshape(-1,1))
                data_new.append(nparray_data[start:,:])

                # print('1--------')
                # print(tmp_count[0], start, remain)
                # print(np.shape(nparray_data))
                # print(np.shape(data_new[-1]))
                # print(np.shape(label_new[-1]))

        # return data_norm, label_correct
        return data_new, label_new

    def __getitem__(self, index):
        label_lan = []
        if self.lab_PfileInfo is None:
            data = self.pfile_chunk_reader.getbatch()
            data, data_mask = self.__pad_nmod(data, self.nmod_pad, 0)
            meta = {}
            meta["mask"] = data_mask
            return data, meta
        else:
            data, label = self.pfile_chunk_reader.getbatch()
            label_ctc = []
            if label[0].shape[1] == 2:
                for idx, element in enumerate(label):
                    e1, e2 = np.split(element, 2, axis=1)
                    label_ctc.append(e2)
                    label[idx] = e1 # xiaobao
            elif label[0].shape[1] == 3:
                for idx, element in enumerate(label):
                    e1, e2, e3 = np.split(element, 3, axis=1)
                    label_ctc.append(e2)
                    label[idx] = e1 # xiaobao
                    final_frame = 0
                    start_frame = 0
                    for idx in range(e2.shape[0]-1, 0 , -1):
                        if e2[idx] not in [15002]:
                            final_frame = idx
                            break
                    for idx in range(e2.shape[0]-1):
                        if e2[idx] not in [15002]:
                            start_frame = idx
                            break
                    e3[:start_frame] = -1
                    e3[final_frame:] = -1
                    # e2[:start_frame//4*3] = -1
                    # end_gap = (e2.shape[0]-final_frame)//4
                    # e2[final_frame+end_gap:] = -1
                    label_lan.append(e3)
            else:
                for idx, element in enumerate(label):
                    label_ctc.append(element)
                    label[idx] = element # xiaobao
            
            # data, label = self.__norm_data(data, label, self.mean, self.var)
            data, label_ctc = self.__norm_data(data, label_ctc, self.mean, self.var)
            data, data_mask = self.__pad_nmod(data, self.nmod_pad, 0)
            #data = data - self.mean
            #data = data * self.var
            label, _ = self.__pad_nmod(label, self.nmod_pad, -1)
            # label = torch.Tensor(np.insert(np.array(label), 0, 9998, axis=3)).int()
            # # print(label)
            # label[label > 1660] = label[label > 1660] - 6739
            # label = torch.Tensor(np.insert(np.array(label), 0, 1, axis=3)).int()  ##for spm unit

            label_mask = label.clone()
            label_mask[label_mask >= 0] = 1
            label_mask[label_mask < 0] = 0

            ### rm t/4 >s sent
            ss=label_mask.sum(3)
            # print(data.shape, data_mask.shape)  #torch.Size([13, 1, 40, 148]) torch.Size([13, 148])
            bb=data.shape[0]-1

            # print('in: ',data.shape, label.shape, label_mask.shape)
            tmp_bb=bb
            while(bb>=0):
                tt=int(data[bb].shape[2]/4)
                if(tt<ss[bb]):
                    if(tmp_bb == 0):
                        print(f'warning: all sent tt<ss: only use first sent')
                    else:
                        tmp_bb-=1
                        data = data[torch.arange(data.size(0))!=bb] 
                        data_mask = data_mask[torch.arange(data_mask.size(0))!=bb] 
                        label = label[torch.arange(label.size(0))!=bb] 
                        label_mask = label_mask[torch.arange(label_mask.size(0))!=bb] 
                        print(f'warning: tt<ss[bb]: {tt}, {ss[bb]}, remove sent')
                bb-=1
            # print('out: ',data.shape, label.shape, label_mask.shape)

            # if(tmp_bb == data.shape[0]):
            #     data=data[0:1,:,:,:]
            #     data_mask = data_mask[0:1,:,:]
            #     label = label[0:1,:,:,:]
            #     label_mask = label_mask[0:1,:,:,:]
            #     print(f'warning: all sent s>100: only use first sent')

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
            # print(', self.nmod_pad: ', self.nmod_pad) #config: pad_num
            meta["inputs_length"] = get_length(meta["rnn_mask"], self.nmod_pad).contiguous()#, self.nmod_pad
            meta["targets_length"] = get_length(meta["att_mask"], self.nmod_pad).contiguous()            
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
    dataset.fea_PfileInfo = PfileInfo(dataset.file_fea)
    dataset.lab_PfileInfo = PfileInfo(dataset.file_lab) if dataset.file_lab is not None else None
    start_index = dataset.start_index
    end_index = dataset.end_index
    sentence_per_worker = int((end_index - start_index) / num_workers)
    worker_start = start_index + sentence_per_worker * worker_idx
    worker_end = min(worker_start + sentence_per_worker, end_index)
    dataset.pfile_chunk_reader = PfileChunkReader(dataset.fea_PfileInfo, dataset.lab_PfileInfo, worker_start, worker_end, 
                                                  dataset.bunchsize, dataset.maxsentframe, dataset.maxnumsent, 
                                                  dataset.nmod_pad, dataset.cachesize, dataset.shuffle_batch, dataset.random_seed)

def collate_fn(batch):
    data, meta = batch[0]
    return data, meta
