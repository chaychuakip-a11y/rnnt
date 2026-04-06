import torch
import asr
import torch.nn.functional as F
import torch.onnx
# from net import conformer
import configparser,sys
from asr.utils import train_helper
from asr.data import Pfileinfo, PfileDataLoader, TestPfileDataLoader
from asr.optim import SGD
from asr.utils import clip_grad_norm

import sys
import torchintx
# print(torchintx.__version__);exit()
import numpy as np

import sentencepiece as spm


phone_list = "/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/get_phone_ubctc/phones.list"

#spmodel="/yrfs4/asrdictt/hjwang11/multilingual/indonesian/rnnt/spm_2000/spm_indonesian_bpe_2000.model"
#sp = spm.SentencePieceProcessor(model_file=spmodel)

vocab="/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/spm_2000/states_list.french_onlyen.spm2.0k"
file_1 = open(vocab,"rb")
lines = file_1.readlines()
dict_vab = []
for line in lines:
    line = line.decode('utf-8').rstrip().split("	")[0]
    dict_vab.append(line)
    
def decode_vocab(dictlab):
    #print("dictlab:", dictlab)
    list0=[]
    for line in dictlab:
        #line = line.decode('utf-8').rstrip()
        sp_lab = int(line)
        str0 = dict_vab[sp_lab]
        if str0 != "<s>" and str0 != "</s>":
            list0.append(dict_vab[sp_lab])
    dst_str = "".join(list0)
    dst_str = dst_str.replace("▁"," ").strip()
    return dst_str
    
class VocabularyRec(object):
    def __init__(self,f_dic):
        self.idx2word = []
        self.word2idx = {}
        #self.idx2word.append("<BLANK>")
        with open(f_dic, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                x = line.strip()
                x = x.split('\t')[0]
                self.idx2word.append(x)
                self.word2idx[x] = idx
                # self.word2idx[x] = idx+1

    def list_to_sent(self, li):
        words = [self.idx2word[i] for i in li]
        return " ".join(words)

class VocabularyLab(object):
    def __init__(self,f_dic):
        self.idx2word = []
        self.word2idx = {}
        with open(f_dic, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                x = line.strip()
                x = x.split('\t')[0]
                self.idx2word.append(x)
                self.word2idx[x] = idx

    def list_to_sent(self, li):
        #print(len(self.idx2word), self.idx2word) ####注意asr/data/test_dataloader.py中起始符终止符对应id也需要修改
        words = [self.idx2word[i] for i in li]
        return " ".join(words)

def edit_dist(labs, recs):
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

def fix_bn(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()

NormFile = "/raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_common_20260209_sp2.0k/lib_fb40/fea.norm0"
 
#[raw]
test_feature_pfile0 = "/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/testset/NO1432_french_changan/pfile/lib_fb40/mix/fea.pfile0"
test_lab_pfile0 = "/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/testset/NO1432_french_changan/pfile/lib_fb40/mix/lab.pfile0"
 
#[fanhua]
train_feature_pfile = "/raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_tune_20260306_add_noise_sp2.0k/lib_fb40/mix/fea.pfile2"
train_lab_pfile = "/raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_tune_20260306_add_noise_sp2.0k/lib_fb40/mix/lab.pfile2"

device = 0  #int(sys.argv[1])
 
count =0
celoss = 0
 
#test_model_name = sys.argv[2]
savepath=sys.argv[1]
test_model_name =sys.argv[2]
print(sys.argv[2])
print(test_model_name)
# count = 200
count0 = count

# config = configparser.ConfigParser()
# config.read("config.ini")
'''
if __name__ == "__main__":
    p1 = [2, 8083, 8247, 8115, 8185, 8222, 8247, 8200, 8213, 8169, 8243, 8121, 6009, 5623, 8157, 1343, 8098, 1940, 8096, 8083, 8113, 8090, 8107, 8096, 8079, 2]
    label_1 = decode_vocab(p1)
    p2= [8186, 8223, 8248, 8201, 8214, 8170, 8244, 8122, 6010, 5624, 8158, 1344, 8099, 1922, 8084, 8114, 8091, 8108, 8097, 8080, 3]
    p3 = [x - 1 for x in p2]
    print("label_1",label_1)
    result_0 = decode_vocab(p3)
    print("result_0",result_0)
'''
                        
                        

if __name__ == "__main__":
    with torch.cuda.device(device):
        total_sentnum = Pfileinfo(train_feature_pfile).num_sentences
        bunchsize=8192
        maxsentframe=4096
        maxnumsent=4096
        padnum=4
        train_iternum = Pfileinfo.estimate_num_batch(Pfileinfo(
        train_feature_pfile).seq_info[0: total_sentnum], bunchsize, maxsentframe, maxnumsent, padnum)
        print(train_iternum)
        d = PfileDataLoader(file_fea=train_feature_pfile,
                            file_lab=train_lab_pfile,
                            file_norm=NormFile,
                            batch_num=train_iternum,
                            bunchsize=bunchsize,
                            maxsentframe=maxsentframe,
                            maxnumsent=maxnumsent,
                            start_sent=0,
                            end_sent=train_iternum,
                            #end_sent=50,
                            ndivide=1,
                            divide_index=0,
                            nmod_pad=padnum,
                            shuffle_batch=True,
                            random_seed=0)
        print('train_iternum: ', train_iternum)

        total_sentnum_mdr = Pfileinfo(test_feature_pfile0).num_sentences
        total_sentnum_mdr = 500
        #total_sentnum_mdr = 10
        #print('total_sentnum_mdr: ', total_sentnum_mdr)
        # dd = TestPfileDataLoader(file_fea=test_feature_pfile0,
        #                     file_lab=test_lab_pfile0,
        #                     file_norm=NormFile,
        #                     batch_num=total_sentnum_mdr,
        #                     # bunchsize=2048,
        #                     # maxsentframe=1200,
        #                     # maxnumsent=1,
        #                     start_sent=0,
        #                     end_sent=total_sentnum_mdr,
        #                     # ndivide=1,
        #                     # divide_index=0,
        #                     nmod_pad=1,
        #                     # shuffle_batch=False,
        #                     # random_seed=0
        #                     )
        # dd = PfileDataLoader(file_fea=test_feature_pfile0,
        #                     file_lab=test_lab_pfile0,
        #                     file_norm=NormFile,
        #                     batch_num=total_sentnum_mdr,
        #                     bunchsize=2048,
        #                     maxsentframe=1200,
        #                     maxnumsent=1,
        #                     start_sent=0,
        #                     end_sent=total_sentnum_mdr,
        #                     ndivide=1,
        #                     divide_index=0,
        #                     nmod_pad=1,
        #                     shuffle_batch=False,
        #                     random_seed=0)
                            
                            
        # model = conformer().cuda()
        # model_ori = train_helper.get_module(config["Model"]["ModelName"])(); 
        # print(model_ori.cpu().state_dict());exit()
        #model = train_helper.get_module("net_relu__addS_phCTC-add-hidden_skip_try2_para_fixstat_fixenc_fixphone_mom0.Transducer")()
        model = train_helper.get_module("net_relu__addS_phCTC-add-hidden_skip_try2_para.Transducer")()
        # print(model_ori.cpu().state_dict());exit()

        # for idx, (data, meta) in enumerate(dd):
        #     # print(data.shape);exit()
        #     torchintx.trace_layers(model, model.encoder, (data,meta) )
        #     break

        clamp_modules=(torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Conv1d,torch.nn.LSTM)

        aa = torch.ones((1,1,40,176))   #.cuda()
        model.eval()
        torchintx.trace_layers(model, model.encoder, aa, fuse_bn=True )

        torchintx.disable_clamp(model.encoder.conv1)
        torchintx.disable_clamp(model.joint.forward_layer)
        torchintx.clamp_module(model.encoder.conv2, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=4, clamp_dynamic_percent=1.0) # 150 70 30 15 8
        torchintx.clamp_module(model.encoder.conv3, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=5, clamp_dynamic_percent=1.0) # 100
        torchintx.clamp_module(model.encoder.conv4, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=1, clamp_dynamic_percent=1.0)#5 loss2  3 loss2 10
    
        # #
        torchintx.clamp_module(model.encoder.conv01, clamp_weight_value=2, clamp_bias_value=1.5, clamp_output_value=4, clamp_dynamic_percent=1.0)##ok
    
        torchintx.clamp_module(model.encoder.conv02, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=7, clamp_dynamic_percent=1.0)#7 ok  10
        torchintx.clamp_module(model.encoder.conv03, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=7, clamp_dynamic_percent=1.0)#30 OK
        torchintx.clamp_module(model.encoder.conv04, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=2, clamp_dynamic_percent=1.0)#8  4  4  20
        
        torchintx.clamp_module(model.encoder.lstm, clamp_weight_value=2, clamp_bias_value=1.5, clamp_output_value=1, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.lstm2, clamp_weight_value=2, clamp_bias_value=1.5, clamp_output_value=1, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.lstm3, clamp_weight_value=2, clamp_bias_value=1.5, clamp_output_value=1, clamp_dynamic_percent=1.0)
        
        torchintx.clamp_module(model.decoder.lstm, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=1, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.decoder.lstm2, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=1, clamp_dynamic_percent=1.0)
        
        torchintx.clamp_module(model.decoder.output_proj, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=1, clamp_dynamic_percent=1.0)
        
        torchintx.clamp_module(model.joint.project_layer, clamp_weight_value=2, clamp_bias_value=2, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
        
        torchintx.clamp_module(model.encoder.dnn_skip1, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=1, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.dnn_skip2, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=1, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.dnn_skip_out, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
        torchintx.clamp_module(model.phone_ce.ctc_dnn, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.phone_ce.ctc_out, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=None, clamp_dynamic_percent=1.0)
        
        model = torchintx.clamp_layers(model, clamp_modules = clamp_modules, clamp_weight_value=None, clamp_bias_value=None, clamp_output_value=None,clamp_dynamic_percent=1.0)
        #model = torchintx.clamp_layers(model, clamp_modules = clamp_modules, clamp_weight_value=8, clamp_bias_value=8, clamp_output_value=None,clamp_dynamic_percent=1.0)
        
        ### quant
        torchintx.SetResizeBilinearQuantMode(True)
        # replace_tuple=(torchintx.ClampConvBN2d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d, torchintx.ResizeBilinear, torch.nn.ConvTranspose2d, torch.nn.Conv1d, torch.nn.LSTM)
        # # replace_tuple=(torch.nn.Linear)
        torchintx.SetPlatFormQuant(platform_quant=torchintx.PlatFormQuant.normal_quant)
        # # torchintx.SetClipDataWeightLimits(enable=True, data_clip=8, weight_clip=8, bias_clip=8) #clamp_dynamic_percent参数目前只推荐调试定位异常层时使用，不推荐实际训练时使用
        
        # replace_tuple_nolstm=(torchintx.ClampConvBN2d, torch.nn.Linear, torchintx.ClampLinear, torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d) #, torch.nn.Conv1d,torch.nn.LSTM
        # torchintx.quant_module_by_type(model, type_modules=replace_tuple_nolstm, data_bits=8, parameter_bits=8, out_bits=8)    

        quant_modules = (torch.nn.Conv2d,torch.nn.BatchNorm2d, torch.nn.Linear,torch.nn.ConvTranspose2d,torch.nn.Conv1d,torch.nn.LSTM)
        torchintx.SetIQTensorTanh(True)
        torchintx.SetIQTensorCat(False)
        torchintx.disable_quant(model.encoder.conv1)
        torchintx.disable_quant(model.joint.forward_layer)

        # torchintx.quant_module(model.encoder.conv1, data_bits=8, parameter_bits=16)
        # torchintx.quant_module(model.encoder.conv2, data_bits=8, parameter_bits=8)
        # torchintx.quant_module(model.encoder.conv3, data_bits=8, parameter_bits=8)
        # torchintx.quant_module(model.encoder.conv4, data_bits=8, parameter_bits=8)

        # torchintx.disable_quant(model.encoder.conv2
        # torchintx.disable_quant(model.encoder.conv3)
        # torchintx.disable_quant(model.encoder.conv4)
        # torchintx.disable_quant(model.encoder.lstm) 
        # torchintx.disable_quant(model.encoder.lstm2)
        # torchintx.disable_quant(model.encoder.lstm3)
        # torchintx.disable_quant(model.encoder.conv01)
        # torchintx.disable_quant(model.encoder.conv02)
        # torchintx.disable_quant(model.encoder.conv03)
        # torchintx.disable_quant(model.encoder.conv04)
        # torchintx.disable_quant(model.decoder.lstm)
        # torchintx.disable_quant(model.decoder.lstm2)
        # torchintx.disable_quant(model.decoder.output_proj)
        # torchintx.disable_quant(model.joint.project_layer)
        # torchintx.disable_quant(model.encoder.dnn_skip1)
        # torchintx.disable_quant(model.encoder.dnn_skip2)
        # torchintx.disable_quant(model.encoder.dnn_skip_out)        
        # torchintx.disable_quant(model.phone_ctc.ctc_dnn)
        # torchintx.disable_quant(model.phone_ctc.ctc_out)
        ###goal 
        #关掉就是float
        model = torchintx.init(model, quant_modules=quant_modules, data_bits=8, parameter_bits=8, out_bits=8, mode=torchintx.QuantMode.MaxValue)
        # import pdb; pdb.set_trace()
        
        state = torch.load(test_model_name)["state_dict"]
        # state = torch.load(test_model_name)

        # # state['encoder.lstm2.weight_ih_l0'] = state.pop('encoder.lstm.weight_ih_l1')
        # # state['encoder.lstm2.bias_ih_l0'] = state.pop('encoder.lstm.bias_ih_l1')
        # # state['encoder.lstm2.weight_hh_l0'] = state.pop('encoder.lstm.weight_hh_l1')
        # # state['encoder.lstm2.bias_hh_l0'] = state.pop('encoder.lstm.bias_hh_l1')

        # # state['encoder.lstm3.weight_ih_l0'] = state.pop('encoder.lstm.weight_ih_l2')
        # # state['encoder.lstm3.bias_ih_l0'] = state.pop('encoder.lstm.bias_ih_l2')
        # # state['encoder.lstm3.weight_hh_l0'] = state.pop('encoder.lstm.weight_hh_l2')
        # # state['encoder.lstm3.bias_hh_l0'] = state.pop('encoder.lstm.bias_hh_l2')

        # # state['decoder.lstm2.weight_ih_l0'] = state.pop('decoder.lstm.weight_ih_l1')
        # # state['decoder.lstm2.bias_ih_l0'] = state.pop('decoder.lstm.bias_ih_l1')
        # # state['decoder.lstm2.weight_hh_l0'] = state.pop('decoder.lstm.weight_hh_l1')
        # # state['decoder.lstm2.bias_hh_l0'] = state.pop('decoder.lstm.bias_hh_l1')
        
        # state.pop('encoder.convout_can.weight')
        # state.pop('encoder.convout_can.bias')
        # state.pop('encoder.convout_mdr.weight')
        # state.pop('encoder.convout_mdr.bias')
        # # state.pop('encoder.convout_lid.weight')
        # # state.pop('encoder.convout_lid.bias')
        # # # print(state);exit()
        # # state['decoder.embedding.weight'] = state.pop('decoder.embedding.embedding.weight')
        # # state['decoder.embedding.embedding.weight'] = state.pop('decoder.embedding.weight')

        model.load_state_dict(state)
        
        # ###print analyse log        
        # torchintx.wb_analyse(state);#exit()

        # for idx, (data, meta) in enumerate(d):
        # #     # print(data.shape);exit()
        # #     torchintx.trace_layers(model, model.encoder, (data,meta) )
        #     # typical_input = torch.randn([1,1,40,248]).cuda()
        #     with torchintx.Dumper() as dumper:
        #         model.train()
        #         dumper.analyse_layer_output(model)   # match_pattern 可支持查看对应哪些层
        #         model(data, meta) #跑一遍前向
        #         dumper.save_out_analyse_log(save_log_path="Analyse_layer_output.log") #日志保存路径
        #     exit()
        

        model = model.cuda()
        # lr=0.00005
        # optimizer = SGD(model.parameters(),
        #                 lr=lr,
        #                 momentum=0.9,
        #                 weight_decay=0,
        #                 nesterov=True)
        lr = 3e-7
        optimizer = torch.optim.Adam(
                model.parameters(), lr=lr)
        count = 30
        for idx, (data, meta) in enumerate(d):
            data = data.cuda()
            # print(data.shape);exit()
            x=data
            (b, c, f, t) = x.size()
            if f != 40:
                x = x.permute(0,1,3,2).reshape(b,c,-1,40).permute(0,1,3,2)
                data=x

            for key in meta:
                if isinstance(meta[key], int) or meta[key]==None:
                    continue
                # print(meta[key].shape)
                meta[key] = meta[key].cuda()

            model.train()
            # # model.apply(fix_bn)
            optimizer.zero_grad()   
            # print(data.shape)

            # celoss = model(data, meta)
            loss_dict = model(data, meta)
            celoss = 0
            for key in loss_dict:
                try:
                    if len(loss_dict[key].shape) > 0:
                        celoss += loss_dict[key][0]
                    else:
                        celoss += loss_dict[key]
                    
                    print('%s: %.4f ' % (key, loss_dict[key].item()), end="")
                except:
                    import pdb;pdb.set_trace()
                    print(0)
                    
            print('total: ', count, '%.4f'%(celoss.item()))
            
            # if(celoss<10):
            #参数是否更新
            if(1):
                if(count>count0):
                    print(count0, ' done');count0+=1;continue
                # model = model.train()
                # celoss, loss_can, loss_mdr = model(data, meta)
                celoss.backward()
            #         # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                # if(count0 % 10 == 9):
                #     optimizer.step()
                #     optimizer.zero_grad()   
                #     print('b', count0, 'step')  
                optimizer.step()
                model = model.eval()
            # else:
            #     continue

            count = count + 1
            count0 = count0 + 1
            #float
            #if(count == savenum):
            #int
            #if((count == 10)):
            if((count % 10 == 0 and (count >= 20)) ): # or (count == 1)
                for p in model.parameters():
                    p.requires_grad=False
                # exit()
                if hasattr(model.encoder.lstm, 'running_h'):
                    model.encoder.lstm.running_h.data.copy_(model.encoder.lstm.running_o.data)
                if hasattr(model.encoder.lstm2, 'running_h'):
                    model.encoder.lstm2.running_h.data.copy_(model.encoder.lstm2.running_o.data)
                if hasattr(model.encoder.lstm3, 'running_h'):
                    model.encoder.lstm3.running_h.data.copy_(model.encoder.lstm3.running_o.data)
                if hasattr(model.decoder.lstm, 'running_h'):
                    model.decoder.lstm.running_h.data.copy_(model.decoder.lstm.running_o.data)
                if hasattr(model.decoder.lstm2, 'running_h'):
                    model.decoder.lstm2.running_h.data.copy_(model.decoder.lstm2.running_o.data)
            # if(count > 196):
            # if(count >= 200):
                # print('aaaaaa')
                # voc = VocabularyRec(word_dict)
                # voc_lab = VocabularyLab(word_dict_lab)
                voc_phone = VocabularyLab(phone_list)

            ####test 
                total_dist = 0
                total_word = 0
                total_sc = 0
                acc = 0
                total_skip_count_enc = 0
                total_skip_count_dec = 0
                total_t_length = 0
                # total_dist_b = 0
                # total_word_b = 0
                # acc_b = 0
                total_ph_dist = 0
                total_ph_word = 0
                phctc_acc = 0

                ff = open("trq_output/model_{}_{}.log".format(lr,count), 'w')
                # ff = open("model_{}_{}.log".format(lr,count), 'w')

                if 1:
                    dd = TestPfileDataLoader(file_fea=test_feature_pfile0,
                                        file_lab=test_lab_pfile0,
                                        file_norm=NormFile,
                                        batch_num=total_sentnum_mdr,
                                        # bunchsize=2048,
                                        # maxsentframe=1200,
                                        # maxnumsent=1,
                                        start_sent=0,
                                        end_sent=total_sentnum_mdr,
                                        # ndivide=1,
                                        # divide_index=0,
                                        nmod_pad=padnum,
                                        # shuffle_batch=False,
                                        # random_seed=0
                                        )
                    for idx, (data, meta) in enumerate(dd):
                        # print("count: ", count)
                        data = data.cuda()

                        x=data
                        (b, c, f, t) = x.size()
                        if f != 40:
                            x = x.permute(0,1,3,2).reshape(b,c,-1,40).permute(0,1,3,2)
                            data=x

                        # if(t != 54):
                        #     print(idx, data.shape);continue
                        # else:
                        #     print(idx, data.shape);exit()
                        # sio.savemat("data.mat",{"data":data.clone().squeeze().cpu().detach().numpy()});#exit()
                        # print(data);exit()

                        for key in meta:
                            # print(meta[key].shape)
                            meta[key] = meta[key].cuda()

                        # celoss = model(data, meta)
                        # print("loss: ", celoss, meta["att_label"].permute(1,0).contiguous())#;exit()

                        targets = meta["att_label"].permute(1,0).contiguous()
                        inputs_length = meta["inputs_length"]
                        
                        # targets_length = meta["targets_length"]
                        att_mask = meta["att_mask"].permute(1,0).contiguous()
                        targets_length = att_mask.sum(1)
                    
                        # targets = F.pad(targets, pad=(1, 0, 0, 0), value=2) #add start label zero
                        # targets_length = targets_length.add(1)
                        
                        model = model.eval()
                        model = model.cuda()

                        # enc_state = model.encoder(data, meta)
                        enc_state, logits_enc_binary = model.encoder(data, meta)
                        # print("enc_state:",enc_state.shape, enc_out_lid.shape)
                        # sio.savemat("enc.mat",{"enc_state":enc_state.clone().squeeze().cpu().detach().numpy()});exit()
                        
                        # token = torch.LongTensor([[0]]).cuda()
                        # dec_state,_ = model.decoder(token)
                        # print("dec_state:",dec_state.shape)
                        #sys.exit()

                        batch_size = x.size(0)
                        # print (batch_size)

                        zero_token = torch.LongTensor([[0]])
                        if x.is_cuda:
                            zero_token = zero_token.cuda()

                        def decode(enc_state, lengths, enc_blank_logit, threshold, threshold_dec):
                            token_list = []
                            time_list = []
                            dec_state, hidden = model.decoder(zero_token)

                            decs = []
                            decs.append(dec_state)
                            probs = []

                            count_enc = 0
                            count_dec = 0

                            for t in range(lengths):
                                if enc_blank_logit[t][0] >float(threshold_dec):
                                    count_enc+=1
                                    continue
                                logits = model.joint(enc_state[t].view(-1), dec_state.view(-1))
                                # logits = model.joint(enc_state[t].view(-1), dec_state.view(-1))
                                probs.append(logits)

                                out_b = F.softmax(torch.cat((logits[:1],logits[-1:])), dim=0).detach()        
                                if out_b[0] >float(threshold):
                                    count_dec+=1
                                    continue

                                out = F.softmax(logits[:-1], dim=0).detach()
                                pred = torch.argmax(out, dim=0)
                                pred = int(pred.item())

                                if pred != 0:
                                    time_list.append(t*4)
                                    token_list.append(pred)
                                    token = torch.LongTensor([[pred]])

                                    if enc_state.is_cuda:
                                        token = token.cuda()
                                    # print(token)
                                    dec_state, hidden = model.decoder(token, hidden=hidden)
                                    decs.append(dec_state)

                            return token_list, time_list, count_enc, count_dec


                        results = []

                        for i in range(batch_size):
                            # print(enc_state.shape, inputs_length[i])
                            logits_enc_binary = logits_enc_binary.reshape(-1,logits_enc_binary.shape[-1])
                            # (logits_enc_pred, logits_enc_idx) = torch.max(logits_enc_binary, dim=-1)
                            logits_enc_binary = logits_enc_binary.softmax(-1)
                            logits_enc_idx = (logits_enc_binary[:,0] < float(0.9))

                            decoded_seq, time_seq, skip_count_enc, skip_count_dec = decode(enc_state[i], inputs_length[i].int(), logits_enc_binary, 0.9, 0.9)

                            # decoded_seq, logits_tmp = decode(enc_state[i], inputs_length[i].int(),threshold=0.9, threshold_dec=0.9)

                            results.append(decoded_seq)

                        # ### stat blank
                        # # print(torch.stack(logits_tmp).shape);exit() #torch.Size([60, 10001])
                        # logits_tmp = torch.stack(logits_tmp)
                        # logits_post_lab = logits_tmp[:,:-1]
                        
                        # (logits_pred_lab, logits_idx_lab) = torch.max(logits_post_lab, dim=-1)
                        # logits_idx_lab[logits_idx_lab>0]=1
                        # # print(logit_idx.shape, logit_idx[0])#;exit()
                        # logits_final = torch.cat((logits_tmp[:,0:1], logits_tmp[:,-1:]), dim=-1)
                        # # logits_final = logits_final.reshape(logits_final.shape[0],-1,logits_final.shape[-1])
                        # logits_final = logits_final.reshape(-1,logits_final.shape[-1])
                        # (logits_pred, logits_idx) = torch.max(logits_final, dim=-1)

                        # # print(logits_idx_lab, logits_idx)

                        # correct = logits_idx.eq(logits_idx_lab).float()
                        # # print(language_lab.squeeze(1), predict);import time;time.sleep(1)
                        # if not hasattr(correct, 'sum'):
                        #     correct = correct.cpu()
                        
                        # total_dist_b += correct.sum().item()
                        # total_word_b += len(logits_idx_lab)
                        # acc_b = total_dist_b/(total_word_b)*100

                        #targets = targets-1  ###标注id错位
                        
                        ###stat acc
                        transcripts = [targets.cpu().numpy()[i][:targets_length[i].int().item()]
                                    for i in range(targets.size(0))]
                        # ## print(targets.shape, ' || ', targets_length, ' || ', targets, ' || ',transcripts)
                        # print(transcripts[0], results, [x - 1 for x in results[0][1:]])
                        # print(type(transcripts[0]), type(results), type([x - 1 for x in results[0][1:]]))
                                    
                        # label_1 = voc_lab.list_to_sent(transcripts[0])
                        
                        label_1 = decode_vocab(transcripts[0].tolist())
                        result_0 = decode_vocab([x for x in results[0][1:]])
                        #label_1 = decode_vocab(transcripts[0].tolist())
                        #print("label_1:",label_1)
                        #result_0 = decode_vocab([x - 1 for x in results[0][1:]])
                        #print("result_0:", result_0)
                        #label_1_id = transcripts[0].tolist()
                        #result_0_id = [x - 1 for x in results[0][1:]]
                        #print("label_1:", label_1+'\n')
                        #print("result_0:", result_0+'\n')
                        #print("label_1_id:", label_1_id)
                        #print("result_0_id:", result_0_id)
                        
                        #print("-----------------------------------------------------------\n")
                        #print("transcripts[0]",transcripts[0])
                        #print("transcripts[0].tolist()",transcripts[0].tolist())
                        #print("label_1",label_1)
                        #print("results[0]",results[0])
                        #print("results[0][1:]",results[0][1:])
                        #
                        #print("result_0",result_0)
                        # print("lab:",voc.list_to_sent(transcripts[0]),(transcripts[0]))
                        # print("rec:",voc.list_to_sent(results[0]),(results[0]));exit()
                        # print("lab:",(label_1))
                        # print("rec:",(result_0))
                        # dist, num_words = computer_cer(results, transcripts)
                        # exit()
                        # import time; time.sleep(3)
                        label_1_tmp = label_1.split(" ")
                        result_0_tmp = result_0.split(" ")
                        dist = edit_dist(label_1_tmp, result_0_tmp)
                        num_words = len(label_1_tmp)

                        #dist = edit_dist(label_1, result_0)
                        #num_words = len(label_1) - 1

                        total_dist += dist
                        total_word += num_words

                        cer = total_dist / total_word * 100
                        acc = 100 - cer
                        # count += 1
                        # if(dist == 0): total_sc += 1 
                        total_skip_count_enc += skip_count_enc
                        total_skip_count_dec += skip_count_dec
                        total_t_length += inputs_length[0].item()
                        skip_rate_enc = total_skip_count_enc/total_t_length*100
                        skip_rate_dec = total_skip_count_dec/total_t_length*100

                        if "label_ctc_ph" in meta:
                            phone_ctc_logit = model.phone_ce(enc_state) # BTD 
                            # phone_ctc_logit = model.ctc_out(model.relu(model.ctc_dnn(enc_state))) # BTD 

                            label_ctc = meta["label_ctc_ph"].permute(1,0).contiguous()
                            label_ctc_mask = meta["label_ctc_ph_mask"].permute(1,0).contiguous()
                            label_ctc_length = label_ctc_mask.sum(1)
                            phctc_acc, ph_dist, ph_words = model.acc_ctc(phone_ctc_logit.squeeze(), label_ctc)
                            total_ph_dist += ph_dist
                            total_ph_word += ph_words
                            phctc_acc = 100-(total_ph_dist/total_ph_word)*100

                            phone_rec = phone_ctc_logit.argmax(dim=-1).squeeze()
                            preds_new = []
                            for pred in phone_rec:
                                if pred == 0:
                                    continue
                                if pred != id:
                                    id = pred
                                    preds_new.append(pred)
                                else:
                                    continue

                            # print(label_ctc, preds_new)
                            # print(voc_phone.list_to_sent(label_ctc[0]))
                            # print(voc_phone.list_to_sent(preds_new))
                            label_ctc_show = voc_phone.list_to_sent(label_ctc[0][label_ctc[0]!=-1])
                            voc_phone_show = voc_phone.list_to_sent(preds_new)

                            ff.write("Sent#%d\n"%idx)
                            ff.write("ctc_label: "+label_ctc_show+"\n")
                            ff.write("ctc_pred : "+voc_phone_show+"\n\n")

                            print('\r test : updating the line of %d acc=%2f, phacc=%2f, enc-skip:%.2f, dec-skip:%.2f, lab:%s, rec:%s' % (idx,acc, phctc_acc, skip_rate_enc, skip_rate_dec, label_1, result_0), end="")
                        else:
                            print('\r test : updating the line of %d acc=%2f, enc-skip:%.2f, dec-skip:%.2f, lab:%s, rec:%s' % (idx,acc, skip_rate_enc, skip_rate_dec, label_1, result_0), end="")

                    ###print result
                    print(count)
                    # print("blank: ", acc_b)
                    if "label_ctc_ph" in meta:
                        print("acc: %2f, phacc: %2f" % (acc, phctc_acc), "\n")
                    else:
                        print("acc: %2f" % (acc), "\n")
                    print("skip_rate_enc: ", skip_rate_enc, "\n")
                    print("skip_rate_dec: ", skip_rate_dec, "\n")

            ####test mdr end


                ####save pt
                ### model_name = "model_{}_{}_cv{:.2f}_bc{:.2f}".format(lr,count,acc,acc_b)+".pt"
                model_name = savepath+"/model_{}_{}_cv{:.2f}_ph{:.2f}-enc{:.2f}-dec{:.2f}".format(lr,count,acc,phctc_acc,skip_rate_enc,skip_rate_dec)+".pt"
                #model_name = sys.argv[1]+"/model_{}_{}_cv{:.2f}_ph{:.2f}-enc{:.2f}-dec{:.2f}".format(lr,count,acc,phctc_acc,skip_rate_enc,skip_rate_dec)+".pt"
                # model_name = "model_{}_{}_cv{:.2f}_ph{:.2f}-enc{:.2f}-dec{:.2f}".format(lr,count,acc,phctc_acc,skip_rate_enc,skip_rate_dec)+".pt"
                # print(model_name);exit()
                torch.save(model.cpu().state_dict(), model_name)   # ;exit()
                model = model.cuda()

                for p in model.parameters():
                    p.requires_grad=True

            #### export onnx
            if(count % 20000 == 0):
            #     exit()
                with torch.no_grad():
                    model = model.eval()
                    model = model.cuda()

                    enc_state, _ = model.encoder(data, meta)

                    # #test
                    print("enc_state:",enc_state.shape)
                    token = torch.LongTensor([[0]]).cuda()
                    dec_state,_ = model.decoder(token)
                    print("dec_state:",dec_state.shape)
                    # #sys.exit()

                    targets = meta["att_label"].permute(1,0).contiguous()
                    targets_length = meta["targets_length"]
                    concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0) #add start label zero
                    # print(concat_targets.shape, targets_length);exit()
                    # dec_state,_ = model.decoder(concat_targets, targets_length.add(1))

    
                    torch.onnx.export(model.encoder, (data), "./model_encoder.onnx",export_params=True,opset_version=12,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
                    print('encoder done')
                    torch.onnx.export(model.decoder, (token), "./model_decoder.onnx",export_params=True,opset_version=12,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
                    print('decoder done')
                    torch.onnx.export(model.joint,(enc_state,dec_state),"model_joint.onnx",export_params=True,opset_version=12,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
                    print('joint done')
                    exit()
