import torch
import asr
import torch.nn.functional as F
import torch.onnx
# from net import conformer
import configparser
from asr.utils import train_helper
from asr.data import Pfileinfo, PfileDataLoader, TestPfileDataLoader
from asr.optim import SGD
from asr.utils import clip_grad_norm

import sys
import torchintx
# print(torchintx.__version__);exit()
import numpy as np

#word_dict = "/work1/asrdictt/zhyou2/workspace/yewu/chezai_syllable_allnodes/cnSyllable_enSubword/bak/states_list.all.new.onlyen"

class VocabularyRec(object):
    def __init__(self,f_dic):
        self.idx2word = []
        self.word2idx = {}
        # self.idx2word.append("<BLANK>")
        with open(f_dic, "r", encoding="GBK") as f:
            for idx, line in enumerate(f):
                x = line.strip()
                self.idx2word.append(x)
                self.word2idx[x] = idx

    def list_to_sent(self, li):
        words = [self.idx2word[i] for i in li]
        tmp_word =[]
        for tmp in words:
            if tmp[0] == "#":
                continue
            if tmp[0]!="@" or len(tmp_word)==0:
                tmp_word.append(tmp)
            elif len(tmp)>2:
                tmp_word[-1]=tmp_word[-1]+tmp[2:]

        return tmp_word
        #return " ".join(tmp_word)

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

test_feature_pfile0 = "/raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_common_20260209_sp2.0k/lib_fb40/mix/fea.pfile0"
test_lab_pfile0 = "/raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_common_20260209_sp2.0k/lib_fb40/mix/lab.pfile0"

train_feature_pfile = "/raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_common_20260209_sp2.0k/lib_fb40/mix/fea.pfile1"
train_lab_pfile = "/raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_common_20260209_sp2.0k/lib_fb40/mix/lab.pfile1"

device = 0  #int(sys.argv[1])

count =0
celoss = 0

test_model_name = "tmp.pt"
print(test_model_name)

count0 = count

# config = configparser.ConfigParser()
# config.read("config.ini")

if __name__ == "__main__":
    with torch.cuda.device(device):
        total_sentnum = Pfileinfo(train_feature_pfile).num_sentences
        #bunchsize=12000
        bunchsize=8192
        maxsentframe=4096
        maxnumsent=4096
        padnum=4
        train_iternum = Pfileinfo.estimate_num_batch(Pfileinfo(
        train_feature_pfile).seq_info[0: total_sentnum], bunchsize, maxsentframe, maxnumsent, padnum)
        d = PfileDataLoader(file_fea=train_feature_pfile,
                            file_lab=train_lab_pfile,
                            file_norm=NormFile,
                            batch_num=train_iternum,
                            bunchsize=bunchsize,
                            maxsentframe=maxsentframe,
                            maxnumsent=maxnumsent,
                            start_sent=0,
                            end_sent=train_iternum,
                            ndivide=1,
                            divide_index=0,
                            nmod_pad=padnum,
                            shuffle_batch=True,
                            random_seed=0)
        print('train_iternum: ', train_iternum)

        total_sentnum_mdr = Pfileinfo(test_feature_pfile0).num_sentences
        total_sentnum_mdr = 1000

        
        # model = conformer().cuda()
        # model_ori = train_helper.get_module(config["Model"]["ModelName"])(); 
        # print(model_ori.cpu().state_dict());exit()
        model = train_helper.get_module("net_relu__addS_phCTC-add-hidden_skip_try2_para.Transducer")()
        # model = train_helper.get_module("net_relu__addS_analyse.Transducer")()
        # print(model_ori.cpu().state_dict());exit()
        
        # for idx, (data, meta) in enumerate(dd):
        #     # print(data.shape);exit()
        #     torchintx.trace_layers(model, model.encoder, (data,meta) )
        #     break

        # ### flops. params = 27.437G 9.243M
        # for idx, (data, meta) in enumerate(d):
        #     model(data, meta) #跑一遍前向
        #     from thop import profile ,clever_format
        #     flops,params = profile(model,inputs=(data, meta))
        #     macs,params = clever_format([flops,params],"%.3f")
        #     print(macs,params); exit()
    
        clamp_modules=(torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Conv1d,torch.nn.LSTM)
                
        aa = torch.ones((1,1,40,176))   #.cuda()
        #model.eval()
        torchintx.trace_layers(model, model.encoder, aa, fuse_bn=True )
        #
        torchintx.disable_clamp(model.encoder.conv1)
        torchintx.disable_clamp(model.joint.forward_layer)
        
        torchintx.clamp_module(model.encoder.conv2, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0) # 150 70 30 15 8
        torchintx.clamp_module(model.encoder.conv3, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0) # 100
        torchintx.clamp_module(model.encoder.conv4, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)#5 loss2  3 loss2 10
        
        # #
        torchintx.clamp_module(model.encoder.conv01, clamp_weight_value=2, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)##ok
        
        torchintx.clamp_module(model.encoder.conv02, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)#7 ok  10
        torchintx.clamp_module(model.encoder.conv03, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)#30 OK
        torchintx.clamp_module(model.encoder.conv04, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)#8  4  4  20
        
        torchintx.clamp_module(model.encoder.lstm, clamp_weight_value=2, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.lstm2, clamp_weight_value=2, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.lstm3, clamp_weight_value=2, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)
        
        torchintx.clamp_module(model.decoder.lstm, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.decoder.lstm2, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)
        
        torchintx.clamp_module(model.decoder.output_proj, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)
        
        torchintx.clamp_module(model.joint.project_layer, clamp_weight_value=2, clamp_bias_value=2, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
        
        torchintx.clamp_module(model.encoder.dnn_skip1, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.dnn_skip2, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.dnn_skip_out, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
        torchintx.clamp_module(model.phone_ce.ctc_dnn, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=8, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.phone_ce.ctc_out, clamp_weight_value=1, clamp_bias_value=1.5, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
        
        model = torchintx.clamp_layers(model, clamp_modules = clamp_modules, clamp_weight_value=8, clamp_bias_value=8, clamp_output_value=None,clamp_dynamic_percent=1.0)
        
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

        # torchintx.disable_quant(model.encoder.conv2)
        # torchintx.disable_quant(model.encoder.conv3)
        # torchintx.disable_quant(model.encoder.conv4)
        # torchintx.disable_quant(model.encoder.lstm)
        # torchintx.disable_quant(model.encoder.lstm2)
        # torchintx.disable_quant(model.encoder.lstm3)  #15->18
        # torchintx.disable_quant(model.encoder.conv01)
        # torchintx.disable_quant(model.encoder.conv02)
        # torchintx.disable_quant(model.encoder.conv03)
        # torchintx.disable_quant(model.encoder.conv04)  
        # torchintx.disable_quant(model.decoder.lstm)   #15->18
        # torchintx.disable_quant(model.decoder.lstm2)  #15->28
        # torchintx.disable_quant(model.decoder.output_proj)
        # torchintx.disable_quant(model.joint.project_layer) 
        ###goal 86  #ok   

        model = torchintx.init(model, quant_modules=quant_modules, data_bits=8, parameter_bits=8, out_bits=8, mode=torchintx.QuantMode.MaxValue)
        # import pdb; pdb.set_trace()
        
        # state = torch.load(test_model_name)["state_dict"]
        state = torch.load(test_model_name)
        
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
        
        # torch.save(model.cpu().state_dict(), "tmp.pt");exit()
        
        ###print analyse log        
        torchintx.wb_analyse(state);#exit()

        for idx, (data, meta) in enumerate(d):
        #     # print(data.shape);exit()
        #     torchintx.trace_layers(model, model.encoder, (data,meta) )
            # typical_input = torch.randn([1,1,40,248]).cuda()
            with torchintx.Dumper() as dumper:
                model.train()
                dumper.analyse_layer_output(model)   # match_pattern 可支持查看对应哪些层
                model(data, meta) #跑一遍前向
                dumper.save_out_analyse_log(save_log_path="Analyse_layer_output.log") #日志保存路径
            exit()
        
