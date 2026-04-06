import torch
import torch.onnx
import configparser
import torchintx
import numpy as np
from net_relu__addS_phCTCaddhidden_skip_try2_para_new import Transducer
from test_dataloader import Pfileinfo, TestPfileDataLoader
from asr.optim import SGD
import torch.nn.utils.prune as prune
#import asr
import sys

class VocabularyRec(object):
    def __init__(self,f_dic):
        self.idx2word = []
        self.word2idx = {}
        self.idx2word.append("<BLANK>")
        with open(f_dic, "r", encoding="GBK") as f:
            for idx, line in enumerate(f):
                x = line.strip()
                self.idx2word.append(x)
                self.word2idx[x] = idx + 1

    def list_to_sent(self, li):
        words = [self.idx2word[i] for i in li]
        return " ".join(words)

class VocabularyLab(object):
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

clamp_rate = 0.75

class L1Unstructured(prune.BasePruningMethod):

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
     
        prune._validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):

        tensor_size = t[1:14840,:].nelement()
  
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
 
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  
 
            topk = torch.topk(torch.abs(t[1:14840,:]).view(-1), k=nparams_toprune, largest=False)
      
            mask.view(-1)[topk.indices+256] = 0
            # import pdb;pdb.set_trace()

        return mask

    @classmethod
    def apply(cls, module, name, amount):
        return super(L1Unstructured, cls).apply(
            module, name, amount=amount
        )

def l1_unstructured(module, name, amount):
    L1Unstructured.apply(
        module, name, amount=amount
    )
    return module
    

if __name__ == "__main__":

    device   = 0
    test_model_name = sys.argv[1]

    # config = configparser.ConfigParser()
    # config.read("config.ini")

    with torch.cuda.device(device):
        
        model = Transducer()
        
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
        
        torchintx.SetResizeBilinearQuantMode(True)
        torchintx.SetPlatFormQuant(platform_quant=torchintx.PlatFormQuant.normal_quant)
        torchintx.SetIQTensorTanh(True)
        torchintx.SetIQTensorCat(False)

        quant_modules = (torch.nn.Conv2d,torch.nn.BatchNorm2d, torch.nn.Linear,torch.nn.ConvTranspose2d,torch.nn.Conv1d,torch.nn.LSTM)
        torchintx.disable_quant(model.encoder.conv1)
        torchintx.disable_quant(model.joint.forward_layer)
        model = torchintx.init(model, quant_modules = quant_modules,mode=torchintx.QuantMode.MaxValue, data_bits=8, parameter_bits=8, out_bits=8)

        state1 = torch.load(test_model_name)    #['state_dict']
        # for k in state1.keys():
        #     print(k, state1[k].shape)
        # print(state1.keys());exit()
        emb_dim = None
        new_state = {}
        for k, v in state1.items():
            if "scale_" in k:
                print('111: ', k, v.shape)
            if "decoder.embedding.embedding" in k:
                if "weight" in k:
                    emb_dim = v.shape[0]; #print(emb_dim);exit()
                    new_k = k.replace("decoder.embedding.embedding", "decoder.embedding")
                    print(k)
                    new_state[new_k] = v
            # elif 'ctc_dnn' in k or 'ctc_out' in k:    ##if no ctc
            #     continue
            # elif 'ctc_dnn' in k:
            #     new_k = k.replace("phone_ce.ctc_dnn", "phone_ctc.ctc_dnn")
            #     new_state[new_k] = v
            # elif 'ctc_out' in k:
            #     new_k = k.replace("phone_ce.ctc_out", "phone_ctc.ctc_out")
            #     if "weight" in k or "bias" in k:
            #         v_new = torch.cat((v[53:54, ...], v[1:53, ...], v[0:1, ...], v[54:55, ...], v[0:1, ...], v[0:1, ...]), dim=0)
            #         # print(v.shape, v_new.shape)
            #         new_state[new_k] = v_new
            #     else:                    
            #         new_state[new_k] = v
            #elif "phone_ce" in k:
            #    new_k = k.replace("phone_ce", "phone_ctc")
            #    print(k)
            #    new_state[new_k] = v
            ## elif "joint.project_layer" in k:
            ##     if "weight" in k:
            ##         print(k)
            ##         new_state[k] = v[:emb_dim, ...]
            ##     elif "bias" in k:
            ##         print(k)
            ##         new_state[k] = v[:emb_dim, ...]
            ##     else:
            ##         new_state[k] = v
            else:
                new_state[k] = v
            if "encoder.dnn_skip_out.conv.weight" in k or "encoder.dnn_skip_out.conv.bias" in k :
                new_state[k] = torch.cat((v[0:1], v[-1:]), dim=0)

        # print(dir(model), model.phone_ctc.ctc_out.scale_x,new_state["phone_ctc.ctc_out.scale_x"])
        # print(model.phone_ctc.ctc_out.scale_w,new_state["phone_ctc.ctc_out.scale_w"])
        # print(model.phone_ctc.ctc_out.scale_o,new_state["phone_ctc.ctc_out.scale_o"])
        #print("-----------000-------------")
        model.load_state_dict(new_state)
        #print("-----------1111-------------")
        # module = model.joint.project_layer
        # l1_unstructured(module, name="weight", amount=clamp_rate)
        # prune.remove(module, name="weight")
        # # import pdb;pdb.set_trace()
        # non_dict = []
        # for i in range(14844):
        #     if torch.logical_not(module.weight[i] == 0.0).sum() > 0.0:
        #         continue
        #     else:
        #         non_dict.append(i)
        
        # np.save("cn_label_data",voc_rec.list_to_sent([i for i in non_dict if i < 6735 ]))
        # np.save("cn_label_ind",np.array([i for i in non_dict if i < 6735 ]))
        # # import pdb;pdb.set_trace()
        NormFile = "/raw15/asrdictt/permanent/hjwang11/traindata/mlg20260210/french/car_fr_common_20260209_sp2.0k/lib_fb40/fea.norm0"
        #NormFile = "/raw15/asrmlg/permanent/ddye2/russian/russian_xdtxzx1993h_xdcz851h_GBgbzczfz672h_hffz2988h_fl_196h_fbnocmn40_rand_20240829_phone/fea.norm0"

        # ValidationDatadir0 = "/yrfs4/asrdictt/zhyou2/testset_mlg/english/tongyong"
        # test_feature_pfile = ValidationDatadir0+"/tongyong.fea.pfile.0"
        # test_lab_pfile = ValidationDatadir0+"/tongyong.lab.pfile.0"
        #ValidationDatadir0 = "/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/turkish/testset/qirui"
        #test_keyword = "QiruiT28_tuerqi_huifang_s11_QrT28trqsbcsj_20240426"
        #test_feature_pfile = ValidationDatadir0+"/"+test_keyword+".fea.pfile.0"
        #test_lab_pfile = ValidationDatadir0+"/"+test_keyword+".lab.pfile.0"
        #test_scp = ValidationDatadir0+"/"+test_keyword+".lab.scp"
        #不要用mixlab，
        ValidationDatadir0 = "/raw15/asrdictt/permanent/hjwang11/multilingual/french/rnnt/testset/NO1432_french_changan/pfile/lib_fb40/mix"
        test_feature_pfile = ValidationDatadir0+"/"+"fea.pfile0"
        test_lab_pfile = ValidationDatadir0+"/"+"lab.pfile0"

        total_sentnum  = Pfileinfo(test_feature_pfile).num_sentences
        print("total_sentnum",total_sentnum)
        total_sentnum = 600

        d = TestPfileDataLoader(file_fea=test_feature_pfile,
                            file_lab=test_lab_pfile,
                            file_norm=NormFile,
                            batch_num=total_sentnum,
                            # bunchsize=2048,
                            # maxsentframe=1200,
                            # maxnumsent=1,
                            start_sent=0,
                            end_sent=total_sentnum,
                            # ndivide=1,
                            # divide_index=0,
                            nmod_pad=4,
                            # shuffle_batch=False,
                            )

        model = model.cpu()

        model = model.eval()
        print("-----------4-------------")
        count = 0

        for idx, (data, meta) in enumerate(d):
            print(count)
            data = data.cpu()
            
            x=data
            (b, c, f, t) = x.size()
            print("x.shape",x.shape)
            if f != 40:
                x = x.permute(0,1,3,2).reshape(b,c,-1,40).permute(0,1,3,2)
                data=x

            meta = meta
            for key in meta:
                print(meta[key].shape)
                meta[key] = meta[key].cpu().long()
            print("-----------5-------------")
            with torch.no_grad():

                model = model.eval()
                model = model.cpu()

                enc_state,_ = model.encoder(data,meta)
                print("enc_state:",enc_state.shape)
                token = torch.LongTensor([[0]]).cpu()
                dec_state,_ = model.decoder(token)
                print("dec_state:",dec_state.shape)
                torch.onnx.export(model.encoder, (data), "model_encoder.onnx",export_params=True,opset_version=12,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
                torch.onnx.export(model.decoder, (token), "model_decoder.onnx",export_params=True,opset_version=12,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
                print(enc_state.shape, dec_state.shape)
                torch.onnx.export(model.joint,(enc_state,dec_state.unsqueeze(0)),"model_joint.onnx",export_params=True,opset_version=12,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
                torch.onnx.export(model.phone_ce,(enc_state),"model_ctc.onnx",export_params=True,opset_version=12,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

                exit()

                
