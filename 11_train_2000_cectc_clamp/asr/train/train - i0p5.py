# pcli2 2019 Nov.
# bmuf logic is from caffe
from ..data import Pfileinfo, PfileDataLoader, txtDataLoader, DebugPfileDataLoader
from ..utils import clip_grad_norm
from ..optim import Lookahead
from ..optim import SGD
from ..utils.message import *
from torch.backends import cudnn
from ..utils.train_helper import *
import configparser
import torch.jit as jit
import torch.distributed as dist
import random
import torch.optim as optim
import torch
import multiprocessing
import torchintx
import math

context = multiprocessing.get_context("spawn")

cudnn.benchmark = False
cudnn.enabled = True


class BMUF():
    def __init__(self, model, config, dist):
        momentums = []
        global_models = []
        for param in model.parameters():
            temp = torch.zeros_like(param, requires_grad=False)
            temp.copy_(param.data)
            global_models.append(temp)
            momentums.append(torch.zeros_like(param, requires_grad=False))

        self.momentums = momentums
        self.global_models = global_models
        self.bmuf_alpha = config["TrainSetting"].getfloat("BMUF_ALPHA")
        self.bmuf_bm = config["TrainSetting"].getfloat("BMUF_BM")
        self.bmuf_blr = config["TrainSetting"].getfloat("BMUF_BLR")
        self.dist = dist

    def update(self, model):
        self.__update_param(model)

    def __update_param(self, model):
        for v, momentums, global_models in zip(model.parameters(), self.momentums, self.global_models):
            size = float(self.dist.get_world_size())
            avg = v.detach().clone()
            self.dist.all_reduce(avg.data, op=dist.ReduceOp.SUM)
            avg.data /= size
            update = self.bmuf_bm * momentums + global_models
            grad = avg - update
            momentums.copy_(self.bmuf_blr * grad + self.bmuf_bm * momentums)
            global_models.copy_(global_models + momentums)
            update = self.bmuf_bm * momentums + global_models
            v.data.copy_(v.detach() - self.bmuf_alpha * (v.detach() - update))


def run_multi_gpu(config, LOG):
    dist.init_process_group("nccl", init_method=config["TrainSetting"]["InitMethod"],
                            world_size=config["TrainSetting"].getint("NGpu"),
                            rank=config["TrainSetting"].getint("GlobalRank"))
    if not torch.cuda.is_available():
        LOG.log("no gpu device is available")
        raise Exception("no gpu device is available")

    np.random.seed(config["TrainSetting"].getint("RandomSeed"))
    torch.manual_seed(config["TrainSetting"].getint("RandomSeed"))
    torch.cuda.manual_seed(config["TrainSetting"].getint("RandomSeed"))
    torch.set_printoptions(10)

    model = get_module(config["Model"]["ModelName"])()

    # # data = torch.load("rnnt_mdr_data.pt")
    # # meta = torch.load("rnnt_mdr_meta.pt")
    # # torchintx.trace_layers(model, model.encoder, (data,meta), fuse_bn=True)
    clamp_modules=(torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Conv1d,torch.nn.LSTM)

    # aa = torch.ones((1,1,40,176))   #.cuda()
    # model.eval()
    # torchintx.trace_layers(model, model.encoder, aa, fuse_bn=True )
    
    # ####step4,all weight bias first,next output one by one
    torchintx.clamp_module(model.encoder.conv2, clamp_weight_value=1, clamp_bias_value=8, clamp_output_value=None, clamp_dynamic_percent=1.0) # 150 70 30 15 8
    # torchintx.clamp_module(model.encoder.conv3, clamp_weight_value=1, clamp_bias_value=8, clamp_output_value=None, clamp_dynamic_percent=1.0) # 360 200 100 50 25
    # torchintx.clamp_module(model.encoder.conv4, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=None, clamp_dynamic_percent=1.0)#100 50 25 8
    # torchintx.clamp_module(model.encoder.conv01, clamp_weight_value=7, clamp_bias_value=8, clamp_output_value=None, clamp_dynamic_percent=1.0)##8
    # torchintx.clamp_module(model.encoder.conv02, clamp_weight_value=1, clamp_bias_value=8, clamp_output_value=None, clamp_dynamic_percent=1.0)#40 20 8
    # torchintx.clamp_module(model.encoder.conv03, clamp_weight_value=1, clamp_bias_value=8, clamp_output_value=None, clamp_dynamic_percent=1.0)#100 50 20 8
    # torchintx.clamp_module(model.encoder.conv04, clamp_weight_value=1, clamp_bias_value=3, clamp_output_value=None, clamp_dynamic_percent=1.0)#100 50 30 8  ###
    # ####step4
    ####step1
    torchintx.clamp_module(model.encoder.lstm, clamp_weight_value=3, clamp_bias_value=3, clamp_output_value=1, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.lstm2, clamp_weight_value=2, clamp_bias_value=3, clamp_output_value=1, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.lstm3, clamp_weight_value=2, clamp_bias_value=2, clamp_output_value=1, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.decoder.lstm, clamp_weight_value=1, clamp_bias_value=2, clamp_output_value=1, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.decoder.lstm2, clamp_weight_value=1, clamp_bias_value=2, clamp_output_value=1, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.decoder.output_proj, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=1, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.joint.project_layer, clamp_weight_value=2, clamp_bias_value=4, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
    ###step1
    ###step2
    torchintx.clamp_module(model.encoder.dnn_skip1, clamp_weight_value=1, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.dnn_skip2, clamp_weight_value=1, clamp_bias_value=3, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.dnn_skip_out, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
    ###step2
    # ###step3
    # torchintx.clamp_module(model.phone_ce.ctc_dnn, clamp_weight_value=2, clamp_bias_value=3, clamp_output_value=None, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.phone_ce.ctc_out, clamp_weight_value=2, clamp_bias_value=4, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
    # ###step3
    model = torchintx.clamp_layers(model, clamp_modules = clamp_modules, clamp_weight_value=None, clamp_bias_value=None, clamp_output_value=None,clamp_dynamic_percent=1.0)

    # ## quant
    # torchintx.SetIQTensorTanh(True)
    # torchintx.SetIQTensorCat(False)

    # quant_modules = (torch.nn.Conv2d,torch.nn.BatchNorm2d, torch.nn.Linear,torch.nn.ConvTranspose2d,torch.nn.Conv1d,torch.nn.LSTM)
    # torchintx.disable_quant(model.encoder.conv1)
    # torchintx.disable_quant(model.encoder.conv2)
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
    # # torchintx.disable_quant(model.decoder.lstm2)
    # torchintx.disable_quant(model.decoder.output_proj)
    
    # torchintx.disable_quant(model.joint.forward_layer)
    
    # model = torchintx.init(model, quant_modules = quant_modules,mode=torchintx.QuantMode.MaxValue, data_bits=8, parameter_bits=8, out_bits=8)
    # ##quant end


    finetune_lm=False
    try:
        finetune_lm=config['NNLMSetting'].getboolean('FinetuneLM')
    except Exception as e:
        finetune_lm=False

    # init
    init_param(model, config["TrainSetting"]["PreviousModel"], finetune_lm)

    # train
    with torch.cuda.device(config["TrainSetting"].getint("LocalRank")):
            
        if config['TrainSetting'].get('TrainType') == "LM":
            d=txtDataLoader(config["NNLMSetting"]["TrainTextfile"],
                config["NNLMSetting"]["DictFile"],
                config["NNLMSetting"].getint("BatchSize"),
                config["NNLMSetting"].getint("TrainStartSent"),
                config["NNLMSetting"].getint("TrainEndSent"),
                ndivide=config["TrainSetting"].getint("NGpu"),
                divide_index=config["TrainSetting"].getint("GlobalRank"),
                shuffle_batch=config["NNLMSetting"].getboolean("Shuffle"),
                num_workers=1)
        if config['TrainSetting'].get('TrainType') == "ASR":
            d = PfileDataLoader(file_fea=config["DataSetting"]["TrainFeaturePfile"],
                file_lab=config["DataSetting"]["TrainLabelPfile"],
                file_norm=config["DataSetting"]["NormFilePath"],
                batch_num=config["DataSetting"].getint(
                    "TrainIterNum"),
                bunchsize=config["DataSetting"].getint(
                    "Bunchsize"),
                maxsentframe=config["DataSetting"].getint(
                    "MaxSentFrame"),
                maxnumsent=config["DataSetting"].getint(
                    "MaxNumSent"),
                start_sent=config["DataSetting"].getint(
                    "TrainStartSent"),
                end_sent=config["DataSetting"].getint(
                    "TrainEndSent"),
                ndivide=config["TrainSetting"].getint("NGpu"),
                divide_index=config["TrainSetting"].getint(
                    "GlobalRank"),
                nmod_pad=config["DataSetting"].getint("PadNum"),
                shuffle_batch=config["DataSetting"].getboolean("ShuffleBatch"),
                random_seed=config["TrainSetting"].getint("RandomSeed"))
        model = model.cuda()
        if config["TrainSetting"].getboolean("JIT") == True:
            model = jit.script(model)
        # optimizer
        if config["TrainSetting"]["Optimizer"] == "SGD":
            optimizer = SGD(
                model.parameters(),
                lr=config["TrainSetting"].getfloat("CurrentLearningRate"),
                momentum=0.9,
                weight_decay=0,
                nesterov=True)
        if config["TrainSetting"]["Optimizer"] == "ADAM":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["TrainSetting"].getfloat("CurrentLearningRate")
            )
        if config["TrainSetting"].getboolean("LookAhead") == True:
            optimizer = Lookahead(optimizer, k=config["TrainSetting"].getint("LookAhead_K"),
                                  alpha=config["TrainSetting"].getfloat("LookAhead_Alpha"))

        # bmuf global states
        bmuf = BMUF(model, config, dist)

        # train
        bmuf_train(model, bmuf, optimizer, config, d, LOG)
        if (config["TrainSetting"].getint("GlobalRank") == 0):
            LOG.log("train accomplished")
            save_model(model.cpu(), config["TrainSetting"]["CurrentModel"])
    #        train_iter.reset()

    dist.barrier()


def run_single_gpu(config, LOG):
    if not torch.cuda.is_available():
        LOG.log("no gpu device is available")
        raise Exception("no gpu device is available")

    np.random.seed(config["TrainSetting"].getint("RandomSeed"))
    torch.manual_seed(config["TrainSetting"].getint("RandomSeed"))
    torch.cuda.manual_seed(config["TrainSetting"].getint("RandomSeed"))
    torch.set_printoptions(10)

    with torch.cuda.device(config["TrainSetting"].getint("LocalRank")):
        model = get_module(config["Model"]["ModelName"])()
            
        # train
        if config['TrainSetting'].get('TrainType') == "LM":
            d=txtDataLoader(config["NNLMSetting"]["TrainTextfile"],
                config["NNLMSetting"]["DictFile"],
                config["NNLMSetting"].getint("BatchSize"),
                config["NNLMSetting"].getint("TrainStartSent"),
                config["NNLMSetting"].getint("TrainEndSent"),
                ndivide=1,
                divide_index=0,
                shuffle_batch=config["NNLMSetting"].getboolean("Shuffle"),
                num_workers=1)
        if config['TrainSetting'].get('TrainType') == "ASR":
            d = DebugPfileDataLoader(file_fea=config["DataSetting"]["TrainFeaturePfile"],
                file_lab=config["DataSetting"]["TrainLabelPfile"],
                file_norm=config["DataSetting"]["NormFilePath"],
                batch_num=config["DataSetting"].getint(
                    "TrainIterNum"),
                bunchsize=config["DataSetting"].getint(
                    "InitBunchsize"),
                maxsentframe=config["DataSetting"].getint(
                    "MaxSentFrame"),
                maxnumsent=config["DataSetting"].getint(
                    "MaxNumSent"),
                start_sent=config["DataSetting"].getint(
                    "TrainStartSent"),
                end_sent=config["DataSetting"].getint(
                    "TrainEndSent"),
                ndivide=1,
                divide_index=0,
                nmod_pad=config["DataSetting"].getint("PadNum"),
                shuffle_batch=config["DataSetting"].getboolean("ShuffleBatch"),
                random_seed=config["TrainSetting"].getint("RandomSeed"))


        clamp_modules=(torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Conv1d,torch.nn.LSTM)

        aa = torch.ones((1,1,40,176))   #.cuda()
        model.eval()
        torchintx.trace_layers(model, model.encoder, aa, fuse_bn=True )
        #
        torchintx.clamp_module(model.encoder.conv2, clamp_weight_value=2, clamp_bias_value=8, clamp_output_value=3, clamp_dynamic_percent=1.0) # 150 70 30 15 8
        torchintx.clamp_module(model.encoder.conv3, clamp_weight_value=1, clamp_bias_value=7, clamp_output_value=4, clamp_dynamic_percent=1.0) # 360 200 100 50 25
        torchintx.clamp_module(model.encoder.conv4, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=1, clamp_dynamic_percent=1.0)#100 50 25 8
        #
        torchintx.clamp_module(model.encoder.conv01, clamp_weight_value=8, clamp_bias_value=8, clamp_output_value=6, clamp_dynamic_percent=1.0)##8
        torchintx.clamp_module(model.encoder.conv02, clamp_weight_value=1, clamp_bias_value=7, clamp_output_value=7, clamp_dynamic_percent=1.0)#40 20 8
        torchintx.clamp_module(model.encoder.conv03, clamp_weight_value=1, clamp_bias_value=8, clamp_output_value=8, clamp_dynamic_percent=1.0)#100 50 20 8
        torchintx.clamp_module(model.encoder.conv04, clamp_weight_value=1, clamp_bias_value=3, clamp_output_value=2, clamp_dynamic_percent=1.0)#100 50 30 8
        #
        torchintx.clamp_module(model.encoder.lstm, clamp_weight_value=3, clamp_bias_value=3, clamp_output_value=1, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.lstm2, clamp_weight_value=3, clamp_bias_value=3, clamp_output_value=1, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.lstm3, clamp_weight_value=3, clamp_bias_value=2, clamp_output_value=1, clamp_dynamic_percent=1.0)
        
        torchintx.clamp_module(model.decoder.lstm, clamp_weight_value=1, clamp_bias_value=2, clamp_output_value=1, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.decoder.lstm2, clamp_weight_value=1, clamp_bias_value=2, clamp_output_value=1, clamp_dynamic_percent=1.0)
        
        torchintx.clamp_module(model.decoder.output_proj, clamp_weight_value=2, clamp_bias_value=1, clamp_output_value=1, clamp_dynamic_percent=1.0)
        
        torchintx.clamp_module(model.joint.project_layer, clamp_weight_value=2, clamp_bias_value=2, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
        #
        torchintx.clamp_module(model.encoder.dnn_skip1, clamp_weight_value=1, clamp_bias_value=2, clamp_output_value=1, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.dnn_skip2, clamp_weight_value=1, clamp_bias_value=3, clamp_output_value=2, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.encoder.dnn_skip_out, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
        #
        torchintx.clamp_module(model.phone_ce.ctc_dnn, clamp_weight_value=2, clamp_bias_value=3, clamp_output_value=8, clamp_dynamic_percent=1.0)
        torchintx.clamp_module(model.phone_ce.ctc_out, clamp_weight_value=2, clamp_bias_value=3, clamp_output_value=None, clamp_dynamic_percent=1.0)
        
        model = torchintx.clamp_layers(model, clamp_modules = clamp_modules, clamp_weight_value=8, clamp_bias_value=8, clamp_output_value=None,clamp_dynamic_percent=1.0)
 
        finetune_lm=False
        try:
            finetune_lm=config['NNLMSetting'].getboolean('FinetuneLM')
        except Exception as e:
            finetune_lm=False

        model = model.cuda()
        if config["TrainSetting"].getboolean("JIT") == True:
            model = jit.script(model)
        if (os.path.split(config["TrainSetting"]["PreviousModel"])[1] != "None"):
            init_param(model, config["TrainSetting"]["PreviousModel"], finetune_lm)

        initEncoderModel = config["TrainSetting"]["initEncoderModel"]
        if initEncoderModel != "NULL":
            print("load initEncoderModel:",initEncoderModel)
            LOG.log("load initEncoderModel:{0}".format(initEncoderModel))
            state_dict = torch.load(initEncoderModel)["state_dict"]
            model_dict = model.state_dict()
            # state_dict.pop("decoder.embedding.embedding.weight")
            # state_dict.pop("joint.project_layer.weight")
            # state_dict.pop("joint.project_layer.bias")
            for k, v in state_dict.items():
                if k in model_dict.keys():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v
                    else:
                        print("{}-------shape not match-------{} {}".format(k, v.size(), model_dict[k].size()))
                        # if k == "decoder.embedding.embedding.weight":
                            # model_dict[k] = v[26:,:]
#                         if k== "joint.project_layer.weight":
#                             # w_blank_1 = torch.randn(1, v.size()[1])
#                             # w_blank_2 = torch.randn(1, v.size()[1])
#                             # w_blank_4 = torch.randn(1, v.size()[1])
#                             # from asr.functions import xavier
#                             # xavier(w_blank_1)                            
#                             # xavier(w_blank_2)            
#                             # xavier(w_blank_4)   
#                             w_blank_1 = v[-1:,:].clone()
#                             w_blank_2 = v[-1:,:].clone()
#                             w_blank_4 = v[-1:,:].clone()
#                             model_dict[k] = torch.cat((v[:-1,:],w_blank_1, w_blank_2, w_blank_4), dim=0)
#                         elif k=="joint.project_layer.bias":
#                             # b_blank_1 = torch.full((1,), 0.1)
#                             # b_blank_2 = torch.full((1,), 0.1)
#                             # b_blank_4 = torch.full((1,), 0.1)
#                             b_blank_1 = v[-1:].clone()
#                             b_blank_2 = v[-1:].clone()
#                             b_blank_4 = v[-1:].clone()
#                             model_dict[k] = torch.cat((v[:-1],b_blank_1, b_blank_2, b_blank_4), dim=0)
# # joint.project_layer.weight-------shape not match-------torch.Size([1602, 256]) torch.Size([1604, 256])
# # joint.project_layer.bias-------shape not match-------torch.Size([1602]) torch.Size([1604])

            model.load_state_dict(model_dict)


        if config["TrainSetting"]["Optimizer"] == "SGD":
            optimizer = SGD(
                model.parameters(),
                lr=config["TrainSetting"].getfloat("CurrentLearningRate"),
                momentum=0.9,
                weight_decay=0,
                nesterov=True)
        if config["TrainSetting"]["Optimizer"] == "ADAM":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["TrainSetting"].getfloat("CurrentLearningRate")
            )
        if config["TrainSetting"].getboolean("LookAhead") == True:
            optimizer = Lookahead(optimizer, k=config["TrainSetting"].getint("LookAhead_K"),
                                  alpha=config["TrainSetting"].getfloat("LookAhead_Alpha"))

        simple_train(model, optimizer, config, d, LOG)
        LOG.log("train accomplished")
        save_model(model.cpu(), config["TrainSetting"]["CurrentModel"])


def run_validation(config, model, LOG):
            
    if config['TrainSetting'].get('TrainType') == "LM":
        d=txtDataLoader(config["NNLMSetting"]["ValidationTextfile"],
            config["NNLMSetting"]["DictFile"],
            1,
            config["NNLMSetting"].getint("ValidationStartSent"),
            config["NNLMSetting"].getint("ValidationEndSent"),
            ndivide=1,
            divide_index=config["TrainSetting"].getint("GlobalRank"),
            shuffle_batch=False,
            num_workers=1)
    if config['TrainSetting'].get('TrainType') == "ASR":
        d = PfileDataLoader(file_fea=config["DataSetting"]["ValidationFeaturePfile"],
            file_lab=config["DataSetting"]["ValidationLabelPfile"],
            file_norm=config["DataSetting"]["NormFilePath"],
            batch_num=config["DataSetting"].getint(
                "ValidationIterNum"),
            bunchsize=config["DataSetting"].getint("Bunchsize"),
            maxsentframe=4096,
            maxnumsent=1,
            start_sent=config["DataSetting"].getint(
                "ValidationStartSent"),
            end_sent=config["DataSetting"].getint(
                "ValidationEndSent"),
            ndivide=1,
            divide_index=0,
            num_workers=1,
            nmod_pad=config["DataSetting"].getint("PadNum"),
            shuffle_batch=False)
    validation(model, config, d, LOG)

class Trainer():
    def __init__(self, train_type):
        self.train_type = train_type

    def lm_forward(self, index, data_element, model):
        x, mask, y= data_element
        x = x.cuda()
        mask=mask.cuda()
        y = y.cuda()
        celoss = model(x, mask, y)
        return celoss
    def asr_forward(self, index, data_element, model):
        data, meta = data_element
        data = data.cuda()
        for key in meta:
            meta[key] = meta[key].cuda()
        celoss = model(data, meta)
        return celoss
    def forward(self, index, data_element, model):
        if self.train_type == "ASR":
            return self.asr_forward(index, data_element, model)
        if self.train_type == "LM":
            return self.lm_forward(index, data_element, model)
    

def bmuf_train(model, bmuf, optimizer, config, data_loader, LOG):
    # if config["TrainSetting"].getint("GlobalRank") == 0:
    #     run_validation(config, model, LOG)
    #     model.cuda()
    model.train()
    # sum_of_last_displayed_loss = 0
    sum_of_last_displayed_loss = {}
    last_t = time.time()

    trainer_class = get_module(config['TrainSetting'].get("Trainer")) if config['TrainSetting'].get("Trainer") is not None else Trainer
    trainer = trainer_class(config['TrainSetting'].get('TrainType'))
            
    for i, data_element in enumerate(data_loader):
        optimizer.zero_grad()
        # loss = trainer.forward(i, data_element, model)
        loss_dict = trainer.forward(i, data_element, model)
        loss = 0
        for key in loss_dict:
            # if config["TrainSetting"].getboolean("FixED"):
            if len(loss_dict[key].shape) > 0:
                loss += loss_dict[key][0]
            else:
                loss += loss_dict[key]
        loss.backward()
	###***clip***###
        for group in optimizer.param_groups:
            for param in group["params"]:
                if(param.grad is not None):
                    if(torch.any(torch.isnan(param.grad.data))):
                        param.grad.data = torch.where(torch.isnan(param.grad.data), torch.full_like(param.grad.data, 0), param.grad.data)
                        print("warning: skip one sentence, gpu {0}, iteration {1}".format(config["TrainSetting"].getint("GlobalRank"), i))
                        LOG.log("warning: skip one sentence, gpu {0}, iteration {1}".format(config["TrainSetting"].getint("GlobalRank"), i))
        # sum_of_last_displayed_loss += loss.item()
        for key in loss_dict:
            if key not in sum_of_last_displayed_loss:
                sum_of_last_displayed_loss[key] = loss_dict[key].item()
            else:
                sum_of_last_displayed_loss[key] += loss_dict[key].item()
        clip_grad_norm(model.parameters(), config["TrainSetting"].getfloat("ClipGradient"),
                       config["TrainSetting"].getfloat("ClipGradient2"), config["TrainSetting"].getfloat("Discount"))
        optimizer.step()
        # if i % config["TrainSetting"].getint("Display") == 0 and i > 0:
        #     considered_iteration = config["TrainSetting"].getint("Display")
        #     displayed_loss = sum_of_last_displayed_loss / considered_iteration
        #     LOG.log(
        #         "gpu: {} loss: {} {} / {}, {:.2f} seconds/iteration".format(config["TrainSetting"].getint("GlobalRank"),
        #                                                                     displayed_loss, i,
        #                                                                     config["DataSetting"].getint(
        #                                                                         "TrainIterNum"),
        #                                                                     (time.time() - last_t) / config[
        #                                                                         "TrainSetting"].getint("Display")),
        #         rank=config["TrainSetting"].getint("GlobalRank"),
        #         logtype=LogType.SYNC, syncid=i, syncheader="iteration{0}".format(i))
        #     sum_of_last_displayed_loss = 0
        
        if i % config["TrainSetting"].getint("Display") == 0 and i > 0:
            considered_iteration = config["TrainSetting"].getint("Display")
            for key in sum_of_last_displayed_loss:
                sum_of_last_displayed_loss[key] = sum_of_last_displayed_loss[key] / considered_iteration
            log_str = "gpu: {} ".format(config["TrainSetting"].getint("GlobalRank"))
            for key in sum_of_last_displayed_loss:
                log_str+="{}: {:.4f} ".format(key,sum_of_last_displayed_loss[key])
            log_str += "{} / {}, {:.2f} seconds/iteration".format(i,config["DataSetting"].getint("TrainIterNum"),
                                                                 (time.time() - last_t) / config["TrainSetting"].getint("Display"))
            LOG.log(log_str,rank=config["TrainSetting"].getint("GlobalRank"),
                logtype=LogType.SYNC, syncid=i, syncheader="iteration{0}".format(i))
            sum_of_last_displayed_loss = {}
            last_t = time.time()

        if i % config["TrainSetting"].getint("BMUF_SYNC") == 0 and i > 0:
            bmuf.update(model)
        if config["TrainSetting"].getint("CVInterval") > 0 and i % config["TrainSetting"].getint("CVInterval") == 0 and \
                config["TrainSetting"].getint("GlobalRank") == 0 and i > 0:
            run_validation(config, model, LOG)
            model.train()
            save_model(model.cpu(), f'{config["TrainSetting"]["CurrentModel"]}.{i}')
            model.cuda()
    # if config["TrainSetting"].getint("CVInterval") == 0 and config["TrainSetting"].getint("GlobalRank") == 0:
    if config["TrainSetting"].getint("GlobalRank") == 0:
        run_validation(config, model, LOG)


def simple_train(model, optimizer, config, data_loader, LOG):
    model.train()
    # sum_of_last_displayed_loss = 0
    sum_of_last_displayed_loss = {}
    last_t = time.time()

    trainer_class = get_module(config['TrainSetting'].get("Trainer")) if config['TrainSetting'].get("Trainer") is not None else Trainer
    trainer = trainer_class(config['TrainSetting'].get('TrainType'))
    
            
    for i, data_element in enumerate(data_loader):
        optimizer.zero_grad()
        # loss = trainer.forward(i, data_element, model)
        loss_dict = trainer.forward(i, data_element, model)
        loss = 0
        for key in loss_dict:
            try:
                if len(loss_dict[key].shape) > 0:
                    loss += loss_dict[key][0]
                else:
                    loss += loss_dict[key]
            except Exception as e:
                import pdb;pdb.set_trace()
                print(0)
        loss.backward()
	###***clip***###
        for name, param in model.named_parameters():
           # print(name); print(param.data);sys.exit()
            if(param.grad is not None):
                if(torch.any(torch.isnan(param.grad.data))):
                    param.grad.data = torch.where(torch.isnan(param.grad.data), torch.full_like(param.grad.data, 0), param.grad.data)
                    print("warning: skip one sentence, gpu {0}, iteration {1}, {2}".format(config["TrainSetting"].getint("GlobalRank"), i, name))
                    LOG.log("warning: skip one sentence, gpu {0}, iteration {1}, {2}".format(config["TrainSetting"].getint("GlobalRank"), i, name))
                    # sys.exit()
        # sum_of_last_displayed_loss += loss.item()
        for key in loss_dict:
            if key not in sum_of_last_displayed_loss:
                sum_of_last_displayed_loss[key] = loss_dict[key].item()
            else:
                sum_of_last_displayed_loss[key] += loss_dict[key].item()
                
        clip_grad_norm(model.parameters(), config["TrainSetting"].getfloat("ClipGradient"),
                       config["TrainSetting"].getfloat("ClipGradient2"), config["TrainSetting"].getfloat("Discount"))
        optimizer.step()

        # if i % config["TrainSetting"].getint("Display") == 0 and i > 0:
        #     considered_iteration = config["TrainSetting"].getint("Display")
        #     displayed_loss = sum_of_last_displayed_loss / considered_iteration
        #     LOG.log(
        #         "gpu: {} loss: {} {} / {}, {:.2f} seconds/iteration".format(config["TrainSetting"].getint("GlobalRank"),
        #                                                                     displayed_loss, i,
        #                                                                     config["DataSetting"].getint(
        #                                                                         "TrainIterNum"),
        #                                                                     (time.time() - last_t) / config[
        #                                                                         "TrainSetting"].getint("Display")))
        #     sum_of_last_displayed_loss = 0
            
        if i % config["TrainSetting"].getint("Display") == 0 and i > 0:
            considered_iteration = config["TrainSetting"].getint("Display")
            for key in sum_of_last_displayed_loss:
                sum_of_last_displayed_loss[key] = sum_of_last_displayed_loss[key] / considered_iteration
            log_str = "gpu: {} ".format(config["TrainSetting"].getint("GlobalRank"))
            for key in sum_of_last_displayed_loss:
                log_str+="{}: {:.6f} ".format(key,sum_of_last_displayed_loss[key])
            log_str += "{} / {}, {:.2f} seconds/iteration".format(i,config["DataSetting"].getint("TrainIterNum"),
                                                                 (time.time() - last_t) / config["TrainSetting"].getint("Display"))
            LOG.log(log_str)

            sum_of_last_displayed_loss = {}
            last_t = time.time()
        if config["TrainSetting"].getint("CVInterval") > 0 and i % config["TrainSetting"].getint(
                "CVInterval") == 0 and i > 0:
            run_validation(config, model, LOG); model.train()
            print(f"save model {i}")
            save_model(model.cpu(), f'{config["TrainSetting"]["CurrentModel"]}.{i}')
            model.cuda()
    if config["TrainSetting"].getint("CVInterval") == 0:
        run_validation(config, model, LOG)


def validation(model, config, data_loader, LOG):
    model.eval()
    sum_of_last_displayed_acc = {}
    # criterion_holder = []

    trainer_class = get_module(config['TrainSetting'].get("Trainer")) if config['TrainSetting'].get("Trainer") is not None else Trainer
    trainer = trainer_class(config['TrainSetting'].get('TrainType'))
            
    for i, data_element in enumerate(data_loader):
        # criterion = trainer.forward(i, data_element, model)
        # criterion = criterion.item()
        acc_dict = trainer.forward(i, data_element, model)
        for key in acc_dict:
            if key not in sum_of_last_displayed_acc:
                sum_of_last_displayed_acc[key] = [acc_dict[key].item()]
            else:
                sum_of_last_displayed_acc[key].append(acc_dict[key].item())
                
        if i % config["TrainSetting"].getint("Display") == 0:
            LOG.log("validated {0} / {1}".format(i,
                                                 config["TrainSetting"].getint("CVSentNum")))
        # criterion_holder.append(criterion)
        
    try:
        model.total_dist = 0
        model.total_dist_skip = 0
        model.total_word = 0
        model.total_skip_count_enc = 0
        model.total_skip_count_dec = 0
        model.total_t_length = 0
    except Exception as e:
        print(e)

    for key in sum_of_last_displayed_acc:
        sum_of_last_displayed_acc[key] = np.array(sum_of_last_displayed_acc[key])
    # criterion_holder = np.array(criterion_holder)

    LOG.log("modelname.. {0}".format(config["TrainSetting"]["CurrentModel"]))
    # LOG.log("acc: {0}".format(np.mean(criterion_holder)))
    for key in sum_of_last_displayed_acc:
        LOG.log("{0}: {1}".format(key,np.mean(sum_of_last_displayed_acc[key])))


def single_gpu_warper(config, message_queue, LOG):
    status = printer(run_single_gpu)(config, LOG)
    message_queue.put(status)


def multi_gpu_warper(config, message_queue, LOG):
    status = printer(run_multi_gpu)(config, LOG)
    message_queue.put(status)


class train():
    def __init__(self):
        self.__done = False
        self.__LOG = None
        self.__message = None
        self.__config = None

    def start_train(self):
        if self.__config is None:
            self.__raise_err("missing config file")
        self.__check_config()
        self.__init_env()
        self.__loop()

    def load_config(self, configPath):
        config = configparser.ConfigParser()
        config.read(configPath)
        self.__config = config

    def __init_env(self):
        if "MASTER_ADDR" in os.environ:
            self.__distributed = True
        else:
            self.__distributed = False

        if self.__distributed:
            #self.__master_addr = socket.gethostbyname(os.environ["MASTER_ADDR"])
            self.__master_addr = os.environ["MASTER_ADDR"]
            self.__master_port = int(os.environ["MASTER_PORT"])
            self.__log_port = get_port_id(self.__master_port)
            self.__num_worker = int(os.environ["WORLD_SIZE"])
            self.__rank = int(os.environ["RANK"])
        else:
            self.__master_addr = socket.gethostbyname("localhost")
            self.__master_port = random.randint(20000, 30000)
            self.__log_port = get_port_id(self.__master_port)
            self.__num_worker = 1
            self.__rank = 0
        self.__message = Message(
            self.__master_addr, self.__log_port, self.__num_worker)
        if self.__rank == 0:
            self.__message.start_server()
        else:
            self.__message.start_client()
        if self.__rank == 0:
            self.__reset_current_log(delete=True)

        self.__LOG = self.__message.getlog()

        self.__config["TrainSetting"]["MasterAddr"] = self.__master_addr
        self.__config["TrainSetting"]["MasterPort"] = str(self.__master_port)
        log = ''
        log += get_log_header("Network Summary")
        log += "Master Addr.. {0}\n".format(
            self.__config["TrainSetting"]["MasterAddr"])
        log += "Master Port.. {0}\n".format(
            self.__config["TrainSetting"]["MasterPort"])
        if self.__distributed:
            log += "Log Port.. {0}\n".format(self.__log_port)
        log += "Num Workers.. {0}".format(self.__num_worker)
        if self.__rank == 0:
            self.__LOG.log(log)

    def __reset_current_log(self, delete=True):
        if self.__rank != 0:
            return
        train_dir = self.__config["TrainSetting"]["OutDir"]
        r = get_current_part_index_and_sent(self.__config)
        if not r:
            current_log_name = "default.log"
        else:
            iter_idx, part_idx, sent = r
            if iter_idx == "init":
                current_log_name = "init.log"
            else:
                current_log_name = "iter{0}.part{1}.log".format(iter_idx, part_idx)
        current_log_name = os.path.join(train_dir, current_log_name)
        self.__log_name = current_log_name
        if not os.path.exists(self.__config["TrainSetting"]["OutDir"]):
            os.mkdir(self.__config["TrainSetting"]["OutDir"])
        if delete:
            if os.path.exists(self.__log_name):
                os.remove(self.__log_name)
        self.__message.set_logpath(
            self.__log_name, self.__config["TrainSetting"].getint("NGpu"))

    def __init_train(self):
        # get last model name
        previous_model_name = get_previous_model_name(self.__config)
        previous_model_name = str(previous_model_name)

        # get output dir
        train_dir = self.__config["TrainSetting"]["OutDir"]

        # get current module name
        self.__config["TrainSetting"]["PreviousModel"] = previous_model_name
        r = get_current_part_index_and_sent(self.__config)
        if not r:
            self.__done = True
            return 0
        self.__done = False
        iter_idx, part_idx, sent = r
        if iter_idx == "init":
            current_model_name = "model.init"
            self.__is_inited = False
        else:
            current_model_name = "model.iter{0}.part{1}".format(
                iter_idx, part_idx)
            self.__is_inited = True
        self.__config["TrainSetting"]["CurrentModel"] = os.path.join(
            train_dir, current_model_name)

        # get training type
        train_type = self.__config['TrainSetting'].get('TrainType')
        train_type = "ASR" if train_type is None else train_type
        self.__config['TrainSetting']["TrainType"] = train_type

        # set data
        if self.__config['TrainSetting'].get('TrainType') == "LM":
            data_dir = self.__config["NNLMSetting"]["DataDir"]
            current_text_file = self.__config["NNLMSetting"]["TextPrefix"].replace('$', str(part_idx))
            current_text_file = os.path.join(data_dir, current_text_file)
            if not os.path.exists(current_text_file):
                self.__raise_err("{0} doesn't exist".format(current_text_file))
        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            data_dir = self.__config["DataSetting"]["DataDir"]
            label_dir = self.__config["DataSetting"]["LabelDir"]
            current_label_pfile = self.__config["DataSetting"]["LabelPrefix"].replace('$', str(part_idx))
            current_feature_pfile = self.__config["DataSetting"]["FeaturePrefix"].replace('$', str(part_idx))
            current_label_pfile = os.path.join(label_dir, current_label_pfile)
            current_feature_pfile = os.path.join(data_dir, current_feature_pfile)
            norm_file = self.__config["DataSetting"]["NormFile"]
            if not os.path.exists(current_label_pfile):
                self.__raise_err("{0} doesn't exist".format(current_label_pfile))
            if not os.path.exists(current_feature_pfile):
                self.__raise_err("{0} doesn't exist".format(current_feature_pfile))
            if not os.path.exists(norm_file):
                self.__raise_err("{0} doesn't exist".format(norm_file))

        if sent == "all":
            start_sent = 0
            if self.__config['TrainSetting'].get('TrainType') == "LM":
                end_sent=0
                with open(current_text_file) as f:
                    for l in f:
                        end_sent+=1
            if self.__config['TrainSetting'].get('TrainType') == "ASR":
                end_sent = int(Pfileinfo(current_feature_pfile).num_sentences)
        else:
            start_sent, end_sent = sent.split('-')
            start_sent = int(start_sent)
            end_sent = int(end_sent)

        validation_sentnum = self.__config["TrainSetting"].getint("CVSentNum")
        if not (validation_sentnum > 0):
            self.__raise_err("CVSentNum must be a positive value")
        
        if self.__config['TrainSetting'].get('TrainType') == "LM":
            validation_datadir = self.__config["NNLMSetting"]["ValidationDatadir"]
            validation_text_file = self.__config["NNLMSetting"]["ValidationText"]
            validation_text_file = os.path.join(validation_datadir, validation_text_file)
            if not os.path.exists(validation_text_file):
                self.__raise_err("{0} doesn't exist".format(validation_text_file))
            validation_start_sent = 0
            validation_end_sent = validation_sentnum

        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            if self.__config["DataSetting"].get("ValidationDatadir"):
                validation_datadir = self.__config["DataSetting"]["ValidationDatadir"]
                if not (self.__config["DataSetting"].get("ValidationLabel")):
                    self.__raise_err("validation label isn't specified")
                if not (self.__config["DataSetting"].get("ValidationFeature")):
                    self.__raise_err("validation feature isn't specified")
                validation_label_pfile = self.__config["DataSetting"]["ValidationLabel"]
                validation_feature_pfile = self.__config["DataSetting"]["ValidationFeature"]
                validation_label_pfile = os.path.join(validation_datadir, validation_label_pfile)
                validation_feature_pfile = os.path.join(validation_datadir, validation_feature_pfile)
                if not os.path.exists(validation_label_pfile):
                    self.__raise_err("{0} doesn't exist".format(
                        validation_label_pfile))
                if not os.path.exists(validation_feature_pfile):
                    self.__raise_err("{0} doesn't exist".format(
                        validation_feature_pfile))
                validation_start_sent = 0
                validation_end_sent = validation_sentnum
            else:
                validation_label_pfile = current_label_pfile
                validation_feature_pfile = current_feature_pfile
                if not (validation_sentnum <= end_sent):
                    self.__raise_err(
                        "CVSentNum must be smaller than train sentnum")
                end_sent = end_sent - validation_sentnum
                validation_start_sent = end_sent
                validation_end_sent = end_sent + validation_sentnum
            bunchsize = self.__config["DataSetting"].getint("Bunchsize")
            padnum = self.__config["DataSetting"].getint("PadNum")
            maxsentframe = self.__config["DataSetting"].getint("MaxSentFrame")
            maxnumsent = self.__config["DataSetting"].getint("MaxNumSent")

        if self.__config['TrainSetting'].get('TrainType') == "LM":
            cnt=0
            with open(current_text_file) as f:
                for l in f:
                    cnt+=1
            train_iternum = cnt // self.__config["NNLMSetting"].getint("BatchSize")
        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            train_iternum = Pfileinfo.estimate_num_batch(Pfileinfo(
            current_feature_pfile).seq_info[start_sent: end_sent], bunchsize, maxsentframe, maxnumsent, padnum)
        if self.__is_inited:
            train_iternum = int(
                train_iternum / self.__config["TrainSetting"].getint("NGpu"))
        validation_iternum = int(validation_end_sent - validation_start_sent)
        
        if self.__config['TrainSetting'].get('TrainType') == "LM":
            self.__config["NNLMSetting"]["TrainTextfile"] = current_text_file
            self.__config["NNLMSetting"]["ValidationTextfile"] = validation_text_file
            self.__config["NNLMSetting"]["ValidationStartSent"] = str(validation_start_sent)
            self.__config["NNLMSetting"]["ValidationEndSent"] = str(validation_end_sent)
            self.__config["NNLMSetting"]["TrainStartSent"] = str(start_sent)
            self.__config["NNLMSetting"]["TrainEndSent"] = str(end_sent)
        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            self.__config["DataSetting"]["TrainFeaturePfile"] = current_feature_pfile
            self.__config["DataSetting"]["TrainLabelPfile"] = current_label_pfile
            self.__config["DataSetting"]["TrainStartSent"] = str(start_sent)
            self.__config["DataSetting"]["TrainEndSent"] = str(end_sent)
            self.__config["DataSetting"]["NormFilePath"] = norm_file
            self.__config["DataSetting"]["ValidationFeaturePfile"] = validation_feature_pfile
            self.__config["DataSetting"]["ValidationLabelPfile"] = validation_label_pfile
            self.__config["DataSetting"]["ValidationStartSent"] = str(
            validation_start_sent)
            self.__config["DataSetting"]["ValidationEndSent"] = str(
            validation_end_sent)
        self.__config["DataSetting"]["TrainIterNum"] = str(train_iternum)
        self.__config["DataSetting"]["ValidationIterNum"] = str(
            validation_iternum)
        if self.__rank == 0:
            log = ''
            log += get_log_header("Model and Data Summary")
            log += "previous model.. {0}\n".format(
                self.__config["TrainSetting"]["PreviousModel"])
            log += "current model.. {0}\n".format(
                self.__config["TrainSetting"]["CurrentModel"])
            if self.__config['TrainSetting'].get('TrainType') == "LM":
                log += "train data.. {0}, {1}-{2}\n".format(self.__config["NNLMSetting"]["TrainTextfile"],
                                                        self.__config["NNLMSetting"]["TrainStartSent"],
                                                        self.__config["NNLMSetting"]["TrainEndSent"])
                log += "validation data.. {0}, {1}-{2}\n".format(self.__config["NNLMSetting"]["ValidationTextfile"],
                                                             self.__config["NNLMSetting"]["ValidationStartSent"],
                                                             self.__config["NNLMSetting"]["ValidationEndSent"])
            if self.__config['TrainSetting'].get('TrainType') == "ASR":
                log += "train data.. {0}, {1}-{2}\n".format(self.__config["DataSetting"]["TrainLabelPfile"],
                                                        self.__config["DataSetting"]["TrainStartSent"],
                                                        self.__config["DataSetting"]["TrainEndSent"])
                log += "validation data.. {0}, {1}-{2}\n".format(self.__config["DataSetting"]["ValidationLabelPfile"],
                                                             self.__config["DataSetting"]["ValidationStartSent"],
                                                             self.__config["DataSetting"]["ValidationEndSent"])
                log += "norm file.. {0}\n".format(
                self.__config["DataSetting"]["NormFilePath"])

            log += "train iternum.. {0}\n".format(
                self.__config["DataSetting"].getint("TrainIterNum"))
            log += "validation iternum.. {0}".format(
                self.__config["DataSetting"].getint("ValidationIterNum"))
            self.__LOG.log(log)

        half_iter_idx = self.__config["TrainSetting"].getint("Half")
        half_hold = self.__config["TrainSetting"].getint("HalfHold")
        if iter_idx == "init":
            current_lr = self.__config["TrainSetting"].getfloat(
                "InitLearningRate")
        else:
            if iter_idx < half_iter_idx:
                current_lr = self.__config["TrainSetting"].getfloat(
                    "LearningRate")
            else:
                # current_lr = self.__config["TrainSetting"].getfloat("LearningRate") / (
                #     2 ** (iter_idx - half_iter_idx + 1))
                current_lr = self.__config["TrainSetting"].getfloat("LearningRate") / (
                    2 ** (math.floor((iter_idx - half_iter_idx)/half_hold) + 1))
        self.__config["TrainSetting"]["CurrentLearningRate"] = str(current_lr)

        if self.__rank == 0:
            log = ''
            log += get_log_header("Train Setting Summary")
            log += "optimizer.. {0}\n".format(
                self.__config["TrainSetting"]["Optimizer"])
            log += "use lookahead.. {0}\n".format(
                self.__config["TrainSetting"].getboolean("LookAhead"))
            if self.__config["TrainSetting"].getboolean("LookAhead"):
                log += "lookahead parameters.. K={0} alpha = {1}\n".format(
                    self.__config["TrainSetting"].getint("LookAhead_K"),
                    self.__config["TrainSetting"].getfloat("LookAhead_Alpha"))
            log += "learning rate.. {0}".format(
                self.__config["TrainSetting"].getfloat("CurrentLearningRate"))
            self.__LOG.log(log)

    def __loop(self):
        while True:
            self.__reset_current_log(delete=False)
            self.__init_train()
            if self.__done:
                break
            self.__train_next_part()
        self.__clean_activites()

    def __clean_activites(self):
        if self.__message is not None:
            self.__message.shutdown()

    def __train_next_part(self):
        if not self.__is_inited:
            if (self.__rank == 0 and self.__distributed):
                self.__config["TrainSetting"]["GlobalRank"] = str(0)
                self.__config["TrainSetting"]["LocalRank"] = str(0)
                message_queue = context.Queue()
                p = context.Process(target=single_gpu_warper, args=(
                    self.__config, message_queue, self.__LOG,))
                p.start()
                status = message_queue.get()
                p.join()
                if status == 0:
                    self.__LOG.log("train process failed")
                    self.__raise_err("train process failed")
                self.__message.wait()
            if (self.__rank != 0 and self.__distributed):
                # pcli2: other workers will wait main worker done.
                self.__message.wait()
                self.__LOG.log(
                    "worker {0} received model initializing finished signal from master worker".format(self.__rank))
            if (not self.__distributed):
                # self.__config["TrainSetting"]["GlobalRank"] = str(0)
                # self.__config["TrainSetting"]["LocalRank"] = str(0)
                # message_queue = context.Queue()
                # p = context.Process(target=single_gpu_warper, args=(
                #     self.__config, message_queue, self.__LOG))
                # p.start()
                # status = message_queue.get()
                # p.join()
                # if status == 0:
                #     self.__LOG.log("train process failed")
                #     self.__raise_err("train process failed")
                run_single_gpu(self.__config, self.__LOG)
        else:
            if self.__distributed:
                self.__config["TrainSetting"]["InitMethod"] = "tcp://{0}:{1}".format(
                    self.__config["TrainSetting"]["MasterAddr"], self.__config["TrainSetting"]["MasterPort"])
                num_gpu_per_node = int(
                    self.__config["TrainSetting"].getint("NGpu") / self.__num_worker)
                process = []
                message_queue = context.Queue()
                for gpu in range(num_gpu_per_node):
                    self.__config["TrainSetting"]["GlobalRank"] = str(
                        num_gpu_per_node * self.__rank + gpu)
                    self.__config["TrainSetting"]["LocalRank"] = str(gpu)
                    p = context.Process(target=multi_gpu_warper, args=(
                        self.__config, message_queue, self.__LOG))
                    p.start()
                    process.append(p)

                status = 1
                for i in range(num_gpu_per_node):
                    status = status * message_queue.get()
                for p in process:
                    p.join()
                if status == 0:
                    self.__LOG.log("train process failed")
                    self.__raise_err("train process failed")

                self.__message.wait()
            else:
                if self.__config["TrainSetting"].getint("NGpu") > 1:
                    self.__config["TrainSetting"]["InitMethod"] = "tcp://{0}:{1}".format(
                        self.__config["TrainSetting"]["MasterAddr"], self.__config["TrainSetting"]["MasterPort"])
                    num_gpu_per_node = int(
                        self.__config["TrainSetting"].getint("NGpu") / self.__num_worker)
                    process = []
                    message_queue = context.Queue()
                    for gpu in range(num_gpu_per_node):
                        self.__config["TrainSetting"]["GlobalRank"] = str(
                            num_gpu_per_node * self.__rank + gpu)
                        self.__config["TrainSetting"]["LocalRank"] = str(gpu)
                        p = context.Process(target=multi_gpu_warper, args=(
                            self.__config, message_queue, self.__LOG))
                        p.start()
                        process.append(p)

                    status = 1
                    for i in range(num_gpu_per_node):
                        status = status * message_queue.get()
                    for p in process:
                        p.join()
                    if status == 0:
                        self.__LOG.log("train process failed")
                        self.__raise_err("train process failed")
                if self.__config["TrainSetting"].getint("NGpu") == 1:
                    self.__config["TrainSetting"]["GlobalRank"] = str(0)
                    self.__config["TrainSetting"]["LocalRank"] = str(0)
                    # message_queue = context.Queue()
                    # p = context.Process(target=single_gpu_warper, args=(
                    #     self.__config, message_queue, self.__LOG))
                    # p.start()
                    # status = message_queue.get()
                    # p.join()
                    # if status == 0:
                    #     self.__LOG.log("train process failed")
                    #     self.__raise_err("train process falied")
                    run_single_gpu(self.__config, self.__LOG)

    def __raise_err(self, info):
        self.__clean_activites()
        raise Exception(info)

    def __check_config(self):
        status = gen_part_list(self.__config)
        if isinstance(status, str):
            self.__raise_err(status)

        try:
            NGpu = self.__config["TrainSetting"].getint("NGpu")
            assert NGpu > 0
        except Exception:
            self.__raise_err("NGpu must be a integers")

        try:
            Iter = self.__config["TrainSetting"].getint("Iter")
            assert Iter > 0
            Half = self.__config["TrainSetting"].getint("Half")
            assert Half > 0
            HalfHold = self.__config["TrainSetting"].getint("HalfHold")
            assert HalfHold > 0
        except Exception:
            self.__raise_err("Iter and Half must be >0 integers")

        try:
            InitLearningRate = self.__config["TrainSetting"].getfloat(
                "InitLearningRate")
            assert InitLearningRate > 0
            LearningRate = self.__config["TrainSetting"].getfloat(
                "LearningRate")
            assert LearningRate > 0
        except Exception:
            self.__raise_err(
                "InitLearningRate and LearningRate must be floats")

        try:
            Optimizer = self.__config["TrainSetting"]["Optimizer"]
            assert Optimizer == "SGD" or Optimizer == "ADAM"
        except Exception:
            self.__raise_err("Optimizer must be SGD or ADAM")
