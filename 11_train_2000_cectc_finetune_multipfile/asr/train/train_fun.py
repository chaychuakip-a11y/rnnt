# pcli2 2019 Nov.
# bmuf logic is from caffe
from ..data import PfileInfo, LmdbInfo, txtDataLoader, UnionDataLoader, TestUnionDataLoader
from ..utils import clip_grad_norm
from ..optim import Lookahead
from ..optim import SGD
from ..utils.message import *
from ..utils.train_helper import *
import copy
import torch.jit as jit
import torch.distributed as dist
import torch.optim as optim
import torch
import random

import torchintx


class BMUF():
    def __init__(self, model, config, dist):
        momentums = []
        global_models = []
        for param in model.parameters():
            temp = torch.zeros_like(param, requires_grad=False)
            temp.copy_(param.data)
            global_models.append(temp)
            momentums.append(torch.zeros_like(param, requires_grad=False))
        momentums = torch.nn.utils.parameters_to_vector(momentums)
        global_models = torch.nn.utils.parameters_to_vector(global_models)
        self.momentums = momentums
        self.global_models = global_models
        ### copy from /work/asrprg/hcwang17/tmp/tmp_for_xiaobao/train_init.py
        if os.path.exists("{}/_momentums.pth".format(config["TrainSetting"]["OutDir"])):
            momentums_dict = torch.load("{}/_momentums.pth".format(config["TrainSetting"]["OutDir"]),map_location=torch.device('cpu'))
            # print('===',momentums_dict['momentums'])
            # print('===',momentums_dict['epoch'])
            # print('===',momentums_dict['gpu_num'])
            if os.path.exists("{}/model.init".format(config["TrainSetting"]["OutDir"])):
                momentums = momentums_dict['momentums'].to(momentums)
                # momentums = momentums_dict['momentums']
            else:
                os.system('rm -rf {}/_momentums.pth'.format(config["TrainSetting"]["OutDir"]))
        ### copy from /work/asrprg/hcwang17/tmp/tmp_for_xiaobao/train_init.py
        self.bmuf_alpha = config["TrainSetting"].getfloat("BMUF_ALPHA")
        # rewrite by ssyan2 auto BMUF_BM
        self.bmuf_bm = config["TrainSetting"].getfloat("BMUF_BM") if config["TrainSetting"].getfloat("BMUF_BM") else  1. - 1./config["TrainSetting"].getint("NGpu") 
        self.bmuf_blr = config["TrainSetting"].getfloat("BMUF_BLR")
        self.dist = dist

    def update(self, model):
        self.__update_param(model)

    def __update_param(self, model):
        # for v, momentums, global_models in zip(model.parameters(), self.momentums, self.global_models):
        #     size = float(self.dist.get_world_size())
        #     avg = v.detach().clone()
        #     self.dist.all_reduce(avg.data, op=dist.ReduceOp.SUM)
        #     avg.data /= size
        #     update = self.bmuf_bm * momentums + global_models
        #     grad = avg - update
        #     momentums.copy_(self.bmuf_blr * grad +
        #                         self.bmuf_bm * momentums)
        #     global_models.copy_(global_models + momentums)
        #     update = self.bmuf_bm * momentums + global_models
        #     v.data.copy_(v.detach() - self.bmuf_alpha * (v.detach() - update))
        size = float(self.dist.get_world_size())
        v = torch.nn.utils.parameters_to_vector(model.parameters())
        avg = v.detach().clone()
        self.dist.all_reduce(avg.data, op=dist.ReduceOp.SUM)
        avg.data /= size
        update = self.bmuf_bm * self.momentums + self.global_models
        grad = avg - update
        self.momentums.copy_(self.bmuf_blr * grad +
                             self.bmuf_bm * self.momentums)
        self.global_models.copy_(self.global_models + self.momentums)
        update = self.bmuf_bm * self.momentums + self.global_models
        v.data.copy_(v.detach() - self.bmuf_alpha * (v.detach() - update))
        torch.nn.utils.vector_to_parameters(v, model.parameters())

def run_multi_gpu(config, LOG):
    dist.init_process_group("nccl", init_method=config["TrainSetting"]["InitMethod"],
                            world_size=config["TrainSetting"].getint("NGpu"),
                            rank=config["TrainSetting"].getint("GlobalRank"))
    if not torch.cuda.is_available():
        LOG.log("no gpu device is available")
        raise Exception("no gpu device is available")
    ##xxtong
    # np.random.seed(config["TrainSetting"].getint("RandomSeed"))
    # torch.manual_seed(config["TrainSetting"].getint("RandomSeed"))
    # torch.cuda.manual_seed(config["TrainSetting"].getint("RandomSeed"))
    torch.set_printoptions(10)

    model = get_module(config["Model"]["ModelName"])()
    
    print("train_fun.py in ")
    # # data = torch.load("rnnt_mdr_data.pt")
    # # meta = torch.load("rnnt_mdr_meta.pt")
    # # torchintx.trace_layers(model, model.encoder, (data,meta), fuse_bn=True)
    
    clamp_modules=(torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Conv1d,torch.nn.LSTM)

    aa = torch.ones((1,1,40,176))   #.cuda()
    model.eval()
    torchintx.trace_layers(model, model.encoder, aa, fuse_bn=True )
    ##3
    torchintx.clamp_module(model.encoder.conv2, clamp_weight_value=1, clamp_bias_value=4, clamp_output_value=2, clamp_dynamic_percent=1.0) # 200   180  150 100    60cv10 50  70cv4 55
    torchintx.clamp_module(model.encoder.conv3, clamp_weight_value=1, clamp_bias_value=3, clamp_output_value=2, clamp_dynamic_percent=1.0)#8
    torchintx.clamp_module(model.encoder.conv4, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=2, clamp_dynamic_percent=1.0)#4 88   8 96
    #
    torchintx.clamp_module(model.encoder.conv01, clamp_weight_value=4, clamp_bias_value=6, clamp_output_value=4, clamp_dynamic_percent=1.0)##ok
    torchintx.clamp_module(model.encoder.conv02, clamp_weight_value=1, clamp_bias_value=8, clamp_output_value=6, clamp_dynamic_percent=1.0)#7 ok  10
    torchintx.clamp_module(model.encoder.conv03, clamp_weight_value=1, clamp_bias_value=8, clamp_output_value=6, clamp_dynamic_percent=1.0)#30 OK
    torchintx.clamp_module(model.encoder.conv04, clamp_weight_value=0.5, clamp_bias_value=6, clamp_output_value=3, clamp_dynamic_percent=1.0)#8  4  4  20
    #
    torchintx.clamp_module(model.encoder.lstm, clamp_weight_value=3, clamp_bias_value=3, clamp_output_value=1, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.lstm2, clamp_weight_value=3, clamp_bias_value=3, clamp_output_value=1, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.lstm3, clamp_weight_value=3, clamp_bias_value=2, clamp_output_value=1, clamp_dynamic_percent=1.0)
    
    torchintx.clamp_module(model.decoder.lstm, clamp_weight_value=2, clamp_bias_value=3, clamp_output_value=1, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.decoder.lstm2, clamp_weight_value=3, clamp_bias_value=3, clamp_output_value=1, clamp_dynamic_percent=1.0)
    #
    torchintx.clamp_module(model.decoder.output_proj, clamp_weight_value=2, clamp_bias_value=1, clamp_output_value=1, clamp_dynamic_percent=1.0)
    #
    torchintx.clamp_module(model.joint.project_layer, clamp_weight_value=2, clamp_bias_value=4, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
    #
    torchintx.clamp_module(model.encoder.dnn_skip1, clamp_weight_value=1, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.dnn_skip2, clamp_weight_value=1, clamp_bias_value=3, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.dnn_skip_out, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
    
    torchintx.clamp_module(model.phone_ce.ctc_dnn, clamp_weight_value=2, clamp_bias_value=3, clamp_output_value=8, clamp_dynamic_percent=1.0)#
    torchintx.clamp_module(model.phone_ce.ctc_out, clamp_weight_value=2, clamp_bias_value=3, clamp_output_value=None, clamp_dynamic_percent=1.0)#
    
    model = torchintx.clamp_layers(model, clamp_modules = clamp_modules, clamp_weight_value=None, clamp_bias_value=None, clamp_output_value=None,clamp_dynamic_percent=1.0)


    # torchintx.SetPlatFormQuant(platform_quant=torchintx.PlatFormQuant.normal_quant)
    # replace_tuple_nolstm=(torchintx.ClampConvBN2d, torch.nn.Linear, torchintx.ClampLinear, torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d) #, torch.nn.Conv1d,torch.nn.LSTM
    # torchintx.quant_module_by_type(model, type_modules=replace_tuple_nolstm, data_bits=8, parameter_bits=8, out_bits=8)    
    # torchintx.SetIQTensorTanh(False)
    # torchintx.SetIQTensorCat(False)
    # torchintx.disable_quant(model.encoder.conv1)
    # torchintx.disable_quant(model.joint.forward_layer)
    # model = torchintx.init(model, quant_modules=replace_tuple_nolstm, data_bits=8, parameter_bits=8, out_bits=8, mode=torchintx.QuantMode.MaxValue)


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
            d=txtDataLoader(
                config["NNLMSetting"]["TrainTextfile"],
                config["NNLMSetting"]["DictFile"],
                config["NNLMSetting"].getint("BatchSize"),
                config["NNLMSetting"].getint("TrainStartSent"),
                config["NNLMSetting"].getint("TrainEndSent"),
                ndivide=config["TrainSetting"].getint("NGpu"),
                divide_index=config["TrainSetting"].getint("GlobalRank"),
                shuffle_batch=config["NNLMSetting"].getboolean("Shuffle"),
                num_workers=1)

        if config['TrainSetting'].get('TrainType') == "ASR":
           
            ########### rewrite by ssyan2 start
            data_infos = eval(config["TrainDataSetting"]["data_infos"])
            
            d = UnionDataLoader(
                data_infos=data_infos,
                bunchsize=config["TrainDataSetting"].getint("Bunchsize"),
                batch_num=config["DataSetting"].getint("TrainIterNum")-1,
                #batch_num=110,
                maxsentframe=config["TrainDataSetting"].getint("MaxSentFrame"),
                maxnumsent=config["TrainDataSetting"].getint("MaxNumSent"),
                ndivide=config["TrainSetting"].getint("NGpu"),
                divide_index=config["TrainSetting"].getint("GlobalRank"),
                nmod_pad=config["TrainDataSetting"].getint("PadNum"),
                shuffle_batch=config["TrainSetting"].getboolean("ShuffleBatch"),
                random_seed=config["TrainSetting"].getint("RandomSeed"),
                batch_ctrl=config["TrainDataSetting"].getboolean("BatchCtrl"))
            ########### rewrite by ssyan2 end

        model = model.cuda()
        
        # for name, parameter in model.named_parameters():
        #     if "encoder" in name:
        #         if "dnn_skip" in name:
        #             continue
        #         else:
        #             parameter.requires_grad=False
        #     if "phone_ce" in name:
        #         parameter.requires_grad=False

        if config["TrainSetting"].getboolean("JIT") == True:
            model = jit.script(model)
        # optimizer
        if config["TrainSetting"]["Optimizer"] == "SGD":
            # for name, parameter in model.named_parameters():
            #     if "encoder" in name:
            #         print(name, parameter.requires_grad)
            # exit()
            optimizer = SGD(
                # model.parameters(),
                filter(lambda p : p.requires_grad, model.parameters()),
                lr=config["TrainSetting"].getfloat("CurrentLearningRate"),
                momentum=0.9,
                weight_decay=0,
                nesterov=True)
            print("SGD optimizer set done")
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

    # np.random.seed(config["TrainSetting"].getint("RandomSeed"))
    # torch.manual_seed(config["TrainSetting"].getint("RandomSeed"))
    # torch.cuda.manual_seed(config["TrainSetting"].getint("RandomSeed"))
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
            ########### rewrite by ssyan2 start
            data_infos = eval(config["TrainDataSetting"]["data_infos"])
            
            d = UnionDataLoader(
                data_infos=data_infos,
                bunchsize=config["TrainDataSetting"].getint("Bunchsize"),
                batch_num=config["DataSetting"].getint("TrainIterNum")-1,
                #batch_num=400,
                maxsentframe=config["TrainDataSetting"].getint("MaxSentFrame"),
                maxnumsent=config["TrainDataSetting"].getint("MaxNumSent"),
                ndivide=1,  ###single_gpu =1!!!
                divide_index=config["TrainSetting"].getint("GlobalRank"),
                nmod_pad=config["TrainDataSetting"].getint("PadNum"),
                shuffle_batch=config["TrainSetting"].getboolean("ShuffleBatch"),
                random_seed=config["TrainSetting"].getint("RandomSeed"),
                batch_ctrl=config["TrainDataSetting"].getboolean("BatchCtrl"))
            ########### rewrite by ssyan2 end
            

        clamp_modules=(torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Conv1d,torch.nn.LSTM)

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
        d=txtDataLoader(
            config["NNLMSetting"]["ValidationTextfile"],
            config["NNLMSetting"]["DictFile"],
            1,
            config["NNLMSetting"].getint("ValidationStartSent"),
            config["NNLMSetting"].getint("ValidationEndSent"),
            ndivide=1,
            divide_index=config["TrainSetting"].getint("GlobalRank"),
            shuffle_batch=False,
            num_workers=1)
        validation(model, config, d, LOG)
    if config['TrainSetting'].get('TrainType') == "ASR":

        ########### rewrite by ssyan2 start
        dsets = []
        # 加载添加的数据集
        for ind,data_add in enumerate(eval(config["TrainDataSetting"]["List"])):
            if config[data_add].getint("ValidationIterNum") > 0:
                dsets.append(data_add)
        print("val datasetings",dsets)
        for datasetting in dsets:
            LOG.log("\nvalidated data {0}".format(datasetting))

            get_data_infos = eval(config["TrainDataSetting"]["data_infos"])
            val_data_infos = {
                "file_norm":get_data_infos["file_norm"],
                "data_list":[datasetting],
                }
            
            val_data_infos[datasetting] = copy.deepcopy(get_data_infos[datasetting])
            for val_k in get_data_infos[datasetting].keys():
                if "val_" == val_k[:4]:
                    val_data_infos[datasetting][val_k[4:]] = get_data_infos[datasetting][val_k]
            val_data_infos[datasetting]["mix_rate"] = 1.0

            d = TestUnionDataLoader(
                data_infos=val_data_infos,
                batch_num=config[datasetting].getint("ValidationIterNum")-1,
                bunchsize=config["TrainDataSetting"].getint("Bunchsize"),
                maxsentframe=4096,
                maxnumsent=1,
                ndivide=1,
                divide_index=0,
                num_workers=1,
                nmod_pad=config["TrainDataSetting"].getint("PadNum"),
                shuffle_batch=False,
                val=True)
            
            

            validation(model, config, d, LOG)
        ########### rewrite by ssyan2 end
    
    

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
        # celoss = model(data, meta)
        totalloss, edloss,celoss = model(data, meta) # xiaobao
        return totalloss,edloss, celoss
    def forward(self, index, data_element, model):
        if self.train_type == "ASR":
            return self.asr_forward(index, data_element, model)
        if self.train_type == "LM":
            return self.lm_forward(index, data_element, model)
    

def bmuf_train(model, bmuf, optimizer, config, data_loader, LOG):
    # if config["TrainSetting"].getint("GlobalRank") == 0:
    #     for i in range(10000,22001, 1000):
    #         path = f'{config["TrainSetting"]["PreviousModel"]}.{i}'
    #         LOG.log("modelname.. {0}".format(path))
    #         print(path)
    #         state_dict = torch.load(path)
    #         model.load_state_dict(state_dict["state_dict"],strict=False)
    #         run_validation(config, model, LOG)
        # run_validation(config, model, LOG)
    # exit()

    model.train()
    # for name, module in model.named_modules():
    #     # print(name, module)
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         module.training = False

    # sum_of_last_displayed_loss = 0
    # sum_of_last_displayed_celoss = 0
    sum_of_last_displayed_loss = {}
    last_t = time.time()

    cur_model = config["TrainSetting"]["CurrentModel"]
    list_model_name = cur_model.split(".")
    print(list_model_name,len(data_loader))
    cur_iter = list_model_name[-2].replace("iter","")
    cur_part = list_model_name[-1].replace("part","")
    
    # if not os.path.exists("train_contiguous_new") and config["TrainSetting"].getint("GlobalRank") == 0:
    #     os.mkdir("train_contiguous_new") 

    trainer_class = get_module(config['TrainSetting'].get("Trainer")) if config['TrainSetting'].get("Trainer") is not None else Trainer
    trainer = trainer_class(config['TrainSetting'].get('TrainType'))
    # print("################################")
    # print("it is using the right dataloader")
    # # print("data_loader=", len(data_loader))
    # print("################################")

    for i, data_element in enumerate(data_loader):
         
        # for name, parameter in model.named_parameters():
        #     if "encoder" in name:
        #         print(i, name, parameter.requires_grad)
        #     if "phone_ce" in name:
        #         print(i, name, parameter.requires_grad)

        optimizer.zero_grad()

        # loss, edloss, celoss = trainer.forward(i, data_element, model) # xiaobao
        
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
        
        # sum_of_last_displayed_loss += edloss.item()
        # sum_of_last_displayed_celoss += celoss.item()
        for key in loss_dict:
            if key not in sum_of_last_displayed_loss:
                sum_of_last_displayed_loss[key] = loss_dict[key].item()
            else:
                sum_of_last_displayed_loss[key] += loss_dict[key].item()

        clip_grad_norm(model.parameters(), config["TrainSetting"].getfloat("ClipGradient"),
                       config["TrainSetting"].getfloat("ClipGradient2"), config["TrainSetting"].getfloat("Discount"))
        optimizer.step()

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
        torch.save({'momentums':bmuf.momentums, 'epoch':config["TrainSetting"]["CurrentModel"], 'gpu_num':int(config["TrainSetting"]["NGpu"])}, \
            "{}/_momentums.pth".format(config["TrainSetting"]["OutDir"]))


def simple_train(model, optimizer, config, data_loader, LOG):
    model.train()
    # sum_of_last_displayed_loss = 0
    # sum_of_last_displayed_celoss = 0
    sum_of_last_displayed_loss = {}
    last_t = time.time()

    trainer_class = get_module(config['TrainSetting'].get("Trainer")) if config['TrainSetting'].get("Trainer") is not None else Trainer
    trainer = trainer_class(config['TrainSetting'].get('TrainType'))
    
    for i, data_element in enumerate(data_loader):
    
        optimizer.zero_grad()

        # loss, edloss, celoss = trainer.forward(i, data_element, model) # xiaobao
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
        # sum_of_last_displayed_loss += edloss.item() 
        # sum_of_last_displayed_celoss += celoss.item()
        for key in loss_dict:
            if key not in sum_of_last_displayed_loss:
                sum_of_last_displayed_loss[key] = loss_dict[key].item()
            else:
                sum_of_last_displayed_loss[key] += loss_dict[key].item()
                
        clip_grad_norm(model.parameters(), config["TrainSetting"].getfloat("ClipGradient"),
                       config["TrainSetting"].getfloat("ClipGradient2"), config["TrainSetting"].getfloat("Discount"))
        optimizer.step()

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
        # criterion,ce,pad = trainer.forward(i, data_element, model)
        # #print(criterion)
        # criterion = criterion.item()
        acc_dict = trainer.forward(i, data_element, model)
        for key in acc_dict:
            if key not in sum_of_last_displayed_acc:
                try:
                    sum_of_last_displayed_acc[key] = [acc_dict[key].item()]
                except:
                    print(key, acc_dict[key])
                    exit()
            else:
                sum_of_last_displayed_acc[key].append(acc_dict[key].item())
                
        if i % config["TrainSetting"].getint("Display") == 0:
            LOG.log("validated {0} / {1}".format(i,
                                                 config["TrainSetting"].getint("CVSentNum")))
        # criterion_holder.append(criterion)
  
    # try:
    #     model.total_dist = 0
    #     model.total_dist_skip = 0
    #     model.total_word = 0
    #     model.total_skip_count_enc = 0
    #     model.total_skip_count_dec = 0
    #     model.total_t_length = 0
    # except Exception as e:
    #     print(e)
    model.fixstat = 0

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
        # celoss = model(data, meta)
        totalloss, edloss,celoss = model(data, meta) # xiaobao
        return totalloss,edloss, celoss
    def forward(self, index, data_element, model):
        if self.train_type == "ASR":
            return self.asr_forward(index, data_element, model)
        if self.train_type == "LM":
            return self.lm_forward(index, data_element, model)