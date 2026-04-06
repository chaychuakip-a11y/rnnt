import torch
import torchintx
import sys,re
sys.path.append("..")

import net

#import net_relu__addS_phCTC-add-hidden_skip_try2_para as net
# import confiner
#test_model_name = sys.argv[1]
#test_model_name = "/yrfs4/asrdictt/wwyang9/car_asr/V2_model_work/local_asr/rnnt/malay/11_train_2000_cectc/out_train_clamp_V2/model.iter0.part0"
#test_model_name = "/yrfs4/asrdictt/wwyang9/car_asr/V2_model_work/local_asr/rnnt/malay/11_train_2000_cectc/out_train_clamp_V1/model.init"
#test_model_name = "/yrfs4/asrdictt/wwyang9/car_asr/V2_model_work/local_asr/rnnt/malay/11_train_2000_cectc/cutted_cpt//out_train_260205_i3p0.pt"
test_model_name = sys.argv[1]
#test_model_name  = "/yrfs4/asrdictt/wwyang9/car_asr/V2_model_work/local_asr/rnnt/malay/11_train_2000_cectc/analysis/model.pt"

# test_model_name = "/yrfs4/asrdictt/wwyang9/car_asr/V2_model_work/local_asr/rnnt/malay/11_train_2000_cectc/out_train_clamp_V7/model.init"

if __name__ == "__main__":

    state = torch.load(test_model_name)["state_dict"]
    #state = torch.load(test_model_name)

    model = net.Transducer()
    #data = torch.load("data.pt")
    #meta = torch.load("meta.pt")
    #model_path="/yrfs4/asrdictt/wwyang9/car_asr/V2_model_work/local_asr/rnnt/malay/11_train_2000_cectc/cutted_cpt//out_train_260205_i3p0.pt"
    #model_path = test_model_name
    data, meta = torch.load("5.pt")

    torchintx.trace_layers(model, model.encoder, (data,meta), fuse_bn=True)
    torchintx.SetBnMomentumUpdate(disable = True)

    clamp_modules=(torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Conv1d,torch.nn.LSTM, torch.nn.Embedding)
    torchintx.disable_clamp(model.encoder.conv1)
    torchintx.disable_clamp(model.joint.forward_layer)
    torchintx.disable_clamp(model.encoder.dnn_skip_out)
    
    # torchintx.clamp_module(model.encoder.conv2, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=128, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.encoder.conv3, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=128, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.encoder.conv4, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.encoder.lstm, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.encoder.lstm2, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.encoder.lstm3, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.encoder.conv01, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=64, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.encoder.conv02, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=64, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.encoder.conv03, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=64, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.encoder.conv04, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.decoder.embedding, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.decoder.lstm, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.decoder.lstm2, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.decoder.output_proj, clamp_weight_value=4, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
    

    # torchintx.clamp_module(model.joint.forward_layer, clamp_weight_value=1, clamp_bias_value=4, clamp_output_value=None, clamp_dynamic_percent=1.0)
    #torchintx.clamp_module(model.joint.project_layer, clamp_weight_value=8, clamp_bias_value=8, clamp_output_value=None, clamp_dynamic_percent=1.0)
    #model = torchintx.clamp_layers(model, clamp_modules = clamp_modules, clamp_weight_value=None, clamp_bias_value=None, clamp_output_value=None,clamp_dynamic_percent=1.0)
    #  torchintx.clamp_module(model.encoder.conv2, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0) # 150 70 30 15 8
    #  torchintx.clamp_module(model.encoder.conv3, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0) # 100
    #  torchintx.clamp_module(model.encoder.conv4, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)#5 loss2  3 loss2 10

    #  # #
    #  torchintx.clamp_module(model.encoder.conv01, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)##ok

    #  torchintx.clamp_module(model.encoder.conv02, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)#7 ok  10
    #  torchintx.clamp_module(model.encoder.conv03, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)#30 OK
    #  torchintx.clamp_module(model.encoder.conv04, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)#8  4  4  20
    #  
    #  torchintx.clamp_module(model.encoder.lstm, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    #  torchintx.clamp_module(model.encoder.lstm2, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    #  torchintx.clamp_module(model.encoder.lstm3, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    #  
    #  torchintx.clamp_module(model.decoder.lstm, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    #  torchintx.clamp_module(model.decoder.lstm2, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    #  
    #  torchintx.clamp_module(model.decoder.output_proj, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    #  
    #  torchintx.clamp_module(model.joint.project_layer, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
    #  
    #  torchintx.clamp_module(model.encoder.dnn_skip1, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    #  torchintx.clamp_module(model.encoder.dnn_skip2, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    #  torchintx.clamp_module(model.encoder.dnn_skip1, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    #  torchintx.clamp_module(model.encoder.dnn_skip2, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    #  torchintx.clamp_module(model.encoder.dnn_skip_out, clamp_weight_value=4, clamp_bias_value=2, clamp_output_value=None, clamp_dynamic_percent=1.0)#butiao out
    model = torchintx.clamp_layers(model, clamp_modules = clamp_modules, clamp_weight_value=None, clamp_bias_value=None, clamp_output_value=None,clamp_dynamic_percent=1.0)
    


    torchintx.SetIQTensorTanh(True)
    torchintx.SetIQTensorCat(False)
    quant_modules = (torch.nn.Conv2d,torch.nn.BatchNorm2d, torch.nn.Linear,torch.nn.ConvTranspose2d,torch.nn.Conv1d,torch.nn.LSTM)
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
    
    model = torchintx.init(model, quant_modules = quant_modules,mode=torchintx.QuantMode.MaxValue, data_bits=8, parameter_bits=8, out_bits=8)
    
    model_dict = model.state_dict()
    model.load_state_dict(state)
    torch.save(model.state_dict(), "model.pt")
    state = model.state_dict()
    torchintx.wb_analyse(state, "all_w.log")
    
    
    with torchintx.Dumper() as dumper:
            model.eval()
            dumper.analyse_layer_output(model,match_pattern="root.")   # match_pattern 可支持查看对应哪些层
            model(data, meta) #跑一遍前向
            dumper.save_out_analyse_log(save_log_path="all_output.log") #日志保存路径
            # 此接口会在当前目录生成一个名为"Analyse_layer_output.log"的文件
