import torch
import torchintx
import sys,re
sys.path.append("..")
import net
# import confiner
test_model_name = sys.argv[1]

if __name__ == "__main__":

    state = torch.load(test_model_name)["state_dict"]
    torchintx.wb_analyse(state, "all_w.log")

    model = net.Transducer()

    data = torch.load("cn_data.pt")
    meta = torch.load("cn_meta.pt")

    torchintx.trace_layers(model, model.encoder, (data,meta), fuse_bn=True)
    # torchintx.SetBnMomentumUpdate(disable = True)

    clamp_modules=(torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Conv1d,torch.nn.LSTM, torch.nn.Embedding)
    # torchintx.disable_clamp(model.encoder.conv1)
    torchintx.clamp_module(model.encoder.conv2, clamp_weight_value=4, clamp_bias_value=8, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.conv3, clamp_weight_value=4, clamp_bias_value=8, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.conv4, clamp_weight_value=2, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.lstm, clamp_weight_value=1, clamp_bias_value=4, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.lstm2, clamp_weight_value=1, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.lstm3, clamp_weight_value=1, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.conv01, clamp_weight_value=1, clamp_bias_value=4, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.conv02, clamp_weight_value=2, clamp_bias_value=4, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.conv03, clamp_weight_value=2, clamp_bias_value=4, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.encoder.conv04, clamp_weight_value=2, clamp_bias_value=4, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.decoder.embedding, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.decoder.lstm, clamp_weight_value=1, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.decoder.lstm2, clamp_weight_value=1, clamp_bias_value=2, clamp_output_value=8, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.decoder.output_proj, clamp_weight_value=1, clamp_bias_value=1, clamp_output_value=8, clamp_dynamic_percent=1.0)
    # torchintx.clamp_module(model.joint.forward_layer, clamp_weight_value=1, clamp_bias_value=4, clamp_output_value=None, clamp_dynamic_percent=1.0)
    torchintx.clamp_module(model.joint.project_layer, clamp_weight_value=2, clamp_bias_value=2, clamp_output_value=None, clamp_dynamic_percent=1.0)
    model = torchintx.clamp_layers(model, clamp_modules = clamp_modules, clamp_weight_value=8, clamp_bias_value=8, clamp_output_value=None,clamp_dynamic_percent=1.0)

    
    model_dict = model.state_dict()
    model.load_state_dict(model_dict)
    
    
    with torchintx.Dumper() as dumper:
            model.eval()
            dumper.analyse_layer_output(model,match_pattern="root.")   # match_pattern 可支持查看对应哪些层
            model(data, meta) #跑一遍前向
            dumper.save_out_analyse_log(save_log_path="all_output.log") #日志保存路径
            # 此接口会在当前目录生成一个名为"Analyse_layer_output.log"的文件
