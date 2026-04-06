#encoding=utf-8
import onnxinfer
#sess = onnxinfer.InferFreezedOnnx("model_decoder.onnx", "fuse_model_decoder_14829.onnx", Only_GraphOptimization = True)
sess = onnxinfer.InferFreezedOnnx("./model_encoder.onnx", "./fuse_model_encoder.onnx", Only_GraphOptimization = True, keep_original= True)

sess = onnxinfer.InferFreezedOnnx("./model_decoder.onnx", "./fuse_model_decoder.onnx", Only_GraphOptimization = True, keep_original= True)

sess = onnxinfer.InferFreezedOnnx("./model_joint.onnx", "./fuse_model_joint.onnx", Only_GraphOptimization = True, keep_original= True)
#sess = onnxinfer.InferFreezedOnnx("model_joint.onnx", "fuse_model_joint_14829.onnx", Only_GraphOptimization = True)
sess = onnxinfer.InferFreezedOnnx("./model_ctc.onnx", "./fuse_model_ctc.onnx", Only_GraphOptimization = True, keep_original= True)
#sess.Run();
