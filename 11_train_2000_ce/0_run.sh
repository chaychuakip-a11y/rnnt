ky exp submit PtJob \
-d fr_multipfile \
-n train-fr \
--isModelTest \
--proID 2193 \
--modelPath /yrfs4/asrdictt/hjwang11/ \
-e "train_multi_local_v100.sh" \
-l train_multi_local_v100.log \
-i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 \
-g 8 \
-w 1 \
-k TeslaV100-PCIE-48GB \
--useGpu \
-r dlp3-asrdictt-car-reserved


# ky exp submit PtJob \
# -d fr_multipfile \
# -n train-fr \
# --isModelTest \
# --proID 2193 \
# --modelPath /yrfs4/asrdictt/hjwang11/ \
# -e "train_multi_local_v100.sh" \
# -l train_multi_local_v100.log \
# -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 \
# -g 8 \
# -w 4 \
# -k TeslaV100-PCIE-24GB \
# --useGpu \
# -r dlp3-asrdictt-car-reserved