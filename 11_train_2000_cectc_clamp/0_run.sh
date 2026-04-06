ky exp submit PtJob \
-d fr_clamp2 \
-n train-fr-clamp2 \
--isModelTest \
--proID 2193 \
--modelPath /yrfs4/asrdictt/hjwang11/ \
-e "train_multi_local_v100.sh" \
-l train_multi_local_v100.log \
-i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 \
-g 8 \
-w 4 \
-k TeslaV100-PCIE-24GB \
--useGpu \
-r dlp3-asrdictt-car-reserved


# ky exp submit PtJob \
# -d id_clamp2 \
# -n train-id-clamp2 \
# --isModelTest \
# --proID 2193 \
# --modelPath /yrfs4/asrdictt/hjwang11/ \
# -e "train_multi_local_v100.sh" \
# -l train_multi_local_v100.log \
# -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 \
# -g 4 \
# -w 6 \
# -k TeslaV100-PCIE-16GB \
# --useGpu \
# -r dlp3-asrdictt-car-reserved
