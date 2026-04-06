ky exp submit PtJob \
-d russian_multipfile \
-n train-cz-german \
--isModelTest \
--proID 2193 \
--modelPath /train8/asrmlg/ddye2/RNNT/german/20260114_gongban/11_train_2000_cectc_cz_260308 \
-e "train_multi_local_v100.sh" \
-l train.log \
-i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 \
-g 8 \
-w 2 \
-k TeslaV100-PCIE-48GB \
--useGpu \
-r dlp3-asrdictt-car-reserved
#-r dlp3-asrdictt-car-reserved
# \
#-s 'dlp2-98-153'
# \
#-s 'dlp2-99-221'
#--weChatOnStatus \
# \
#--weChatOnStatus \
#-s 'dlp2-98-135'
           
#--useDist \

        # ky exp submit PtJob \
        # -a zhyou2 \
        # --isModelTest \
        # --modelPath /work/asrdictt/zhyou2/0.test \
        # -d $name-simulation-ed-decode-$i-$p \
        # -e "train.sh" \
        # -l run.log \
        # -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 \
        # -g 4 \
        # -w 4 \
        # --elasticV100 \
        # --useGpu \
        # -r dlp3-asrdictt-reserved
        # # -k TeslaV100-PCIE-24GB \
        # # --modelName model.ky \