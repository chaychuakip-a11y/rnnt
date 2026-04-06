decode_path=$1
python /train20/asrmlg/permanent/mingli18/multi/get_acc/get_french_res.py $decode_path
perl /yrfs4/asrtrans/yyhu7/work/2020_asr/train/多语种大模型效果测试-阿俄/french_to_ynsong/get_result_ysp_Num.pl $decode_path