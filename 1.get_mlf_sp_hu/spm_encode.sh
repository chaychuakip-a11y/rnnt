python3 /raw15/asrdictt/permanent/zhyou2/202311_mlg_car/turkish/0_data_prepare/spm/scripts/spm_encode.py \
   --model "$(dirname "$0")/spm_hu_bpe_2000.model" \
   --output_format=piece \
   --inputs out.mlf_sy.sent \
   --outputs out.mlf_sy.sent.mlf_sp
