#!/bin/bash
# Stage 0 - 匈牙利语通用数据 FA 处理
# 使用前请先确认 config.json 中 hdfs_src_root / hdfs_out_root 已填写正确路径
#
# 执行顺序：
#   1. FA 对齐（基础数据）
#   2. 生成 seed MLF（随机打乱 index）
#
# 如需指定特定 HDFS 输入路径，可以在脚本参数中传入：
#   perl 1_dnnfa.pl /workdir/asrdictt/YOUR_PATH/hungarian_data

perl 1_dnnfa.pl
perl 2.0_GenSeedMlf.pl
