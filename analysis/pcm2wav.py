#encoding=utf-8
import os
import sys

scp_path = sys.argv[1]
dst_wav_dir = sys.argv[2]

if not os.path.exists(dst_wav_dir):
    os.makedirs(dst_wav_dir)

for src_path in open(scp_path, "r").readlines():
    src_path = src_path.rstrip().split("=")[-1]
    dst_path = dst_wav_dir + "/" + os.path.basename(src_path).replace(".pcm", ".wav")
    cmd = "sox -t raw -c 1 -e signed-integer -b 16 -r 16000 %s %s" % (src_path, dst_path)
    #print(cmd)
    os.system(cmd)

