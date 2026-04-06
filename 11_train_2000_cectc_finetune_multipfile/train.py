#/home/sptrans/whzhang9/anaconda3/envs/v5mocha/lib/python3.8/site-packages
# from train_init import train
from asr.train import train, Train
import time

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys

if __name__ == "__main__":
    # thying
    # while True:
    #mytrain = train()
    mytrain = Train()
    mytrain.load_config(sys.argv[1])
    mytrain.start_train()

    print("----please restart----")
    print("----sleep 10s----",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print("----begin restart----",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
