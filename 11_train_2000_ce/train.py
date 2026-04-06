import sys,time
from asr.train import train
if __name__ == "__main__":
    #time.sleep(100000000)
    mytrain = train()
    mytrain.load_config(sys.argv[1])
    mytrain.start_train()