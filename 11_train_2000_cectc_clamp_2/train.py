import sys, os
from asr.train import train
import configparser

if __name__ == "__main__":

    '''
    Add by qhxu3.
    Copy configuration and net to train dir.
    '''
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    train_dir = config["TrainSetting"]["OutDir"]
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
        
    train_net = config["Model"]["ModelName"].split(".")[0]

    if os.path.exists(train_dir + "/" + sys.argv[1]):
        print("Warning: this dir has already config files, make sure set the right OutDir!")
        print("Warning: Old files will be sent to .files_back")

        if not os.path.exists(train_dir + "/.files_back"):
            os.mkdir(train_dir + "/.files_back")

        os.system("mv {} ./{}/".format(train_dir + "/" + sys.argv[1], train_dir + "/.files_back/"))
        os.system("mv {} ./{}/".format(train_dir + "/" + train_net + ".py", train_dir + "/.files_back/"))
        os.system("mv {} ./{}/".format(train_dir + "/train.py", train_dir + "/.files_back/"))
        os.system("mv {} ./{}/".format(train_dir + "/dataloader.py", train_dir + "/.files_back/"))
        
    os.system("cp {} ./{}/".format(sys.argv[1], train_dir))
    os.system("cp {} ./{}/".format(train_net + ".py", train_dir))
    os.system("cp {} ./{}/".format("./asr/train/train.py", train_dir))
    os.system("cp {} ./{}/".format("./asr/data/dataloader.py", train_dir))

    # ! for reloading configuration
    # if config["TrainSetting"].getboolean["Reload"]:
    #     os.system("cp {} config_for_reload.ini")

    # origin

    mytrain = train()
    mytrain.load_config(sys.argv[1])

    mytrain.start_train()