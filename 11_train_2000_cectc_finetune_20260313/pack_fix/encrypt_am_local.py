#!/opt/tool/anaconda3.5/bin/python3
# coding=utf8

import hashlib
import os
import sys
import re

def GetFileMd5(filename):
    if not os.path.isfile(filename):
        return
    myhash = hashlib.md5()
    with open(filename,'rb') as ff:
        while True:
            b = ff.read(8096)
            if not b :
                break
            myhash.update(b)
    return myhash.hexdigest()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: py filename')
        exit(-1)
    filename = sys.argv[1]
    
    filename_enc = re.sub('(\.[^\.]+)$', lambda x:'_enc'+x.group(1), filename)
    bin_encrypt = '/work1/asrdictt/taoyu/bin/encrypt_test'
    #os.system('{} {} 12676506 {}'.format(bin_encrypt, filename, filename_enc))
    os.system('{} {} ivw70sse16 {}'.format(bin_encrypt, filename, filename_enc))

    md5sum = GetFileMd5(filename_enc)

    if md5sum and len(md5sum) == 32:
        print(md5sum)
        md5sum = md5sum[28:]
        filename_new = re.sub('(\.[^\.]+)$', lambda x:'_'+md5sum+x.group(1), filename_enc)
        if filename_new == filename:
            filename_new = filename+'_'+md5sum
        print(filename_new)
        os.rename(filename_enc, filename_new)
    else:
        print('Error: fail to get md5sum of file, '+filename)
