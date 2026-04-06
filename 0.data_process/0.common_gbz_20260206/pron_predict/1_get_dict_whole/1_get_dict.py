#coding=utf-8
import re

def deletnum(char):
    remove_chars = '[0-9()/]+' #去掉数字
    return re.sub(remove_chars,'',char)

def deletpunc(char):
    char=char.lower()#去除标点
    mypunc = '=|]'
    return re.sub(mypunc,'',char)

def process_line(line):
    line = line.strip()
    thai = line.split('[')[0].strip()
    phone = line.split('[')[1]
    phone = deletnum(deletpunc(phone))
    phone = ' '.join(phone.strip().split(' '))
    return thai+'\t'+phone
if __name__ == "__main__":
    #获取OOV发音
    in_file = open('output_202311/new_word.txt.out', 'r', encoding='utf-16le')
    #写到文件
    dict_file = open('output_202311/new_word.txt.out.dict', 'w')
    oov_dict = []
    for line in in_file.readlines():
        if line !='':
            line = line.strip()
            # เห[= ((hh ie)5)* ยี[=((jj ii)1)]* ยน [=((jj ohn)1)]
            if '*' or "#" in line:
                line = line.replace('#','*').replace(')]','*')
                words = line.split('*')
                str_word = ''
                str_ovv = ''
                for word_sub in words:
                    str_word += word_sub.split('[')[0].rstrip().lstrip()
                    str_ovv += word_sub.split('=')[-1].split(']')[0].rstrip() +" "
                line_merge = str_word.replace(' ','') +' [=' + str_ovv.rstrip() +']'

                #oov_dict += line.split('*')
                oov_dict.append(line_merge)

            else:
                words = line.split('[')
 
                p1 = words[0].replace(' ','')
 
                word = p1 + '[' + words[-1]
                #print (word)
                #print (line)
                #oov_dict.append(line)
                oov_dict.append(word)
    print ("oov before filer: {}".format(len(oov_dict)))
    oov_dict = list(filter(None, oov_dict))
    print ("oov after  filer: {}".format(len(oov_dict)))
    in_file.close()

    oov_dict_process = []
    for i in oov_dict:
        res = process_line(i)
        oov_dict_process.append(res)
        dict_file.write(res+'\n')
    dict_file.close()

    ##合并OOV和现有词汇
    ##base_file = open('./spanish_base.word2phone_nosp_dict', 'r',encoding="utf-8")
    #base_file = open('../lexicon_jap_cz.0302.txt', 'r')
    #base_dict = []
    #for line in base_file.readlines():
    #    line = line.strip()
    #    base_dict.append(line)
    #base_file.close()
    #
    #all_dict = oov_dict_process + base_dict
    #print ("base            : {}".format(len(base_dict)))
    #print ("base+oov        : {}".format(len(all_dict)))
    #all_dict = list(set(all_dict))
    #all_dict.sort()
    #print ("base+oov unique : {}".format(len(all_dict)))
    #
    ##out_file = open('train_data/spanish_base_new.word2phone_nosp_dict', 'w',encoding="utf-8")
    #out_file = open('../lexicon_jap_cz.0302_ddye2.txt', 'w')
    #for i in all_dict:
    #    out_file.write(i + '\n')
    #out_file.close()
