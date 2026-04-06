

import re


# def load_dict(file_dict:str, encoding:str='gb18030'):
#     dd = dict()
#     with open(file_dict, mode='r', encoding=encoding) as fi:
#         for line in fi:
#             line = line.strip()
#             ll = re.split(r'\s+', line)
#             if ll[0] not in dd:
#                 dd[ll[0]] = list()
#             dd[ll[0]].append(' '.join(ll[1:]))
#     return dd
def load_dict(file_dict:str, encoding:str='gb18030'):
    dd = dict()
    with open(file_dict, mode='r', encoding=encoding) as fi:
        for line in fi:
            line = line.strip()
            ll = re.split(r'\s+', line)
            if ll[0].lower() not in dd:
                dd[ll[0].lower()] = list()
            dd[ll[0].lower()].append(' '.join(ll[1:]))
    return dd
file_dict = "/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/french/data_prepare/gen_fst_bin/dict/french_ly1213h_ly2k_xw696h_openSRC412h_fz1655h_ttsMusicVideoPoi_v16.word_dict_phone_nosp"
dd = load_dict(file_dict, 'utf-8')

oov_list=[]

filename = "out.mlf_ori"
output_filename = "lab.mlf_sp"  # 输出文件名
output_filename2 = "lab.mlf_sy.nopunc"  # 输出文件名
output_filename3 = "lab.mlf_sy.danzi"  # 输出文件名
oov_scp_filename = "lab.scp.oov"
good_scp_filename = "lab.scp.fine"

# ##arabic
# diatonic_notes = {
    # 'ً':'',
    # 'ٍ':'',
    # 'َ':'',
    # 'ِ':'',
    # 'ّ':'',
    # 'ُ':'',
    # 'ٌ':'',
    # 'ْ':'',
    # 'ـ':'',
    # '-':'',
    # '۰':'٠','۱':'١','۲':'٢','۳':'٣','۴':'٤','۵':'٥','۶':'٦','۷':'٧','۸':'٨','۹':'٩',    #bosi number
    # 'ڤ':'ف',
    # 'ک':'ك',
    # 'ٶ':'ؤ',
    # 'ٸ':'ئ',
    # 'ٱ':'آ',
    # 'ی':'ى',
    # 'پ':'ب',
    # 'چ':'ج',
    # 'گ':'ك',
    # 'ڨ':'ق',
    # 'ہ':'ه',
    # 'ۆ':'ؤ',
    # 'ۍ':'ى',
    # 'ڈ':'ذ',
    # 'ٲ':'آ',
    # 'ٳ':'إ',
    # 'ۈ':'ؤ',
    # 'ژ':"ز",
    # 'ڭ':'ك',
    # 'ھ':'ه',
    # 'ٰ':'',
    # 'ۗ':'',
    # 'ۖ':'',
    # 'ۗ':'',
    # 'ۚ':'',
    # 'ۜ':'',
    # 'ٔ':'',
    # 'ۙ':'',
    # 'ٓ':'',
    # 'ۦ':'',
    # 'ٖ':'',
    # 'ۡ':'',
    # 'ٗ':'',
    # 'ٕ':'',
    # 'ٞ':'',
    # 'ۥ':'',
    # 'ۢ':'',
    # '﴾':'',
    # '﴿':'',
    # 'ﷺ':'',
    # '۞':'',
# }
# ##doubtable marks: ـ- 
# # بتزينها

# chars = [
# "ا","ب","ج","د","ه","و","ز","ح","ط","ي","ك","ل","م","ن","س","ع","ف","ص","ق","ر","ش","ت","ث","خ","ذ","ض","ظ","غ",
# "ى","ئ","ء","أ","إ","آ","ؤ","ة",
# '٠','١','٢','٣','٤','٥','٦','٧','٨','٩',
# 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
# ]   

##french
diatonic_notes = {}

chars = [
'à', 'â', 'ç', 'é', 'è', 'ê', 'ë', 'î', 'ï', 'ô', 'û', 'ù', 'ü', 'ÿ', 'æ', 'œ', '\'', '-', ' ',
'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
]   

excluded_chars = [".", "..", "<s>", "</s>", "،", ".", "؟", "!", "؛", "douhao", "juhao", "wenhao", "gantanhao", "?", "!", "<juhao>", ",", "«", "»", ":", "<", ">", "\"",]  # 要排除的字符列表

cc=0

skp=0

file = open(filename, "r")
lines = file.readlines()  # 读取文件中的所有行
file.close()

# output_file = open(output_filename, "w")
output_file2 = open(output_filename2, "w")
output_file3 = open(output_filename3, "w")
output_file2.write("#!MLF!#\n")  # 将前面的内容写入到输出文件中
output_file3.write("#!MLF!#\n")  # 将前面的内容写入到输出文件中
            
oov_scp_file = open(oov_scp_filename, "w")
good_scp_file = open(good_scp_filename, "w")

oov_filename = "oov.list"
oov_list_file = open(oov_filename, "w")

bad_filename = "bad.list"
bad_list_file = open(bad_filename, "w")

modified_lines = []
modified_lines_nopunc = []
modified_lines_danzi = []
mlf_name = None
has_oov = False
for line in lines:
    line = line.strip()  # 去掉行首和行尾的空格
    # print(line);import time; time.sleep(1)
    if line.endswith(".mlf_ori\""):
        if mlf_name is not None and skp==0:
            # output_file.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
            # merged_lines = "\n".join(modified_lines)  # 使用join()函数将所有行合并为一个字符串
            # output_file.write(merged_lines + "\n.\n")  # 将前面的内容写入到输出文件中          
            ##nopunc
            output_file2.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
            modified_lines_nopunc = "\n".join(modified_lines_nopunc)  # 使用join()函数将所有行合并为一个字符串
            output_file2.write(modified_lines_nopunc + "\n.\n")  # 将前面的内容写入到输出文件中                 
            ##danzi
            output_file3.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
            modified_lines_danzi = "\n".join(modified_lines_danzi)  # 使用join()函数将所有行合并为一个字符串
            output_file3.write(modified_lines_danzi + "\n.\n")  # 将前面的内容写入到输出文件中
            ##check oov scp
            if has_oov is False:
                good_scp_file.write(mlf_name+"\n")
            else:
                has_oov = False  
        modified_lines = []  
        modified_lines_nopunc = []     
        modified_lines_danzi = []  
        content = line[:line.find(".mlf_ori\"")]  # 获取前面的内容并去掉开头的引号和空格
        mlf_name = content[1:] if content.startswith("\"") else content  # 去掉开头的引号
        skp=0
    elif line not in excluded_chars:  # 检查当前行是否在字符列表中
        # for key, value in diatonic_notes.items():
            # line = line.replace(key, value)  # 删除指定字符
            
        words = line.split(' ')
            
        for line in words:
            if line.lower() not in dd:
                if has_oov is False:
                    oov_scp_file.write(mlf_name+"\n")
                    has_oov = True
                for char in line.lower():
                    if char not in chars and skp ==0:
                        bad_list_file.write(line+"\n")
                        # print(cc, line.lower()); import time; time.sleep(1)
                        cc+=1
                        skp=1
                        break
                if line.lower() not in oov_list and skp==0:  
                    oov_list.append(line.lower())
        
            modified_lines_nopunc.append(line.lower())
            modified_lines_danzi.extend(line.lower())
        

print(cc)#;exit()

for ele in oov_list:
    oov_list_file.write(ele+"\n")
oov_list_file.close()
    
if skp==0:
    # output_file.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
    # merged_lines = "\n".join(modified_lines)  # 使用join()函数将所有行合并为一个字符串
    # output_file.write(merged_lines + "\n.\n")  # 将前面的内容写入到输出文件中
    # output_file.close()

    output_file2.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
    modified_lines_nopunc = "\n".join(modified_lines_nopunc)  # 使用join()函数将所有行合并为一个字符串
    output_file2.write(modified_lines_nopunc + "\n.\n")  # 将前面的内容写入到输出文件中
    output_file2.close()

    output_file3.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
    modified_lines_danzi = "\n".join(modified_lines_danzi)  # 使用join()函数将所有行合并为一个字符串
    output_file3.write(modified_lines_danzi + "\n.\n")  # 将前面的内容写入到输出文件中
    output_file3.close()
