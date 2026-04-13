

import re
import sentencepiece as spm
import sys

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
# file_dict = ""
# dd = load_dict(file_dict, 'utf-8')

oov_list=[]

# spmodel_path = "/raw15/asrdictt/permanent/zhyou2/202311_mlg_car/turkish/0_data_prepare/spm/spm_tk_bpe2000.model"
# spmodel = spm.SentencePieceProcessor(model_file=spmodel_path)
def is_defined(variable_name):
    try:
        # 尝试访问变量
        eval(variable_name)
        return True
    except NameError:
        # 如果变量未定义，将抛出NameError
        return False

# filename = sys.argv[1]
filename = "out.mlf_sy"
output_filename = "{}.mlf_sp".format(filename)  # 输出文件名
output_filename2 = "{}.nopunc".format(filename)  # 输出文件名
output_filename3 = "{}.danzi".format(filename)  # 输出文件名
output_filename4 = "{}.sent".format(filename)  # 输出文件名
output_filename5 = "{}.scp".format(filename)  # 输出文件名
oov_scp_filename = "lab.scp.oov"
good_scp_filename = "lab.scp.fine"

excluded_chars = ["#!MLF!#", "#!mlf!#", ".", "..", "<s>", "</s>", "،", ".", "؟", "!", "؛", "douhao", "juhao", "wenhao", "gantanhao", "?", "!"]  # 要排除的字符列表
diatonic_notes = {
    'ً':'',
    'ٍ':'',
    'َ':'',
    'ِ':'',
    'ّ':'',
    'ُ':'',
    'ٌ':'',
    'ْ':'',
    'ـ':'',
    '-':'',
    '۰':'٠','۱':'١','۲':'٢','۳':'٣','۴':'٤','۵':'٥','۶':'٦','۷':'٧','۸':'٨','۹':'٩',    #bosi number
    'ڤ':'ف',
    'ک':'ك',
    'ٶ':'ؤ',
    'ٸ':'ئ',
    'ٱ':'آ',
    'ی':'ى',
    'پ':'ب',
    'چ':'ج',
    'گ':'ك',
    'ڨ':'ق',
    'ہ':'ه',
    'ۆ':'ؤ',
    'ۍ':'ى',
    'ڈ':'ذ',
    'ٲ':'آ',
    'ٳ':'إ',
    'ۈ':'ؤ',
    'ژ':"ز",
    'ڭ':'ك',
    'ھ':'ه',
    'ٰ':'',
    'ۗ':'',
    'ۖ':'',
    'ۗ':'',
    'ۚ':'',
    'ۜ':'',
    'ٔ':'',
    'ۙ':'',
    'ٓ':'',
    'ۦ':'',
    'ٖ':'',
    'ۡ':'',
    'ٗ':'',
    'ٕ':'',
    'ٞ':'',
    'ۥ':'',
    'ۢ':'',
    '﴾':'',
    '﴿':'',
    'ﷺ':'',
    '۞':'',
	'_':' ',
    '-':' ',
    '–':' ',
    '—':' ',
    '−':' ',
    '̇':'',
    '￼':'',
    '™':'',
    '♂':'',
    '️':'',
    '•':' ',
    '′':"'",
    'õ':'ő',  # Hungarian special: handle common encoding error
    'Õ':'ő',

    # ### 匈牙利语 (Hungarian) 专用
    # 'á':'á',
    # 'é':'é',
    # 'í':'í',
    # 'ó':'ó',
    # 'ö':'ö',
    # 'ő':'ő',
    # 'ú':'ú',
    # 'ü':'ü',
    # 'ű':'ű',
}
##doubtable marks: ـ- 
# بتزينها


replace_dict = {}


chars = [
'á', 'é', 'í', 'ó', 'ö', 'ő', 'ú', 'ü', 'ű',
'0','1','2','5','3','4','6','7','8','9',
'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
]
cc=0

ex_sents=[
    '٪',
    '?',
    '٬',
    '|',
    '٫',
    '٭',
    ]
skp=0

file = open(filename, "r")
lines = file.readlines()  # 读取文件中的所有行
file.close()

output_file = None
if is_defined('spmodel'):
    output_file = open(output_filename, "w")

output_file2 = open(output_filename2, "w")
# output_file3 = open(output_filename3, "w")
output_file4 = open(output_filename4, "w")
output_file5 = open(output_filename5, "w")
oov_scp_file = open(oov_scp_filename, "w")
good_scp_file = open(good_scp_filename, "w")

oov_filename = "oov.list"
oov_list_file = open(oov_filename, "w")

modified_lines = []
modified_lines_nopunc = []
modified_lines_danzi = []
mlf_name = None
has_oov = False
for line in lines:
    line = line.strip()  # 去掉行首和行尾的空格
    # print(line);import time; time.sleep(1)
    if line.endswith(".lab\""):
        if mlf_name is not None and skp==0:
            if is_defined('spmodel'):
                output_file.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
                merged_lines = "\n".join(modified_lines)  # 使用join()函数将所有行合并为一个字符串
                output_file.write(merged_lines + "\n.\n")  # 将前面的内容写入到输出文件中          
            ##nopunc
            output_file2.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
            modified_lines = "\n".join(modified_lines_nopunc)  # 使用join()函数将所有行合并为一个字符串
            output_file2.write(modified_lines + "\n.\n")  # 将前面的内容写入到输出文件中                 
            ##danzi
            # output_file3.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
            # modified_lines_danzi = "\n".join(modified_lines_danzi)  # 使用join()函数将所有行合并为一个字符串
            # output_file3.write(modified_lines_danzi + "\n.\n")  # 将前面的内容写入到输出文件中
            ##sent
            modified_lines_sent = " ".join(modified_lines_nopunc)  # 使用join()函数将所有行合并为一个字符串
            output_file4.write(modified_lines_sent + "\n")  # 将前面的内容写入到输出文件中  
            ##scp
            output_file5.write(mlf_name + "\n")  # 将前面的内容写入到输出文件中

            ##check oov scp
            if has_oov is False:
                good_scp_file.write(mlf_name+"\n")
            else:
                has_oov = False  
        modified_lines = []  
        modified_lines_nopunc = []     
        modified_lines_danzi = []  
        content = line[:line.find(".lab\"")]  # 获取前面的内容并去掉开头的引号和空格
        mlf_name = content[1:] if content.startswith("\"") else content  # 去掉开头的引号
        skp=0
    elif line not in excluded_chars:  # 检查当前行是否在排除的字符列表中
        for key, value in diatonic_notes.items():
            line = line.replace(key, value)  # 删除指定字符
        line = line.lower() ##转小写
        for key, value in replace_dict.items():
            if line.strip() == key:
                line = value  # 删除指定字符
        if is_defined('spmodel'):
            spline = spmodel.encode(line)
            # print(line ,spline);import time; time.sleep(0.1)        
            spline = list(map(str, spline))
            modified_lines.extend(spline)
        modified_lines_nopunc.append(line)
        modified_lines_danzi.extend(line)
        
        for char in line.lower():
            if char in ex_sents:
                print('skip: ', line.lower())
                skp=1
            if char not in chars and skp ==0:
                print(cc, line.lower())
                cc+=1
        # # if has_oov is False:
        # #     if line.lower() not in dd:  
        # #         oov_scp_file.write(mlf_name+"\n")
        # #         has_oov = True
        # if line.lower() not in dd:  
        #     if has_oov is False:
        #         oov_scp_file.write(mlf_name+"\n")
        #         has_oov = True
            if line.lower() not in oov_list and skp==0:  
                oov_list.append(line.lower())
                
print(cc)#;exit()
if skp==0:
    if is_defined('spmodel'):
        output_file.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
        merged_lines = "\n".join(modified_lines)  # 使用join()函数将所有行合并为一个字符串
        output_file.write(merged_lines + "\n.\n")  # 将前面的内容写入到输出文件中
        output_file.close()

    output_file2.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
    modified_lines = "\n".join(modified_lines_nopunc)  # 使用join()函数将所有行合并为一个字符串
    output_file2.write(modified_lines + "\n.\n")  # 将前面的内容写入到输出文件中
    output_file2.close()

    # output_file3.write("\"" + mlf_name + ".lab\"\n")  # 将前面的内容写入到输出文件中
    # modified_lines_danzi = "\n".join(modified_lines_danzi)  # 使用join()函数将所有行合并为一个字符串
    # output_file3.write(modified_lines_danzi + "\n.\n")  # 将前面的内容写入到输出文件中
    # output_file3.close()

    ##sent
    modified_lines_sent = " ".join(modified_lines_nopunc)  # 使用join()函数将所有行合并为一个字符串
    output_file4.write(modified_lines_sent + "\n")  # 将前面的内容写入到输出文件中  
    output_file4.close()

    ##scp
    output_file5.write(mlf_name + "\n")  # 将前面的内容写入到输出文件中
    output_file5.close()
    
    ##check oov scp
    if has_oov is False:
        good_scp_file.write(mlf_name+"\n")
    else:
        has_oov = False  

for ele in oov_list:
    oov_list_file.write(ele+"\n")