import os
from shutil import copyfile
import codecs

folder_src = r'C:\Users\sWX1017677\Desktop\synthtext\bg_img'
folder_dst = r'C:\Users\sWX1017677\Desktop\bg_imnames\bg'

f = codecs.open( "imnames.cp", "r", "utf-8" )
names = f.readlines()
#print(len(names))
end = '.jpg'
for i in names:
    if i[:-1].endswith(end):
        print(i)
        filename = i[2:-1]
        print(filename)
        copyfile(os.path.join(folder_src, filename), os.path.join(folder_dst, filename))