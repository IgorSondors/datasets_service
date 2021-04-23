import codecs

f = codecs.open( "for_deletion.txt", "r", "utf-8" )

fd = f.readlines()

"""print(len(fd))
print(len(fd[0]))
print((fd[0]))"""

for_deletion = []
for txt_line in fd:
    string_charRecall = txt_line[:-2].split(',')
    for_deletion.append(string_charRecall[0])

print(len(for_deletion))
print(for_deletion[0])

f2 = codecs.open( "ds_base_without_real.csv", "r", encoding = "cp1251" )

fd2 = f2.readlines()

counter = 0
with open("out_of_30k.csv", "w", encoding ="cp1251" ) as out_of_bad:
    for txt_line in fd2:
        img_name = (txt_line.split(',') )[0][:-4]

        if len(for_deletion) > 0:
            if img_name in for_deletion:
                for_deletion.remove(img_name)
                counter = counter + 1
                print(counter)
            else:
                out_of_bad.write(txt_line[:-1])      
        else:
                out_of_bad.write(txt_line[:-1])
