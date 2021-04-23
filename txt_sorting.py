import codecs

f = codecs.open( "string_charRecall_1.6kk.txt", "r", "utf-8" )

fd = f.readlines()

print(len(fd))
print(len(fd[0]))
print((fd[0]))

list_string_charRecall = []
list_string = []
list_charRecall = []
for txt_line in fd:
    string_charRecall = txt_line[:-2].split(',')
    list_string.append(string_charRecall[0])
    list_charRecall.append(float(string_charRecall[1]))
    #list_string_charRecall.append(string_charRecall)

#print(len(list_string_charRecall))
#print(len(list_string_charRecall[0]))
#print((list_string_charRecall[0]))

list_charRecall, list_string = zip(*sorted(zip(list_charRecall, list_string)))

with open('string_charRecall_sorted.txt', 'w', encoding="UTF-8") as string_charRecall_sorted:
    for i in range(len(list_charRecall)):
        string_charRecall_sorted.write(str(list_string[i]) + ',' + str(list_charRecall[i]) + '\n')
