import os
folder1 = 'txts'

f1 = next(os.walk(folder1))
names1 = f1[2]
start_for_remove = 'our_real'
for i in names1:
    if i.startswith(start_for_remove):
        print(i)
        os.remove(os.path.join(folder1, i))