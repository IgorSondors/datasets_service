import codecs
import os 

f = codecs.open( "ds_images.csv", "r", "utf-8" )

fd = f.readlines()


f1 = next(os.walk('conterclockwise'))
names = f1[2]

with open('ds_images_filtered.csv', 'w', encoding="UTF-8") as ds_images_filtered:
    for i in fd:
        for_remove = False
        for j in names:
            if i.startswith(j):
                for_remove = True
        if not for_remove:
            ds_images_filtered.write(str(i[:-1]))
