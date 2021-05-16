import lmdb
import os

import numpy as np
import cv2
import six
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

path = '/home/psmirnov/ds/synth/text_renderer/combined_data/output/rus_300k_lines_for_AA'
path_to_save = '/home/psmirnov/ds/synth/text_renderer/combined_data/output/temp'

path_to_save_imgs = os.path.join(path_to_save, 'images')
path_to_save_txts = os.path.join(path_to_save, 'text')
os.makedirs(path_to_save_imgs, exist_ok=True)
os.makedirs(path_to_save_txts, exist_ok=True)

num_crops_to_save = 200

env = lmdb.open(path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)


with env.begin(write=False) as txn:
    nSamples = int(txn.get('num-samples'.encode()))
    print('Total number of symbols:', nSamples)

    for index in tqdm(range(1, nSamples)):
        if index > num_crops_to_save:
            break

        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key).decode('utf-8')

        char_ann_key = 'chars_ann-%09d'.encode() % index
        char_ann = txn.get(char_ann_key).decode('utf-8')

        print(repr(label))
        print(repr(char_ann))

        image_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(image_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)        

        img_pil = Image.open(buf).convert('RGB')
        img = np.array(img_pil)
        

        out_img_filename = os.path.join(path_to_save_imgs, f"{index}.jpg")
        out_txt_filename = os.path.join(path_to_save_txts, f"{index}.txt")
        cv2.imwrite(out_img_filename, img)

        with open(out_txt_filename, 'w') as out_f:
            out_f.write(label + '\n')
            out_f.write(char_ann)

        # cv2.imshow(label, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()