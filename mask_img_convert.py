"""
Converting original run-length encoded file into gray-mask image
The pixels are numbered from top to bottom, then left to right
"""
import os
import os.path as osp
import csv
import cv2
import numpy as np
import scipy.misc



NUM_CLASSES = 4
GRAY_SPAN = 60

def generate_mask(mask_output_root, image, img_fn, anno_list):
    mask_fn = osp.join(mask_output_root, img_fn)
    
    height, width ,chanels = image.shape
    mask_img = np.zeros((height, width), dtype=int)

    for i in range(NUM_CLASSES):
        cls_id = i+1
        gray_value = cls_id*GRAY_SPAN
        anno_str = anno_list[cls_id]        
        if len(anno_str)>0:
            start_length_pairs = list(anno_str.split(' '))
            print('start_length_pairs:', start_length_pairs)
            for j in range(0, len(start_length_pairs), 2):
                start = int(start_length_pairs[j])
                length = int(start_length_pairs[j+1])

                col_idx = int(start/height)-1
                row_idx = start%height-1

                if row_idx+length > height:
                    print("WARNING: convert (",start,length,") -> (", row_idx, col_idx,"), image width:", width, ', height:', height)
                    print("!!!!!!!need to continue the padding")
                    mask_img[row_idx:row_idx+length, col_idx] = gray_value
                    mask_img[:row_idx+length-height, col_idx+1] = gray_value
                else:
                    mask_img[row_idx:row_idx+length, col_idx] = gray_value   

                #assert row_idx+length < height
                #mask_img[row_idx:row_idx+length, col_idx] = gray_value   

                #print('convert (',start,length,") -> (", row_idx, col_idx,"), image width:", width, ', height:', height)

    scipy.misc.toimage(mask_img, cmin=0, cmax=255).save(mask_fn)

    print('Done')

def main(train_images_root, train_anno_file, mask_output_root):
    if not osp.exists(mask_output_root):
        os.mkdir(mask_output_root)

    line_counter = 0 #debug
    with open(train_anno_file, 'r') as train_file:
        train_reader = csv.reader(train_file, delimiter=',')
        cur_img_fn = None
        anno_dict = {}        

        for image_cls_id, anno in train_reader:
            if image_cls_id == "ImageId_ClassId":
                continue

            img_fn = image_cls_id.split('_')[0]
            if cur_img_fn is None:
                cur_img_fn = img_fn
            
            cls_id = int(image_cls_id.split('_')[1])

            anno_dict[cls_id] = anno

            line_counter += 1

            if cls_id == 4:
                image = cv2.imread(osp.join(train_images_root, cur_img_fn))
                generate_mask(mask_output_root, image, cur_img_fn, anno_dict)
                #reset anno & cur_img_fn
                cur_img_fn = None
                anno_dict = {}

        train_file.close()


if __name__ == "__main__":
    train_images_root = './data/train_images'
    mask_output_root = './data/mask_images'
    train_anno_file = './data/train.csv'
    main(train_images_root, train_anno_file, mask_output_root)

