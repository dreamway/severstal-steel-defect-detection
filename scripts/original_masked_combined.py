import os
import os.path as osp
import cv2
import csv


def generate_combined_image(image_path, mask_path, combined_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    combined = cv2.addWeighted(image, 0.7, mask, 0.3, 0.01)
    cv2.imwrite(combined_path, combined)


def main(train_images_root, train_anno_file, mask_images_root, combined_image_root):
    line_counter = 0
    with open(train_anno_file, 'r') as train_file:
        train_reader = csv.reader(train_file, delimiter=',')
        generate_flag = False

        for image_cls_id, anno in train_reader:
            if image_cls_id == "ImageId_ClassId":
                continue
            
            img_fn = image_cls_id.split('_')[0]
            cls_id = int(image_cls_id.split('_')[1])

            if len(anno) > 0: # have mask
                generate_flag = True
            
            if cls_id == 4 and generate_flag is True:
                image_path = osp.join(train_images_root, img_fn)
                mask_path = osp.join(mask_images_root, img_fn)
                combined_path = osp.join(combined_image_root, img_fn)
                print("need to generate combined:", image_path)
                generate_combined_image(image_path, mask_path, combined_path)
                generate_flag = False

            ''' #debug
            line_counter += 1
            if line_counter == 8:
                break
            '''

if __name__ == "__main__":
    train_images_root = './data/train_images'
    mask_images_root = './data/mask_images'
    train_anno_file = './data/train.csv'
    combined_image_root = './data/combined_images'
    if not osp.exists(combined_image_root):
        os.mkdir(combined_image_root)

    main(train_images_root, train_anno_file, mask_images_root, combined_image_root)