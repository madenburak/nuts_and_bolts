import os
import cv2
import shutil
from tqdm import tqdm
import argparse
import json
from utils import *
from collections import defaultdict


class DatasetManager():
    
    def __init__(self, dataset_path, num_of_class, video_path = "", image_dir = "", json_dir = ""):
        self.dataset_path = dataset_path
        self.num_of_class = num_of_class
        self.train_dataset_dir = self.dataset_path + "/train"
        self.val_dataset_dir = self.dataset_path + "/val"
        self.test_dataset_dir = self.dataset_path + "/val"
        self.video_path = video_path
        self.image_path = image_dir
        self.json_dir = json_dir
        
    def convert_yolo_bbox_to_pascal_bbox(self, box, image_size=(640, 640)):
        w = box[2] * image_size[0]
        h = box[3] * image_size[1]
        x1 = ((2 * box[0] * image_size[0]) - w)/2
        y1 = ((2 * box[1] * image_size[1]) - h)/2
        x2 = x1 + w
        y2 = y1 + h
        
        return [int(x1), int(y1), int(x2), int(y2)]

    def mask_specific_label(self, data_type, mask_class):
        
        if data_type == "train":
            label_dir = self.train_dataset_dir + "/labels"
            image_dir = self.train_dataset_dir + "/images"
        
        elif data_type == "val":
            label_dir = self.val_dataset_dir + "/labels"
            image_dir = self.val_dataset_dir + "/images"
        
        else:
            label_dir = self.test_dataset_dir + "/labels"
            image_dir = self.test_dataset_dir + "/images"
            
        if not os.path.exists(self.dataset_path + "/extracted_labels"):
            extracted_labels_dir = os.mkdir(self.dataset_path + "/extracted_labels")
            extracted_images_dir = os.mkdir(self.dataset_path + "/extracted_images")
        else:
            extracted_labels_dir = self.dataset_path + "/extracted_labels"
            extracted_images_dir = self.dataset_path + "/extracted_images"
            
        for file in tqdm(os.listdir(label_dir), desc="Copying masked images and labels"):
            
            file_read = open(label_dir +"/"+ file, 'r')
            data = file_read.readlines()
            
            for index, line in enumerate(data):

                class_id = line.split(" ")[0]
                #print(class_id, file)
                
                if class_id == mask_class: #if bolt, it's labels will delete
                    if not os.path.exists(extracted_labels_dir +"/"+ file.split(".")[0] + "_masked.txt"):
                        shutil.copyfile(label_dir +"/"+ file, extracted_labels_dir +"/"+ file.split(".")[0] + "_masked.txt") 

                    data[index] = ""
                    file_modify_on_copy = open(extracted_labels_dir +"/"+ file.split(".")[0] + "_masked.txt", 'w')
                    file_modify_on_copy.writelines(data)
                    file_modify_on_copy.close()
                    
                    x_norm = float(line.split(" ")[1])
                    y_norm = float(line.split(" ")[2])
                    w_norm = float(line.split(" ")[3])
                    h_norm = float(line.split(" ")[4])
                    coordinates = self.convert_yolo_bbox_to_pascal_bbox([x_norm,y_norm,w_norm,h_norm])
                    #print(coordinates)
                    if not os.path.exists(extracted_images_dir +"/"+ file.split(".")[0] + "_masked.jpg"):
                        shutil.copyfile(image_dir +"/"+ file.split(".")[0] + ".jpg", extracted_images_dir +"/"+ file.split(".")[0] + "_masked.jpg")
                    label_image = cv2.imread(extracted_images_dir +"/"+ file.split(".")[0] + "_masked.jpg")
                    # cv2.imshow("re",label_image)
                    # cv2.waitKey(0)
                    masked_image = cv2.rectangle(label_image,(coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]),
                                color = (90,90,90), thickness=-1)
                    # cv2.imshow("re",masked_image)
                    # cv2.waitKey(0)
                    cv2.imwrite(extracted_images_dir +"/"+ file.split(".")[0] + "_masked.jpg", masked_image)
                    
            file_read.close()
            
        for file in tqdm(os.listdir(extracted_labels_dir), desc="Deleting empty files"):
            if os.stat(extracted_labels_dir +"/"+ file).st_size==0:
                os.remove(extracted_labels_dir +"/"+ file)
                os.remove(extracted_images_dir +"/"+ file.split(".")[0] + ".jpg")
        
    def count_class_labels(self, data_type):
        
        if data_type == "train":
            label_dir = self.train_dataset_dir + "/labels"
        
        elif data_type == "val":
            label_dir = self.val_dataset_dir + "/labels"
        
        else:
            label_dir = self.test_dataset_dir + "/labels"
        
        num_of_class_list = list(range(0, self.num_of_class))
        dict_class = {}
        
        for class_ in num_of_class_list:

            dict_class[class_] = 0 # Each class count 0, initially

        for file in tqdm(os.listdir(label_dir), desc="Counting labels"):

            file_read = open(label_dir +"/"+ file, 'r')
            data = file_read.readlines()
            
            for index, line in enumerate(data):

                class_id = int(line.split(" ")[0])
                dict_class[class_id] += 1
                        
        print(data_type + "_class_dict: ", dict_class)    
    
    
    def video2image(self, fps):

        cap = cv2.VideoCapture(self.video_path)

        curr_frame = 0
        while(True):
            ret, frame = cap.read()
            if not ret: break
            #if curr_frame % fps == 0:
            name = self.image_dir + "/" + str(curr_frame).zfill(4) + ".jpg"
            cv2.imwrite(name, frame)
            curr_frame += 1
        cap.release()
         
            
    def convert_coco_json_to_yolo(self, use_segments=False, cls91to80=False):
        save_dir = make_dirs()  # output directory
        coco80 = coco91_to_coco80_class()

        # Import json
        for json_file in sorted(Path(self.json_dir).resolve().glob('*.json')):
            fn = Path(save_dir) / 'labels' / json_file.stem.replace('instances_', '')  # folder name
            fn.mkdir()
            with open(json_file) as f:
                data = json.load(f)

            # Create image dict
            images = {'%g' % x['id']: x for x in data['images']}
            # Create image-annotations dict
            imgToAnns = defaultdict(list)
            for ann in data['annotations']:
                imgToAnns[ann['image_id']].append(ann)

            # Write labels file
            for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
                img = images['%g' % img_id]
                h, w, f = img['height'], img['width'], img['file_name']

                bboxes = []
                segments = []
                for ann in anns:
                    if ann['iscrowd']:
                        continue
                    # The COCO box format is [top left x, top left y, width, height]
                    box = np.array(ann['bbox'], dtype=np.float64)
                    box[:2] += box[2:] / 2  # xy top-left corner to center
                    box[[0, 2]] /= w  # normalize x
                    box[[1, 3]] /= h  # normalize y
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue

                    cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1  # class
                    #cls = coco80[ann['category_id']] if cls91to80 else ann['category_id'] # class
                    track_id = ann['track_id']
                    box = [cls] + box.tolist()# + [track_id]
                    if box not in bboxes:
                        bboxes.append(box)
                    # Segments
                    if use_segments:
                        if len(ann['segmentation']) > 1:
                            s = merge_multi_segment(ann['segmentation'])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        if s not in segments:
                            segments.append(s)

                # Write
                with open((fn / f).with_suffix('.txt'), 'a') as file:
                    for i in range(len(bboxes)):
                        line = *(segments[i] if use_segments else bboxes[i]),  # cls, box or segments
                        file.write(('%g ' * len(line)).rstrip() % line + '\n')       
            
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="D:/Stroma/labels_new/yololabels_no_track_id_with_mask", 
                        help = "dataset path")
    parser.add_argument("--number-class", "-n", default=2, type=int, help = "number of class")
    parser.add_argument("--video-path", default = " ", help = "video file that will be convert to image")
    parser.add_argument("--image-dir", default = " ", help = "image directory that store extracted image")
    parser.add_argument("--json-dir", default = " ", default="D:/Stroma/labels_new/annotations/", 
                        help = "json directory thath will be convert to yolo")
    
    args = parser.parse_args()
    
    manage_dataset = DatasetManager(args.dataset_dir, args.number_class, args.video_path, args.image_dir, args.json_dir)
    #mask_specific_label("train", 0)
    #mask_specific_label("val", 0)
    #manage_dataset.convert_coco_json_to_yolo(use_segments=False, cls91to80=False)
    train_count_labels = manage_dataset.count_class_labels("train")
    val_count_labels = manage_dataset.count_class_labels("val")


if __name__ == "__main__":
    main()