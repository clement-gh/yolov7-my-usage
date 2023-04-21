
import json
from PIL import Image
import tqdm
from tqdm import tqdm
import os
from zipfile import ZipFile

def json_to_yolo( parent_dir, json_file):
    
    if not os.path.exists(json_file):
        raise Exception(json_file + " was not found" )
        

    # image directory
    img_dir = "images/"+ parent_dir
    
    if not os.path.exists(img_dir):
        exception = "Image directory does not exist"
        raise Exception(exception)
        
        
    # label directory
    
    
    
    label_dir = "labels/"+ parent_dir
    # delete the content of the directory if it already exists
    
    if not os.path.exists(label_dir):        
        os.makedirs(label_dir)
   
    # Open JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Iteration for each entry in JSON file to create label file for each image
        for entry in tqdm(data):
            # get image name and bounding box
            img_name = entry['filename']
            bbox = entry['bbox']

            # load image to get width and height
            img_path = os.path.join(img_dir, img_name)
            with Image.open(img_path) as img:
                width, height = img.size

            # label file name
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(label_dir, label_name)

            # Ouverture du fichier de label
            with open(label_path, 'w') as label:
                
                # Itération sur les objets dans l'image
                for i in range(len(bbox['label'])):
                    
                
                    # get coordinates of bounding box and label
                    # set label, x_min, y_min, x_max, y_max 
                    
                    l = bbox['label'][i]
                    x_min = bbox['left'][i] 
                    y_min = bbox['top'][i]
                    x_max = x_min + bbox['width'][i]
                    y_max = y_min + bbox['height'][i]

                    # Normalisation of coordinates
                    # calculating the center of the box
                    x_center = (x_min + x_max) / (2 * width) 
                    y_center = (y_min + y_max) / (2 * height)
                    # calculating the width and height of the box
                    w = bbox['width'][i] / width
                    h = bbox['height'][i] / height

                    # Writing label and coordinates in label file
                    
                    label.write(f'{l} {x_center} {y_center} {w} {h}\n')



def create_architecture_files( train_img_path = "images/train", val_img_path= "images/test"):
    
        #Training 
    with open('train.txt', "w") as f:
        img_list = os.listdir(train_img_path)
        for img in img_list:
            f.write(train_img_path+'/'+img+'\n')
    # Validation 
    with open('test.txt', "w") as f:
        img_list = os.listdir(val_img_path)
        for img in img_list:
            f.write(val_img_path+'/'+img+'\n')
            
def unzip(file_name):
    if not os.path.exists(file_name):
        raise Exception(file_name + " was not found" )
    with ZipFile(file_name, 'r') as zipObj:
        zipObj.extractall()

def main():
    if not os.path.exists("images"):
        unzip("images.zip")
    
    json_to_yolo("train", "train.json")
    json_to_yolo("test", "test.json")
    create_architecture_files()
    
if __name__ == '__main__':
    main()