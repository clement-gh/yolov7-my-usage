import os
import scipy.io as sio
import cv2
import tqdm
from tqdm import tqdm



svhn_dir = './'

out_dir = './'
train_data = sio.loadmat(os.path.join(svhn_dir, 'train_32x32.mat'))
test_data = sio.loadmat(os.path.join(svhn_dir, 'test_32x32.mat'))

def make_dir():
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
    



def write_data(data, out_dir, split):
    
    images = data['X']
    labels = data['y']
    file = open(os.path.join(out_dir, f'{split}.txt'), 'w')
    for i in tqdm(range(images.shape[3])):
       
        #img  is a 32x32x3 array of uint8 where 3 is the RGB channels 
        # and i is the image number in the dataset
        img = images[:, :, :, i]
        
        # if the label is 10, change it to 0 (because 10 is not a digit)
        label = labels[i][0] if labels[i][0] != 10 else 0
        
        #naming the image and the label with the split and the image number
        img_file = os.path.join(out_dir, 'images', f'{split}_{i+1}.jpg')
        label_file = os.path.join(out_dir, 'labels', f'{split}_{i+1}.txt')
        
        #saving the image and the label
        cv2.imwrite(img_file , img)
        with open(label_file, 'w') as f:
            # the label is written in the format: label x_center y_center width height 
            f.write(f'{label} 0.5 0.5 1 1 \n')
        file.write(f'{img_file} \n')
    #closing the file that contains the path to the images    
    file.close()
    
    
def main():
    make_dir()
    write_data(train_data, out_dir, 'train')
    write_data(test_data, out_dir, 'test')
    
if __name__ == '__main__':
    main()
    
