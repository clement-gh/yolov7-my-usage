import h5py
import json
import tqdm
from tqdm import tqdm
import os
from zipfile import ZipFile


def get_bbox(mat_file, index):
    bbox = {}
    item = mat_file['digitStruct']['bbox'][index].item()
    bbox['label'] = []
    bbox['left'] = []
    bbox['top'] = []
    bbox['width'] = []
    bbox['height'] = []
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = mat_file[item][key]
        if attr.shape[0] == 1:
            bbox[key] = [int(attr[0][0])]
        else:
            bbox[key] = [int(mat_file[attr[j].item()][0][0]) for j in range(attr.shape[0])]
    return bbox

def mat_to_json(mat_file, output_file, name):
    data = []
    for i in tqdm(range(len(mat_file['digitStruct']['bbox']))):
        filename = ''.join([chr(c[0]) for c in mat_file[mat_file['digitStruct']['name'][i][0]]])
        bbox = get_bbox(mat_file, i)
        entry = {'filename': name+'_'+filename, 'bbox': bbox}
        data.append(entry)
    with open(output_file, 'w') as f:
        json.dump(data, f)
        
def unzip():
    # unzip mat file
    if not os.path.exists('./mat'):
        with ZipFile('mat.zip', 'r') as zipObj:
        # Extract all the contents of zip file in current directory
            zipObj.extractall()
    else:
        print('mat folder already exists')
          
        
def main ():
    unzip()
    
    output_file = './train.json'
    output_file2 = './test.json'
    mat_file = h5py.File('./mat/train.mat', 'r') 
    mat_file2 = h5py.File('./mat/test.mat', 'r')
    
    print('creating train json')
    mat_to_json(mat_file, output_file, 'train')
    mat_file.close()
    print('json created')
    
    print('creating test json')
    mat_to_json(mat_file2, output_file2, 'test')
    mat_file2.close()
    print('json created')
    
if __name__ == '__main__':
    main()
    