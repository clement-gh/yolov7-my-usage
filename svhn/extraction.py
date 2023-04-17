import os
import scipy.io as sio
import cv2

# Chemin d'accès aux fichiers MatLab SVHN
svhn_dir = './'

# Chemin d'accès au dossier de sortie pour les images et les annotations
out_dir = './'

# Créer les dossiers de sortie si nécessaire
os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)

# Charger les données SVHN
train_data = sio.loadmat(os.path.join(svhn_dir, 'train_32x32.mat'))
test_data = sio.loadmat(os.path.join(svhn_dir, 'test_32x32.mat'))

# Créer les fichiers image et de texte contenant les annotations pour les images de formation
train_images = train_data['X']
train_labels = train_data['y']
train_file = open(os.path.join(out_dir, 'train.txt'), 'w')
for i in range(train_images.shape[3]):
    img = train_images[:, :, :, i]
    label = train_labels[i][0] if train_labels[i][0] != 10 else 0
    img_file = os.path.join(out_dir, 'images', f'train_{i+1}.jpg')
    label_file = os.path.join(out_dir, 'labels', f'train_{i+1}.txt')
    cv2.imwrite(img_file, img)
    with open(label_file, 'w') as f:
        f.write(f'{label} 0.5 0.5 1 1\n')
    train_file.write(f'{img_file}\n')
train_file.close()

# Créer les fichiers image et de texte contenant les annotations pour les images de test
test_images = test_data['X']
test_labels = test_data['y']
test_file = open(os.path.join(out_dir, 'test.txt'), 'w')
for i in range(test_images.shape[3]):
    img = test_images[:, :, :, i]
    label = test_labels[i][0] if test_labels[i][0] != 10 else 0
    img_file = os.path.join(out_dir, 'images', f'test_{i+1}.jpg')
    label_file = os.path.join(out_dir, 'labels', f'test_{i+1}.txt')
    cv2.imwrite(img_file, img)
    with open(label_file, 'w') as f:
        f.write(f'{label} 0.5 0.5 1 1\n')
    test_file.write(f'{img_file}\n')
test_file.close()
