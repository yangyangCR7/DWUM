import os
import torch
import numpy as np
import pandas as pd
import PIL.Image
import scipy.io
import torchvision.transforms as transforms
from torch.utils.data import Dataset, BatchSampler


class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        im = PIL.Image.merge('RGB', [b, g, r])
        return im

class MetricDataset(Dataset):
    def __init__(self, data_path, model, is_train=True, size_crops=(224, 224), scale=(0.08, 1)):
        super(MetricDataset, self).__init__()
        
        self.data_path = data_path  
        self.is_train = is_train
        
        self.samples = [] 
        self.ys = []        
        self.true_ys = []   
        
        if 'CUB' in self.data_path.upper():
            idx = self.data_path.upper().find('CUB')
            self.clean_root = self.data_path[:idx + 3]
            self._load_cub200()
        elif 'CARS196' in self.data_path.upper():
            idx = self.data_path.upper().find('CARS196')
            self.clean_root = self.data_path[:idx + 7]
            self._load_cars196()
        elif 'SOP' in self.data_path.upper():
            idx = self.data_path.upper().find('SOP')
            self.clean_root = self.data_path[:idx + 3]
            self._load_sop()
        else:
            raise ValueError(f"Unrecognized dataset in path: {self.data_path}")

        self.classes = list(sorted(set(self.ys)))
        self.label_map = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(self.classes)}
        
        self.ys = [self.label_map[y] for y in self.ys]
        self.true_ys = [self.label_map.get(ty, ty) for ty in self.true_ys] 
        
        self.samples = [(p, self.label_map[y], ty) for (p, y), ty in zip(self.samples, self.true_ys)]
        self.classes = list(sorted(set(self.ys)))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if is_train:
            if model == "BNInception":
                self.trans = transforms.Compose([
                    RGBToBGR(),
                    transforms.RandomResizedCrop(size_crops, scale),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4078, 0.4588, 0.502], std=[0.0039, 0.0039, 0.0039]),
                ])
            else:
                self.trans = transforms.Compose([
                    transforms.RandomResizedCrop(size_crops, scale),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
        else:
            if model == "BNInception":
                self.trans = transforms.Compose([   
                    RGBToBGR(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4078, 0.4588, 0.502], std=[0.0039, 0.0039, 0.0039]),                  
                ])
            else:
                self.trans = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

    def _load_cub200(self):
        images_txt = os.path.join(self.clean_root, 'images.txt')
        images_dir = os.path.join(self.clean_root, 'images')

        noisy_labels_txt = os.path.join(self.data_path, 'image_class_labels.txt')
        clean_labels_txt = os.path.join(self.clean_root, 'image_class_labels.txt') 
        
        img_paths = {}
        with open(images_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_paths[parts[0]] = parts[1].replace('\\', '/')
 
        true_labels_dict = {}
        if os.path.exists(clean_labels_txt):
            with open(clean_labels_txt, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    true_labels_dict[parts[0]] = int(parts[1])
                    
        with open(noisy_labels_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_id = parts[0]
                noisy_label = int(parts[1])
                true_label = true_labels_dict.get(img_id, noisy_label) 
          
                if self.is_train and noisy_label <= 100:
                    self.samples.append((os.path.join(images_dir, img_paths[img_id]), noisy_label))
                    self.ys.append(noisy_label)
                    self.true_ys.append(true_label)
                elif not self.is_train and noisy_label > 100:
                    self.samples.append((os.path.join(images_dir, img_paths[img_id]), noisy_label))
                    self.ys.append(noisy_label)
                    self.true_ys.append(true_label)

    def _load_cars196(self):
        noisy_mat_path = os.path.join(self.data_path, 'cars_annos.mat')
        clean_mat_path = os.path.join(self.clean_root, 'cars_annos.mat')

        true_labels_dict = {}
        if os.path.exists(clean_mat_path):
            clean_data = scipy.io.loadmat(clean_mat_path)
            clean_annotations = clean_data['annotations']
            for i in range(clean_annotations.shape[1]):
                name = str(clean_annotations[0, i][0])[2:-2]
                clean_label = int(clean_annotations[0, i][5])
                true_labels_dict[name] = clean_label

        data = scipy.io.loadmat(noisy_mat_path)
        annotations = data['annotations']
        
        for i in range(annotations.shape[1]):
            name = str(annotations[0, i][0])[2:-2]
            noisy_label = int(annotations[0, i][5])
            true_label = true_labels_dict.get(name, noisy_label)
            image_path = os.path.join(self.clean_root, name)
 
            if self.is_train and noisy_label <= 98:
                self.samples.append((image_path, noisy_label))
                self.ys.append(noisy_label)
                self.true_ys.append(true_label)
            elif not self.is_train and noisy_label > 98:
                self.samples.append((image_path, noisy_label))
                self.ys.append(noisy_label)
                self.true_ys.append(true_label)

    def _load_sop(self):
        txt_file = 'Ebay_train.txt' if self.is_train else 'Ebay_test.txt'
        noisy_txt_path = os.path.join(self.data_path, txt_file)
        clean_txt_path = os.path.join(self.clean_root, txt_file)

        true_labels_dict = {}
        if os.path.exists(clean_txt_path):
            with open(clean_txt_path, 'r') as f:
                lines = f.readlines()[1:] 
                for line in lines:
                    parts = line.strip().split()
                    clean_label = int(parts[1])
                    rel_path = parts[3].replace('\\', '/')
                    true_labels_dict[rel_path] = clean_label

        with open(noisy_txt_path, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split()
                noisy_label = int(parts[1])
                linux_rel_path = parts[3].replace('\\', '/')
                true_label = true_labels_dict.get(linux_rel_path, noisy_label)
                img_path = os.path.join(self.clean_root, linux_rel_path)
                
                self.samples.append((img_path, noisy_label))
                self.ys.append(noisy_label)
                self.true_ys.append(true_label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, noisy_label, true_label = self.samples[index]
        image = PIL.Image.open(path).convert('RGB')
        
        if hasattr(self, 'trans') and self.trans is not None:
            image = self.trans(image)
            
        return image, noisy_label, true_label

    @property
    def targets(self):
        return self.ys

class Inshop_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/IN_SHOP'
        self.mode = mode
        self.transform = transform
        self.train_ys, self.train_im_paths = [], []
        self.query_ys, self.query_im_paths = [], []
        self.gallery_ys, self.gallery_im_paths = [], []
                    
        data_info = np.array(pd.read_table(self.root +'/list_eval_partition.txt', header=1, delim_whitespace=True))[:,:]
        train, query, gallery = data_info[data_info[:,2]=='train'][:,:2], data_info[data_info[:,2]=='query'][:,:2], data_info[data_info[:,2]=='gallery'][:,:2]

        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
        train[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:,1]])

        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:,1], gallery[:,1]])])))}
        query[:,1]   = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
        gallery[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])

        for img_path, key in train:
            self.train_im_paths.append(os.path.join(self.root, img_path))
            self.train_ys += [int(key)]
        for img_path, key in query:
            self.query_im_paths.append(os.path.join(self.root, img_path))
            self.query_ys += [int(key)]
        for img_path, key in gallery:
            self.gallery_im_paths.append(os.path.join(self.root, img_path))
            self.gallery_ys += [int(key)]
            
        if self.mode == 'train':
            self.im_paths = self.train_im_paths
            self.ys = self.train_ys
        elif self.mode == 'query':
            self.im_paths = self.query_im_paths
            self.ys = self.query_ys
        elif self.mode == 'gallery':
            self.im_paths = self.gallery_im_paths
            self.ys = self.gallery_ys

    def nb_classes(self):
        return len(set(self.ys))
            
    def __len__(self):
        return len(self.ys)
            
    def __getitem__(self, index):
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            if len(list(im.split())) == 1 : im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im
        
        im = img_load(index)
        target = self.ys[index]
        return im, target
    
    @property
    def targets(self):
        return self.ys
    
    @property
    def classes(self):
        return list(sorted(set(self.ys)))


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples, partial_rate, imbalance=False, gamma=100):
        self.labels = labels
        self.labels_set = np.array(list(sorted(set(labels))))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.label_to_indices = {k: v[:int(len(v)*partial_rate)]
                                 for k, v in self.label_to_indices.items()}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.imbalance = imbalance
        if self.imbalance:
            self.N1 = min([v.size for k,v in self.label_to_indices.items()])
            self.gamma = gamma
            self.Nk = [int(self.N1*pow(gamma,-(k)/(len(self.labels_set)-1))) if int(self.N1*pow(gamma,-(k)/(len(self.labels_set)-1)))>0 else 1 for k in self.labels_set]
            for k,v in self.label_to_indices.items():
                self.label_to_indices[k] = v[:self.Nk[k]]
            self.n_dataset = int(sum([v.size for k,v in self.label_to_indices.items()])) 
        else:    
            self.n_dataset = int(len(self.labels)*partial_rate) 
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            if self.imbalance:
                classes = np.random.choice(self.labels_set, self.n_classes, replace=False, p=(np.array(self.Nk)/self.n_dataset).ravel())
            else:
                classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            
            for class_ in classes:
                if self.used_label_indices_count[class_]+self.n_samples>len(self.label_to_indices[class_]):
                    for i in range(self.n_samples):
                        chose = np.random.choice(len(self.label_to_indices[class_]), 1 , replace=False)
                        indices.extend(self.label_to_indices[class_][chose])
                else:
                    indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                                 class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size