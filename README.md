# DWUM: Dimension-Wise Uncertainty Modeling for Robust Deep Metric Learning with Noisy Labels (DWUM)

It offers source code for replicating the experiments conducted on three benchmark datasets (CUB200, Cars196, SOP).

## Requirements
+ Python 3.8
+ PyTorch 1.8.1+cu111
+ numpy
+ tqdm
+ tensorboardX
+ scikit-learn
+ scipy

## Training and Evaluation

1. Training
You can train the DWUM model using the provided configuration `.yml` files. Here is an example of running the training script on the CUB200 dataset:

CUDA_VISIBLE_DEVICES=0 python train.py --cfg scripts/cfgs/RES_PAN_CUB200.yml

## Datasets
2. Download four benchmark datasets.
    + [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158)
    + [Cars196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
    + [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)
    + [InShop Clothes Retrieval](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
Extract the tgz or zip file into `./Dataset` folder.

3. The data folder is constructed as followed:
```
Place them in `Dataset` dir as follows:
```
Dataset/
├── Cars196/
│   ├── Symmetric/
│   │   ├── 0.1_Symmetric/
│   │   │   ├── car_ims/
│   │   │   └── cars_annos.mat
│   │   ├── ...
│   │   └── 0.9_Symmetric/
│   └── SmallCluster/
├── CUB/
│   ├── Symmetric/
│   │   ├── 0.1_Symmetric/
│   │   │   ├── images/
│   │   │   │   ├── 001.Black_footed_Albatross/
│   │   │   │   ├── ...
│   │   │   │   └── 200.Common_Yellowthroat/
│   │   │   ├── image_class_labels.txt
│   │   │   └── images.txt
│   │   ├── ...
│   │   └── 0.9_Symmetric/
│   └── SmallCluster/
└── SOP/
    ├── Symmetric/
    │   ├── 0.1_Symmetric/
    │   │   ├── bicycle_final/
    │   │   ├── ...
    │   │   ├── toaster_final/
    │   │   ├── Ebay_train.txt
    │   │   └── Ebay_test.txt
    │   ├── ...
    │   └── 0.9_Symmetric/
    └── SmallCluster/
```