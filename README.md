
# "Improving deep metric learning via self-distillation and online batch diffusion process (OBD-SD)" in PyTorch

---
This repository contains all code and implementations used in:

```
Improving deep metric learning via self-distillation and online batch diffusion process (OBD-SD)
```
Paper Link: https://link.springer.com/article/10.1007/s44267-024-00051-0

## Some Notes
If you use this code in your research, please cite:
```
@article{zeng2022self,
  title={Self-distillation with Online Diffusion on Batch Manifolds Improves Deep Metric Learning},
  author={Zeng, Zelong and Yang, Fan and Liu, Hong and Satoh, Shin'ichi},
  journal={arXiv preprint arXiv:2211.07566},
  year={2022}
}
```
This repository contains (in parts) code that has been adapted from: 
- https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch
- https://github.com/fyang93/diffusion

---
## Usage
### Requirements

* PyTorch 1.6.0+ & Faiss-Gpu
* Python 3.6+
* pretrainedmodels, torchvision 0.3.0+

You can run:
```
(1) conda create -n DL python=3.6
(2) conda activate DL
(3) conda install matplotlib scipy scikit-learn scikit-image tqdm pandas pillow
(4) conda install pytorch torchvision faiss-gpu cudatoolkit=10.1 -c pytorch
(5) pip install wandb pretrainedmodels
```

### Dataset Preparation:
We use three datasets: 

- CUB200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200.html)
- CARS196 (https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- Stanford Online Products (http://cvgl.stanford.edu/projects/lifted_struct/)

You can be downloaded either from the respective project sites or directly via Dropbox (provided by [here](https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch)):

-   CUB200-2011 (1.08 GB):  [https://www.dropbox.com/s/tjhf7fbxw5f9u0q/cub200.tar?dl=0](https://www.dropbox.com/s/tjhf7fbxw5f9u0q/cub200.tar?dl=0)
-   CARS196 (1.86 GB):  [https://www.dropbox.com/s/zi2o92hzqekbmef/cars196.tar?dl=0](https://www.dropbox.com/s/zi2o92hzqekbmef/cars196.tar?dl=0)
-   SOP (2.84 GB):  [https://www.dropbox.com/s/fu8dgxulf10hns9/online_products.tar?dl=0](https://www.dropbox.com/s/fu8dgxulf10hns9/online_products.tar?dl=0)

**The latter ensures that the folder structure is already consistent with this pipeline and the dataloaders**.

After unzip, please make sure the data fold structure will look like below:

* For CUB200-2011/CARS196:
```
cub200/cars196
└───images
|    └───001.Black_footed_Albatross
|           │   Black_Footed_Albatross_0001_796111
|           │   ...
|    ...
```

* For Stanford Online Products:
```
online_products
└───images
|    └───bicycle_final
|           │   111085122871_0.jpg
|    ...
|
└───Info_Files
|    │   bicycle.txt
|    │   ...
```
### Installation
Before training, you need to install this project from github: 
```
git clone https://github.com/ZelongZeng/OBD-SD_Pytorch.git
cd OBD-SD_Pytorch
mkdir Training_Results
cd Training_Results
mkdir cub200
mkdir cars196
mkdir online_products
cd ..
```

### Training:

If you want to run baseline:

```
python main.py --dataset cub200 --kernels 6 --source $datapath/cub200 --n_epochs 150 \
--log_online --project OBD_SD --group CUB_MS \
--seed 0 --gpu 0 --bs 112 --samples_per_class 2 --loss multisimilarity_obd_sd \
--arch resnet50_normalize_frozen
```

The purpose of each flag explained:

* `--loss <loss_name>`: Name of the training objective used. See folder `criteria` for implementations of these methods.
* `--log_online`: Log metrics online via W&B (Default). You need to fill your W&B key in the `paramters.py`. If you don't want to use W&B, please remove it. 
* `--project`: Project name in W&B.
* `--group`: The local and online savename.
* `--seed`, `--gpu`, `--source`: Training seed, the used GPU and the path to the parent folder containing the respective Datasets.
* `--arch`: The utilized backbone, e.g. ResNet50. You can append `_frozen`, `_normalize` and `double` to freeze BN layers, normalize embeddings and use double pooling.
* `--samples_per_class`: The number of samples for each class in the mini-batch.
* `--lr`, `--n_epochs`, `--bs` ,`--embed_dim`: Learning rate, number of training epochs, the batchsize and the embedding dimensionality.  

Please see more information in `parameters.py`.
You can also do the training/validation split for training, set `--use_tv_split` and `--tv_split_perc <train/val split percentage>`.


If you want to run baseline with our OBD-SD:

```
python main.py --dataset cub200 --kernels 6 --source $datapath/cub200 --n_epochs 150 \
--log_online --project OBD_SD --group CUB_MS_obdsd_e150_lamda-1000.0_d-w0.3 \
--kd_lambda 1000.0 --diffusion_w 0.3 --seed 0 --gpu 0 --bs 112 \
--samples_per_class 2 --loss multisimilarity_obd_sd --arch resnet50_normalize_frozen\
--PSD --OBDP
```

* `--PSD`: If set, use PSD in our paper.
* `--PSD`, `--OBD-SD`: If both set, use OBD-SD in our paper.

* You can see more samples runs in `Sample_Runs/SampleRuns.sh`


### Metrics results & Pre-trained models

If you use W&B, all metrics results will be recored on it, you can check the all results from it. 

We also share some pre-trained models of our experiments.  All models are trainied with MS loss. 
You can download the pre-trained models from this Dropbox link: https://www.dropbox.com/sh/zjnzv5v5m68v6e8/AABP3TFV6YnaKGC3azFCTmSWa?dl=0

After unzip, you will get two folders namely `NMI` and `Recall`.  Please put these two folders into the folder `OBD-SD_Pytorch/Training_Results/`. 
Then, if you want to test the results of applying only `PSD`, you can run: 

- CUB200-2011:
```
python Test_pretrained_models.py --dataset cub200 --source $datapath/cub200 \
--arch resnet50_normalize_frozen --embed_dim 128 --PSD
```
- CARS196:
```
python Test_pretrained_models.py --dataset cars196 --source $datapath/cars196 \
--arch resnet50_normalize_frozen --embed_dim 128 --PSD
```
- SOP:
```
python Test_pretrained_models.py --dataset online_products --source $datapath/online_products \
--arch resnet50_normalize_frozen --embed_dim 128 --PSD
```
If you want to test the results of applying both `PSD` and `OBDP`, you can run: 

- CUB200-2011:
```
python Test_pretrained_models.py --dataset cub200 --source $datapath/cub200 \
--arch resnet50_normalize_frozen_double --embed_dim 128 --PSD --OBDP
```
- CARS196:
```
python Test_pretrained_models.py --dataset cars196 --source $datapath/cars196 \
--arch resnet50_normalize_frozen_double --embed_dim 128 --PSD --OBDP
```
- SOP:
```
python Test_pretrained_models.py --dataset online_products --source $datapath/online_products \
--arch resnet50_normalize_frozen_double --embed_dim 128 --PSD --OBDP
```
For the results using both `PSD` and `OBDP`, we provide models trained in two different feature dimensions, *i.e.*, 128-dim and 512-dim. You can see the different results by modifying the flag `--embed_dim` ( *i.e.*, using `--embed_dim 128` or `--embed_dim 512`).


## Citations

If you use this project in your research, please cite

```
@article{zeng2024improving,
  title={Improving deep metric learning via self-distillation and online batch diffusion process},
  author={Zeng, Zelong and Yang, Fan and Liu, Hong and Satoh, Shin’ichi},
  journal={Visual Intelligence},
  volume={2},
  number={1},
  pages={1--13},
  year={2024},
  publisher={Springer}
}
```
