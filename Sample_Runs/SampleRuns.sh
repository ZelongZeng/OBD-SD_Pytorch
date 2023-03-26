#-------------------
#   CUB200
#-------------------
### ResNet50, dim = 128
### MS baseline
python main.py --dataset cub200 --kernels 6 --source $datapath/cub200 --n_epochs 150 \
--log_online --project OBD_SD --group CUB_MS \
--seed 0 --gpu 0 --bs 112 --samples_per_class 2 --loss multisimilarity_obd_sd \
--arch resnet50_normalize_frozen

### MS+OBD-SD
python main.py --dataset cub200 --kernels 6 --source $datapath/cub200 --n_epochs 150 \
--log_online --project OBD_SD --group CUB_MS_obdsd_e150_lamda-1000.0_d-w0.3 \
--kd_lambda 1000.0 --diffusion_w 0.3 --seed 0 --gpu 0 --bs 112 \
--samples_per_class 2 --loss multisimilarity_obd_sd --arch resnet50_normalize_frozen --PSD --OBDP

### ResNet50, dim = 512
### MS+OBD-SD | DoublePool
python main.py --dataset cub200 --kernels 6 --source $datapath/cub200 --n_epochs 150 \
--log_online --project OBD_SD --group CUB_MS_Resnet-512-double_obdsd_e150_lamda-1000.0_d-w0.3 \
--kd_lambda 1000.0 --diffusion_w 0.3 --seed 0 --gpu 0 --bs 112 \
--samples_per_class 2 --loss multisimilarity_obd_sd --arch resnet50_normalize_frozen_double --PSD --OBDP --embed_dim 512 \

#-------------------
#   CARS196
#-------------------
### ResNet50, dim = 128
### MS baseline
python main.py --dataset cars196 --kernels 6 --source $datapath/cars196 --n_epochs 150 \
--log_online --project OBD_SD --group CARS_MS \
--seed 0 --gpu 0 --bs 112 --samples_per_class 2 --loss multisimilarity_obd_sd \
--arch resnet50_normalize_frozen

### MS+OBD-SD
python main.py --dataset cars196 --kernels 6 --source $datapath/cars196 --n_epochs 150 \
--log_online --project OBD_SD --group CARS_MS_obdsd_e150_lamda-75.0_d-w0.99 \
--kd_lambda 75.0 --diffusion_w 0.99 --seed 0 --gpu 0 --bs 112 \
--samples_per_class 2 --loss multisimilarity_obd_sd --arch resnet50_normalize_frozen --PSD --OBDP

### ResNet50, dim = 512
### MS+OBD-SD | DoublePool
python main.py --dataset cars196 --kernels 6 --source $datapath/cars196 --n_epochs 300 \
--log_online --project OBD_SD --group CARS_MS_Resnet-512-double_obdsd_e300_lamda-200.0_d-w0.99 \
--kd_lambda 200.0 --diffusion_w 0.99 --seed 0 --gpu 0 --bs 112 \
--samples_per_class 2 --loss multisimilarity_obd_sd --arch resnet50_normalize_frozen_double --PSD --OBDP --embed_dim 512 \
--tau 200 250 --gamma 0.3


#-------------------
#   SOP
#-------------------
### ResNet50, dim = 128
### MS baseline
python main.py --dataset online_products --kernels 6 --source $datapath/online_products --n_epochs 100 \
--log_online --project OBD_SD --group SOP_MS \
--seed 0 --gpu 0 --bs 112 --samples_per_class 2 --loss multisimilarity_obd_sd \
--arch resnet50_normalize_frozen

### MS+OBD-SD
python main.py --dataset online_products --kernels 6 --source $datapath/online_products --n_epochs 100 \
--log_online --project OBD_SD --group SOP_MS_obdsd_e150_lamda-100.0_d-w0.5 \
--kd_lambda 100.0 --diffusion_w 0.5 --diffusion_tau 0.005 --seed 0 --gpu 0 --bs 112 \
--samples_per_class 2 --loss multisimilarity_obd_sd --arch resnet50_normalize_frozen --PSD --OBDP

### ResNet50, dim = 512
### MS+OBD-SD | DoublePool
python main.py --dataset online_products --kernels 6 --source $datapath/online_products --n_epochs 200 \
--log_online --project OBD_SD --group SOP_MS_Resnet-512-double_obdsd_e200_lamda-200.0_d-w0.5 \
--kd_lambda 200.0 --diffusion_w 0.5 --diffusion_tau 0.005 --seed 0 --gpu 0 --bs 112 \
--samples_per_class 2 --loss multisimilarity_obd_sd --arch resnet50_normalize_frozen_double --PSD --OBDP --embed_dim 512 \
--tau 70 100 --gamma 0.2
