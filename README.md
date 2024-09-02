Training on CIFAR-10/100 dataset

lam_hard='0.025'
lam_con='0.05'
python Train_cifar.py --dataset 'cifar100' --num_class 100 --data_path 'your data-path' --noise_mode 'sym' --r 0.9  --lambda_hard lam_hard  --lambda_con lam_con
