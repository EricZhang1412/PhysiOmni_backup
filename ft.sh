OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 train_finetune.py \
    --dataset_dir /data2/dataset/hmc_bak/HMC/preprocessed \
    --out_dir ./ \
    --dataset HMC \
    --dist_eval \
    --pretrained_dir ./PhysioOmni.pt