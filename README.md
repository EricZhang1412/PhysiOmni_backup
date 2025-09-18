# PhysioOmni

## Enviroment preparation
```bash
conda create -n PhysioOmni python=3.12
conda activate PhysioOmni
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install wandb einops pandas scikit-learn
```

## Prepare dataset
Using scripts in /prepare_dataset folder to prepare pre-training and downstream dataset.

## Decoupled Multimodal Tokenizer Training 
```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 train_vq.py \
    --dataset_dir /path/to/your/dataset \
    --out_dir /path/to/save \
    --wandb_log \
    --wandb_project your_project_name \
    --wandb_runname your_runname \
    --wandb_api_key your_api_key \
```

## Masked Signal Pre-training
```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 train_msm.py \
    --dataset_dir /path/to/your/dataset \
    --out_dir /path/to/save \
    --wandb_log \
    --wandb_project your_project_name \
    --wandb_runname your_runname \
    --wandb_api_key your_api_key \
```

## Masked Signal Pre-training
```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 train_msm.py \
    --dataset_dir /path/to/your/dataset \
    --out_dir /path/to/save \
    --wandb_log \
    --wandb_project your_project_name \
    --wandb_runname your_runname \
    --wandb_api_key your_api_key \
```

## Resilient Fine-tuning with Prototype Alignment
```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 train_finetune.py \
    --dataset_dir /path/to/your/dataset \
    --out_dir /path/to/save \
    --dataset SEED-VII \
    --dist_eval \
    --wandb_log \
    --wandb_project your_project_name \
    --wandb_runname your_runname \
    --wandb_api_key your_api_key \
```
