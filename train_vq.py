import os
import time
from contextlib import nullcontext

import torch
import torch._dynamo.config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.VQ import VQ
from model.neural_transformer import NTConfig
from utils import cosine_scheduler, prepare_pretrain_dataset, prepare_pretrain_VQ_dataset
import argparse
import numpy as np


master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None

def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    # various inits, derived attributes, I/O setup
    backend = 'nccl' # 'nccl', 'gloo', etc.
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def get_num_unused_code(model):
    if hasattr(model, 'shared_quantize'):
        try:
            codebook_cluster_size = model.shared_quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.shared_quantize.cluster_size
        shared_zero_cnt = (codebook_cluster_size == 0).sum().item()
    if hasattr(model, 'EEG_quantize'):
        try:
            codebook_cluster_size = model.EEG_quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.EEG_quantize.cluster_size
        EEG_zero_cnt = (codebook_cluster_size == 0).sum().item()
    if hasattr(model, 'EOG_quantize'):
        try:
            codebook_cluster_size = model.EOG_quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.EOG_quantize.cluster_size
        EOG_zero_cnt = (codebook_cluster_size == 0).sum().item()
    if hasattr(model, 'ECG_quantize'):
        try:
            codebook_cluster_size = model.ECG_quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.ECG_quantize.cluster_size
        ECG_zero_cnt = (codebook_cluster_size == 0).sum().item()
    if hasattr(model, 'EMG_quantize'):
        try:
            codebook_cluster_size = model.EMG_quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.EMG_quantize.cluster_size
        EMG_zero_cnt = (codebook_cluster_size == 0).sum().item()
    return shared_zero_cnt, EEG_zero_cnt, EOG_zero_cnt, ECG_zero_cnt, EMG_zero_cnt


@torch.no_grad
def evaluate(model, dataloader):
    global ctx
    model.eval()
    log_all = {}
    for _, (batch) in enumerate(dataloader):
        data = {
            'EEG_X': batch['EEG_X'].float().to(device, non_blocking=True) if 'EEG_X' in batch else None,
            'EOG_X': batch['EOG_X'].float().to(device, non_blocking=True) if 'EOG_X' in batch else None,
            'ECG_X': batch['ECG_X'].float().to(device, non_blocking=True) if 'ECG_X' in batch else None,
            'EMG_X': batch['EMG_X'].float().to(device, non_blocking=True) if 'EMG_X' in batch else None,
            'EEG_Y': batch['EEG_Y'].float().to(device, non_blocking=True) if 'EEG_Y' in batch else None,
            'EOG_Y': batch['EOG_Y'].float().to(device, non_blocking=True) if 'EOG_Y' in batch else None,
            'ECG_Y': batch['ECG_Y'].float().to(device, non_blocking=True) if 'ECG_Y' in batch else None,
            'EMG_Y': batch['EMG_Y'].float().to(device, non_blocking=True) if 'EMG_Y' in batch else None,
            'EEG_input_chans': batch['EEG_input_chans'].to(device, non_blocking=True) if 'EEG_input_chans' in batch else None,
            'EOG_input_chans': batch['EOG_input_chans'].to(device, non_blocking=True) if 'EOG_input_chans' in batch else None,
            'ECG_input_chans': batch['ECG_input_chans'].to(device, non_blocking=True) if 'ECG_input_chans' in batch else None,
            'EMG_input_chans': batch['EMG_input_chans'].to(device, non_blocking=True) if 'EMG_input_chans' in batch else None,
            'EEG_input_time': batch['EEG_input_time'].to(device, non_blocking=True) if 'EEG_input_time' in batch else None,
            'EOG_input_time': batch['EOG_input_time'].to(device, non_blocking=True) if 'EOG_input_time' in batch else None,
            'ECG_input_time': batch['ECG_input_time'].to(device, non_blocking=True) if 'ECG_input_time' in batch else None,
            'EMG_input_time': batch['EMG_input_time'].to(device, non_blocking=True) if 'EMG_input_time' in batch else None,
        }
        with ctx:
            loss, log = model(data)
        
        for key, value in log.items():
            if log_all.get(key) is None:
                log_all[key] = [value]
            else:
                log_all[key].append(value)

    for key, value in log_all.items():
        log_all[key] = np.mean(log_all[key])

    model.train()
    
    return log_all


def get_dataloader(dataset, batch_size, is_train=True):
    global ddp, ddp_world_size, ddp_rank
    if is_train:
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False
        batch_size = int(1.5 * batch_size)
    if ddp:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=shuffle
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=drop_last,
            shuffle=shuffle
        )
    return data_loader


def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/VQ')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
    print('prepare dataloader...')
    dataset_train_list = prepare_pretrain_dataset('dataset.yaml')
    _, dataset_val = prepare_pretrain_VQ_dataset('./SEED-VII', contain_EEG=True, contain_EOG=True, contain_ECG=True)
    print('finished!')

    data_loader_train_list = [get_dataloader(dataset, args.batch_size) for dataset in dataset_train_list]
    data_loader_val = get_dataloader(dataset_val, args.batch_size, is_train=False)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0

    # model init
    EEG_encoder_args = dict(n_layer=12, n_head=10, n_embd=200, block_size=1024, patch_size=200, 
                            bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=8)
    EEG_decoder_args = dict(n_layer=3, n_head=10, n_embd=200, block_size=1024, patch_size=200, 
                            bias=False, dropout=0., num_classes=0, in_chans=128)
    EOG_encoder_args = dict(n_layer=12, n_head=10, n_embd=100, block_size=1024, patch_size=100, emb_after_conv_size=104,
                            bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=8, n_query=1)
    EOG_decoder_args = dict(n_layer=3, n_head=10, n_embd=100, block_size=1024, patch_size=100, 
                            bias=False, dropout=0., num_classes=0, in_chans=128, n_query=-1)
    ECG_encoder_args = dict(n_layer=12, n_head=10, n_embd=100, block_size=1024, patch_size=100, emb_after_conv_size=104,
                            bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=8, n_query=1)
    ECG_decoder_args = dict(n_layer=3, n_head=10, n_embd=100, block_size=1024, patch_size=100, 
                            bias=False, dropout=0., num_classes=0, in_chans=128, n_query=-1)
    EMG_encoder_args = dict(n_layer=12, n_head=10, n_embd=100, block_size=1024, patch_size=100, emb_after_conv_size=104,
                            bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=8, n_query=1)
    EMG_decoder_args = dict(n_layer=3, n_head=10, n_embd=100, block_size=1024, patch_size=100, 
                            bias=False, dropout=0., num_classes=0, in_chans=128, n_query=-1)

    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')):
        init_from = 'resume'
    else:
        init_from = 'scratch'

    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        EEG_encoder_conf = NTConfig(**EEG_encoder_args)
        EEG_decoder_conf = NTConfig(**EEG_decoder_args)
        EOG_encoder_conf = NTConfig(**EOG_encoder_args)
        EOG_decoder_conf = NTConfig(**EOG_decoder_args)
        ECG_encoder_conf = NTConfig(**ECG_encoder_args)
        ECG_decoder_conf = NTConfig(**ECG_decoder_args)
        EMG_encoder_conf = NTConfig(**EMG_encoder_args)
        EMG_decoder_conf = NTConfig(**EMG_decoder_args)

        model = VQ(EEG_encoder_conf, EOG_encoder_conf, ECG_encoder_conf, EMG_encoder_conf,
                   EEG_decoder_conf, EOG_decoder_conf, ECG_decoder_conf, EMG_decoder_conf)
        
        start_epoch = 0
        best_val_loss = 999
    elif init_from == 'resume':
        print(f"Resuming training from {checkpoint_out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        EEG_checkpoint_model_args = checkpoint['EEG_encoder_args']
        EOG_checkpoint_model_args = checkpoint['EOG_encoder_args']
        ECG_checkpoint_model_args = checkpoint['ECG_encoder_args']
        EMG_checkpoint_model_args = checkpoint['EMG_encoder_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
            EEG_encoder_args[k] = EEG_checkpoint_model_args[k]
            EOG_encoder_args[k] = EOG_checkpoint_model_args[k]
            ECG_encoder_args[k] = ECG_checkpoint_model_args[k]
            EMG_encoder_args[k] = EMG_checkpoint_model_args[k]
        EEG_checkpoint_model_args = checkpoint['EEG_decoder_args']
        EOG_checkpoint_model_args = checkpoint['EOG_decoder_args']
        ECG_checkpoint_model_args = checkpoint['ECG_decoder_args']
        EMG_checkpoint_model_args = checkpoint['EMG_decoder_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
            EEG_decoder_args[k] = EEG_checkpoint_model_args[k]
            EOG_decoder_args[k] = EOG_checkpoint_model_args[k]
            ECG_decoder_args[k] = ECG_checkpoint_model_args[k]
            EMG_decoder_args[k] = EMG_checkpoint_model_args[k]
        # create the model
        EEG_encoder_conf = NTConfig(**EEG_encoder_args)
        EEG_decoder_conf = NTConfig(**EEG_decoder_args)
        EOG_encoder_conf = NTConfig(**EOG_encoder_args)
        EOG_decoder_conf = NTConfig(**EOG_decoder_args)
        ECG_encoder_conf = NTConfig(**ECG_encoder_args)
        ECG_decoder_conf = NTConfig(**ECG_decoder_args)
        EMG_encoder_conf = NTConfig(**EMG_encoder_args)
        EMG_decoder_conf = NTConfig(**EMG_decoder_args)
        model = VQ(EEG_encoder_conf, EOG_encoder_conf, ECG_encoder_conf, EMG_encoder_conf,
                   EEG_decoder_conf, EOG_decoder_conf, ECG_decoder_conf, EMG_decoder_conf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if args.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    # logging
    if args.wandb_log and master_process:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(project=args.wandb_project, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'), resume=False)

    num_training_steps_per_epoch = sum([len(dataset) for dataset in dataset_train_list]) // args.batch_size // ddp_world_size
    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs
    )


    # training loop
    t0 = time.time()
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    for epoch in range(start_epoch, args.epochs):
        local_iter_num = 0
        for num, data_loader_train in enumerate(data_loader_train_list):
            if master_process:
                print(f"Training on dataset {num + 1}/{len(data_loader_train_list)}")
            for step, (batch) in enumerate(data_loader_train):
                # determine and set the learning rate for this iteration
                lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # forward backward update, with optional gradient accumulation to simulate larger batch size
                # and using the GradScaler if data type is float16
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0

                data = {
                    'EEG_X': batch['EEG_X'].float().to(device, non_blocking=True) if 'EEG_X' in batch else None,
                    'EOG_X': batch['EOG_X'].float().to(device, non_blocking=True) if 'EOG_X' in batch else None,
                    'ECG_X': batch['ECG_X'].float().to(device, non_blocking=True) if 'ECG_X' in batch else None,
                    'EMG_X': batch['EMG_X'].float().to(device, non_blocking=True) if 'EMG_X' in batch else None,
                    'EEG_Y': batch['EEG_Y'].float().to(device, non_blocking=True) if 'EEG_Y' in batch else None,
                    'EOG_Y': batch['EOG_Y'].float().to(device, non_blocking=True) if 'EOG_Y' in batch else None,
                    'ECG_Y': batch['ECG_Y'].float().to(device, non_blocking=True) if 'ECG_Y' in batch else None,
                    'EMG_Y': batch['EMG_Y'].float().to(device, non_blocking=True) if 'EMG_Y' in batch else None,
                    'EEG_input_chans': batch['EEG_input_chans'].to(device, non_blocking=True) if 'EEG_input_chans' in batch else None,
                    'EOG_input_chans': batch['EOG_input_chans'].to(device, non_blocking=True) if 'EOG_input_chans' in batch else None,
                    'ECG_input_chans': batch['ECG_input_chans'].to(device, non_blocking=True) if 'ECG_input_chans' in batch else None,
                    'EMG_input_chans': batch['EMG_input_chans'].to(device, non_blocking=True) if 'EMG_input_chans' in batch else None,
                    'EEG_input_time': batch['EEG_input_time'].to(device, non_blocking=True) if 'EEG_input_time' in batch else None,
                    'EOG_input_time': batch['EOG_input_time'].to(device, non_blocking=True) if 'EOG_input_time' in batch else None,
                    'ECG_input_time': batch['ECG_input_time'].to(device, non_blocking=True) if 'ECG_input_time' in batch else None,
                    'EMG_input_time': batch['EMG_input_time'].to(device, non_blocking=True) if 'EMG_input_time' in batch else None,
                }

                with ctx:
                    loss, log = model(data)
                    loss = loss / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # clip the gradient
                    if args.grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    # step the optimizer and scaler if training in fp16
                    scaler.step(optimizer)
                    scaler.update()
                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)

                # evaluate the loss on train/val sets and write checkpoints
                if (local_iter_num + 1) % args.log_interval == 0 and master_process:
                    shared_zero_cnt, EEG_zero_cnt, EOG_zero_cnt, ECG_zero_cnt, EMG_zero_cnt = get_num_unused_code(raw_model)

                    message = f'Epoch {epoch} step [{local_iter_num + 1}/{num_training_steps_per_epoch}]: '
                    for key, value in log.items():
                        message += key.split('/')[-1]
                        message += f' {value:.4f}, '
                    message += f'shared_unused_code {shared_zero_cnt}, EEG_unused_code {EEG_zero_cnt}, EOG_unused_code {EOG_zero_cnt}, ECG_unused_code {ECG_zero_cnt}, EMG_unused_code {EMG_zero_cnt}'
                    print(message)
                    
                    if args.wandb_log:
                        log_train = log.copy()
                        log_train.update({
                            "iter": iter_num,
                            "train/shared_unused_code": shared_zero_cnt,
                            "train/EEG_unused_code": EEG_zero_cnt,
                            "train/EOG_unused_code": EOG_zero_cnt,
                            "train/ECG_unused_code": ECG_zero_cnt,
                            "train/EMG_unused_code": EMG_zero_cnt,
                            "lr": lr
                        })
                        wandb.log(log_train)

                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                iter_num += 1
                local_iter_num += 1

        # validation
        log_val = evaluate(model, data_loader_val)
        if log_val['val/total_loss'] < best_val_loss:
            best_val_loss = log_val['val/total_loss']
        if master_process:
            print('='* 10)
            message = 'Evaluate: '
            for key, value in log_val.items():
                message += key.split('/')[-1]
                message += f' {value:.4f}, '
            print(message[:-2])
            print('='* 10)
            if args.wandb_log:
                wandb.log(log_val)
        
        if master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'EEG_encoder_args': EEG_encoder_args,
                'EEG_decoder_args': EEG_decoder_args,
                'EOG_encoder_args': EOG_encoder_args,
                'EOG_decoder_args': EOG_decoder_args,
                'ECG_encoder_args': ECG_encoder_args,
                'ECG_decoder_args': ECG_decoder_args,
                'EMG_encoder_args': EMG_encoder_args,
                'EMG_decoder_args': EMG_decoder_args,
                'iter_num': iter_num,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }
            print(f"saving checkpoint to {checkpoint_out_dir}")
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))
        
            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))

    if ddp:
        destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--out_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='MMFM')
    parser.add_argument('--wandb_runname', default='VQ')
    parser.add_argument('--wandb_api_key', type=str)
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--block_size', default=512, type=int)

    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    parser.add_argument('--compile', default=False, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
