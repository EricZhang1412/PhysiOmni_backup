import os
import time
from contextlib import nullcontext
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model.neural_transformer import NTConfig
from utils import cosine_scheduler, prepare_SEED7_dataset, prepare_HMC_dataset, prepare_FBM_dataset, prepare_EEGMAT_dataset
import argparse
import numpy as np
from model.FT import FT
from utils import get_metrics
import itertools
import torch.distributed as dist


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


@torch.no_grad
def evaluate(model, dataloader, metrics, use_EEG, use_EOG, use_ECG, use_EMG, args):
    global ctx, ddp_world_size
    model.eval()
    pred = []
    true = []
    total_loss = []
    log = {}
    for _, (batch) in enumerate(dataloader):
        data = {
            'EEG_X': batch['EEG_X'].float().to(device, non_blocking=True) if use_EEG else None,
            'EOG_X': batch['EOG_X'].float().to(device, non_blocking=True) if use_EOG else None,
            'ECG_X': batch['ECG_X'].float().to(device, non_blocking=True) if use_ECG else None,
            'EMG_X': batch['EMG_X'].float().to(device, non_blocking=True) if use_EMG else None,
            'Y': batch['Y'].to(device, non_blocking=True) if 'Y' in batch else None,
            'EEG_input_chans': batch['EEG_input_chans'].to(device, non_blocking=True) if use_EEG else None,
            'EOG_input_chans': batch['EOG_input_chans'].to(device, non_blocking=True) if use_EOG else None,
            'ECG_input_chans': batch['ECG_input_chans'].to(device, non_blocking=True) if use_ECG else None,
            'EMG_input_chans': batch['EMG_input_chans'].to(device, non_blocking=True) if use_EMG else None,
            'EEG_input_time': batch['EEG_input_time'].to(device, non_blocking=True) if use_EEG else None,
            'EOG_input_time': batch['EOG_input_time'].to(device, non_blocking=True) if use_EOG else None,
            'ECG_input_time': batch['ECG_input_time'].to(device, non_blocking=True) if use_ECG else None,
            'EMG_input_time': batch['EMG_input_time'].to(device, non_blocking=True) if use_EMG else None,
        }

        with ctx:
            loss, logits = model(data, metrics=metrics)
        
        pred.append(logits)
        true.append(data['Y'])
        total_loss.append(loss)

    pred = torch.cat(pred, dim=0)
    true = torch.cat(true, dim=0)

    if args.dist_eval:
        gathered_pred = [torch.zeros_like(pred, device=pred.device) for _ in range(ddp_world_size)]
        gathered_target = [torch.zeros_like(true, device=true.device) for _ in range(ddp_world_size)]
        dist.all_gather(gathered_pred, pred)
        dist.all_gather(gathered_target, true)
        pred = torch.cat(gathered_pred, dim=0)
        true = torch.cat(gathered_target, dim=0)

    total_loss = np.mean(total_loss)

    if 'r2' in metrics:
        results = get_metrics(pred.cpu().numpy(), true.cpu().numpy(), metrics, is_binary=False)
    elif 'f1_weighted' not in metrics:
        # binary classification
        results = get_metrics(torch.sigmoid(pred).cpu().numpy(), true.cpu().numpy(), metrics, is_binary=True)
    else:
        # multi-class classification
        results = get_metrics(pred.cpu().numpy(), true.cpu().numpy(), metrics, is_binary=False)
    
    condition = f'EEG({use_EEG})_EOG({use_EOG})_ECG({use_ECG})_EMG({use_EMG})'
    log[f'val_{condition}/total_loss'] = total_loss
    for key, value in results.items():
        log[f'val_{condition}/{key}'] = value

    model.train()
    
    return log


def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints', args.dataset)
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
    print('prepare dataloader...')
    n_embedings = 256
    if args.dataset == 'SEED-VII':
        dataset_train, dataset_val, dataset_test = prepare_SEED7_dataset(args.dataset_dir, args.dataset, contain_EEG=True, contain_EOG=True, contain_ECG=True)
        contain_EEG = True
        contain_EOG = True
        contain_ECG = True
        contain_EMG = False
        n_classes = 7
        regression = False
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        monitor = "cohen_kappa"
        loss_ratio = [1, 0.5, 0.5, 0.5, 0.5]
        test_conditions = []
        for test_condition in list(itertools.product([True, False], repeat=3)):
            if sum(test_condition) == 0:
                continue
            test_conditions.append(list(test_condition) + [False])
    elif args.dataset == 'HMC':
        dataset_train, dataset_val, dataset_test = prepare_HMC_dataset(args.dataset_dir, args.dataset, contain_EEG=True, contain_EOG=True, contain_ECG=False, contain_EMG=True)
        contain_EEG = True
        contain_EOG = True
        contain_ECG = False
        contain_EMG = True
        n_classes = 5
        regression = False
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        monitor = "cohen_kappa"
        loss_ratio = [1, 0.1, 0.01, 4, 0.5]
        test_conditions = []
        for test_condition in list(itertools.product([True, False], repeat=3)):
            if sum(test_condition) == 0:
                continue
            test_condition = list(test_condition)
            test_condition.insert(2, False)
            test_conditions.append(test_condition)
    elif args.dataset == 'FBM':
        dataset_train, dataset_val, dataset_test = prepare_FBM_dataset(args.dataset_dir, args.dataset, contain_EEG=True, contain_EOG=True, contain_ECG=False, contain_EMG=True)
        contain_EEG = True
        contain_EOG = True
        contain_ECG = False
        contain_EMG = True
        n_classes = 66
        regression = True
        n_embedings = 256
        metrics = ["r2", "rmse", "pearsonr"]
        monitor = "r2"
        loss_ratio = [0.01, 0.1, 2, 0.5, 1]
        test_conditions = []
        for test_condition in list(itertools.product([True, False], repeat=3)):
            if sum(test_condition) == 0:
                continue
            test_condition = list(test_condition)
            test_condition.insert(2, False)
            test_conditions.append(test_condition)
    elif args.dataset == 'EEGMAT':
        dataset_train, dataset_val, dataset_test = prepare_EEGMAT_dataset(args.dataset_dir, args.dataset, contain_EEG=True, contain_EOG=False, contain_ECG=True, contain_EMG=False)
        contain_EEG = True
        contain_EOG = False
        contain_ECG = True
        contain_EMG = False
        n_classes = 1
        regression = False
        metrics = ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc"]
        monitor = "roc_auc"
        loss_ratio = [1, 0.5, 0.5, 0.5, 0.5]
        test_conditions = []
        for test_condition in list(itertools.product([True, False], repeat=2)):
            if sum(test_condition) == 0:
                continue
            test_condition = list(test_condition)
            test_condition.insert(1, False)
            test_condition.append(False)
            test_conditions.append(test_condition)
    print('finished!')

    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
        )
        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
            )
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=True,
            drop_last=False,
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_metric = -10

    # model init
    EEG_encoder_args = dict(n_layer=12, n_head=10, n_embd=200, block_size=1024, patch_size=200, 
                            bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=8)
    EOG_encoder_args = dict(n_layer=12, n_head=10, n_embd=100, block_size=1024, patch_size=100, emb_after_conv_size=104,
                            bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=8, n_query=1)
    ECG_encoder_args = dict(n_layer=12, n_head=10, n_embd=100, block_size=1024, patch_size=100, emb_after_conv_size=104,
                            bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=8, n_query=1)
    EMG_encoder_args = dict(n_layer=12, n_head=10, n_embd=100, block_size=1024, patch_size=100, emb_after_conv_size=104,
                            bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=8, n_query=1)

    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')):
        init_from = 'resume'
    else:
        init_from = 'pretrained'

    if init_from == 'pretrained':
        # init a new model from pretrained weights
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-pretrained training
        EEG_encoder_conf = NTConfig(**EEG_encoder_args)
        EOG_encoder_conf = NTConfig(**EOG_encoder_args)
        ECG_encoder_conf = NTConfig(**ECG_encoder_args)
        EMG_encoder_conf = NTConfig(**EMG_encoder_args)
        pretrained_ckpt_path = os.path.join(args.out_dir, args.pretrained_dir, 'ckpt-49.pt')
        model = FT(EEG_encoder_conf if contain_EEG else None, EOG_encoder_conf if contain_EOG else None, 
                   ECG_encoder_conf if contain_ECG else None, EMG_encoder_conf if contain_EMG else None, 
                   pretrained_ckpt_path, n_classes, regression, loss_ratio=loss_ratio, n_embedings=n_embedings)
        start_epoch = 0
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
        # create the model
        EEG_encoder_conf = NTConfig(**EEG_encoder_args)
        EOG_encoder_conf = NTConfig(**EOG_encoder_args)
        ECG_encoder_conf = NTConfig(**ECG_encoder_args)
        EMG_encoder_conf = NTConfig(**EMG_encoder_args)
        model = FT(EEG_encoder_conf if contain_EEG else None, EOG_encoder_conf if contain_EOG else None, 
                   ECG_encoder_conf if contain_ECG else None, EMG_encoder_conf if contain_EMG else None,
                   n_classes, regression, loss_ratio=loss_ratio, n_embedings=n_embedings)
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
        best_val_metric = checkpoint['best_val_metric']
        best_test_metric = checkpoint['best_test_metric']

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
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

    # logging
    if args.wandb_log and master_process:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(project=args.wandb_project, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'), resume=False)

    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // ddp_world_size
    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs
    )


    # training loop
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    for epoch in range(start_epoch, args.epochs):
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
                'Y': batch['Y'].to(device, non_blocking=True) if 'Y' in batch else None,
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
                loss, log = model(data, metrics=metrics)
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
            if (iter_num + 1) % args.log_interval == 0 and master_process:
                message = f'Epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: '
                for key, value in log.items():
                    message += key.split('/')[-1]
                    message += f' {value:.4f}, '
                print(message[:-2])
                
                if args.wandb_log:
                    log_train = log.copy()
                    log_train.update({
                        "iter": iter_num,
                        "lr": lr
                    })
                    wandb.log(log_train)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            iter_num += 1
            local_iter_num += 1

        is_better = False
        test_metric = []
        for (use_EEG, use_EOG, use_ECG, use_EMG) in test_conditions:
            # validation
            log_val = evaluate(model, data_loader_val, metrics, use_EEG, use_EOG, use_ECG, use_EMG, args)
            if master_process:
                print('='* 10)
                message = f'Evaluate EEG({use_EEG})_EOG({use_EOG})_ECG({use_ECG})_EMG({use_EMG}): '
                for key, value in log_val.items():
                    message += key.split('/')[-1]
                    message += f' {value:.4f}, '
                print(message[:-2])
                print('='* 10)
                if args.wandb_log:
                    wandb.log(log_val)

            # test
            log = evaluate(model, data_loader_test, metrics, use_EEG, use_EOG, use_ECG, use_EMG, args)
            log_test = {k.replace('val', 'test'): v for k, v in log.items()}
            if master_process:
                print('='* 10)
                message = f'Test EEG({use_EEG})_EOG({use_EOG})_ECG({use_ECG})_EMG({use_EMG}): '
                for key, value in log_test.items():
                    message += key.split('/')[-1]
                    message += f' {value:.4f}, '
                print(message[:-2])
                print('='* 10)
                if args.wandb_log:
                    wandb.log(log_test)

            if (use_EEG, use_EOG, use_ECG, use_EMG) == (contain_EEG, contain_EOG, contain_ECG, contain_EMG):
                monitor_metric = f'val_EEG({contain_EEG})_EOG({contain_EOG})_ECG({contain_ECG})_EMG({contain_EMG})/' + monitor
                if log_val[monitor_metric] > best_val_metric:
                    is_better = True
                    best_val_metric = log_val[monitor_metric]

            if is_better:
                test_metric.append(log_test.copy())

        if is_better:
            best_test_metric = test_metric.copy()
        
        if master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'EEG_encoder_args': EEG_encoder_args,
                'EOG_encoder_args': EOG_encoder_args,
                'ECG_encoder_args': ECG_encoder_args,
                'EMG_encoder_args': EMG_encoder_args,
                'best_val_metric': best_val_metric,
                'best_test_metric': best_test_metric,
                'iter_num': iter_num,
                'epoch': epoch,
            }
            print(f"saving checkpoint to {checkpoint_out_dir}")
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))
        
            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))
    
    if master_process:
        for test_metric in best_test_metric:
            for key, value in test_metric.items():
                m = key.split('/')[0][5:]
                break
            message = f'Best test metrics {m}: '
            for key, value in test_metric.items():
                message += key.split('/')[-1]
                message += f' {value:.5f}, '
            print(message[:-2])

    if ddp:
        destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--out_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--dataset_dir', default='./', help='path where data is')
    parser.add_argument('--pretrained_dir', default='checkpoints/MSM', help='path where pretrained model is')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--dataset', default='SEED-VII', help='dataset: SEED-VII | HMC | FBM | EEGMAT')
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='MMFM')
    parser.add_argument('--wandb_runname', default='SEED-VII')
    parser.add_argument('--wandb_api_key', type=str)
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--block_size', default=512, type=int)

    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    parser.add_argument('--dist_eval', default=False, action='store_true')
    parser.add_argument('--compile', default=False, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
