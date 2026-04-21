from torch import nn
from model.neural_transformer import NeuralTransformer, NTConfig
import torch
from torch.functional import F
import inspect
from collections import OrderedDict
from utils import get_metrics
from model.transformer import Block
from model.norm_ema_quantizer import l2norm, kmeans


class AlignmentModule(nn.Module):
    def __init__(self, n_classes = 7, num_position = 128, dim=256):
        super(AlignmentModule, self).__init__()
        self.base_vectors = nn.Parameter(torch.randn(n_classes, dim))
        self.conv = nn.Conv2d(1, 1, kernel_size=(num_position, 1), stride=(1, 1), padding=(0, 0), bias=False)
    
    def forward(self, input, Y):
        # input [batchsize, n, 256]
        # Y [batchsize]
        # input = torch.mean(input, dim=1, keepdim=True) # [batch_size, 1, 256]
        input = torch.unsqueeze(input, dim=1) # [batch_size, 1, 128, 256]
        input = self.conv(input) # [batch_size, 1, 1, 256]
        input = torch.squeeze(input, dim=2) # [batch_size, 1, 256]
        Y = torch.unsqueeze(Y, dim=-1) # [batch_size, 1]
        base_vectors = F.embedding(Y, self.base_vectors) # [batch_size, 1, 256]
        sim = torch.mean((base_vectors - input) ** 2, dim=-1) # [batch_size, 1]
        sim = torch.mean(sim)

        return input, sim


class AlignmentModuleForRegression(nn.Module):
    def __init__(self, n_embedings=256, num_position=128, dim=256):
        super(AlignmentModuleForRegression, self).__init__()
        self.dim = dim
        self.n_embedings = n_embedings
        self.base_vectors = nn.Parameter(torch.randn(n_embedings, dim))
        self.conv = nn.Conv2d(1, 1, kernel_size=(num_position, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.register_buffer('initted', torch.Tensor([False]))
        self.cluster_size = nn.Parameter(torch.zeros(n_embedings), requires_grad = False)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        print("Performing Kmeans init for codebook")
        embed, cluster_size = kmeans(data, self.n_embedings, 10, use_cosine_sim=True)
        self.base_vectors.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
    
    def forward(self, input, Y, indices=None):
        # d = (Y - self.values).abs()
        # indices = torch.argmin(d, dim=1)

        input = torch.unsqueeze(input, dim=1) # [batch_size, 1, 128, 256]
        input = self.conv(input) # [batch_size, 1, 1, 256]
        input = torch.squeeze(input, dim=2) # [batch_size, 1, 256]

        if indices is None:
            z = l2norm(input)
            base_vectors = l2norm(self.base_vectors)
            z_flattened = z.reshape(-1, self.dim)
            self.init_embed_(z_flattened)
            d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
                base_vectors.pow(2).sum(dim=1) - 2 * \
                torch.einsum('bd,nd->bn', z_flattened, base_vectors)
            indices = torch.argmin(d, dim=1)

        Y = torch.unsqueeze(Y, dim=-1) # [batch_size, 1]
        base_vectors = F.embedding(indices, self.base_vectors) # [batch_size, 1, 256]
        sim = torch.mean((base_vectors - input) ** 2, dim=-1) # [batch_size, 1]
        sim = torch.mean(sim)

        return input, sim, indices


# model for fine-tuning
class FT(nn.Module):
    def __init__(self, EEG_config, EOG_config, ECG_config, EMG_config, pretrained_ckpt_path=None
                 , n_classes=7, regression=False, emb_dropout=0.5, loss_ratio=[1, 0.5, 0.5, 0.5, 0.5], n_embedings=256
                 , **kwargs):
        super().__init__()
        self.EEG_encoder = NeuralTransformer(EEG_config) if EEG_config is not None else None
        self.EOG_encoder = NeuralTransformer(EOG_config) if EOG_config is not None else None
        self.ECG_encoder = NeuralTransformer(ECG_config) if ECG_config is not None else None
        self.EMG_encoder = NeuralTransformer(EMG_config) if EMG_config is not None else None

        if pretrained_ckpt_path is not None:
            print('loading weight from pretrained_ckpt')
            pretrained_ckpt = torch.load(pretrained_ckpt_path, weights_only=False)['model']
            EEG_dict = OrderedDict()
            EOG_dict = OrderedDict()
            ECG_dict = OrderedDict()
            EMG_dict = OrderedDict()
            for key in list(pretrained_ckpt.keys()):
                if key.startswith('EEG_encoder.'):
                    EEG_dict[key[len('EEG_encoder.'):]] = pretrained_ckpt[key]
                elif key.startswith('EOG_encoder.'):
                    EOG_dict[key[len('EOG_encoder.'):]] = pretrained_ckpt[key]
                elif key.startswith('ECG_encoder.'):
                    ECG_dict[key[len('ECG_encoder.'):]] = pretrained_ckpt[key]
                elif key.startswith('EMG_encoder.'):
                    ECG_dict[key[len('EMG_encoder.'):]] = pretrained_ckpt[key]
            if EEG_config is not None:
                self.EEG_encoder.load_state_dict(EEG_dict, strict=False)
            if EOG_config is not None:
                self.EOG_encoder.load_state_dict(EOG_dict, strict=False)
            if ECG_config is not None:
                self.ECG_encoder.load_state_dict(ECG_dict, strict=False)
            if EMG_config is not None:
                self.EMG_encoder.load_state_dict(EMG_dict, strict=False)
        
        self.loss_ratio = loss_ratio
        dim = 128
        # Embedding for EEG, EOG, ECG to the same feature length
        self.EEG_embedding = nn.Sequential(
            nn.LayerNorm(EEG_config.n_embd),
            nn.Linear(EEG_config.n_embd, dim),
            nn.LayerNorm(dim)
        ) if EEG_config is not None else None
        self.EOG_embedding = nn.Sequential(
            nn.LayerNorm(EOG_config.n_embd),
            nn.Linear(EOG_config.n_embd, dim),
            nn.LayerNorm(dim)
        ) if EOG_config is not None else None
        self.ECG_embedding = nn.Sequential(
            nn.LayerNorm(ECG_config.n_embd),
            nn.Linear(ECG_config.n_embd, dim),
            nn.LayerNorm(dim)
        ) if ECG_config is not None else None
        self.EMG_embedding = nn.Sequential(
            nn.LayerNorm(EMG_config.n_embd),
            nn.Linear(EMG_config.n_embd, dim),
            nn.LayerNorm(dim)
        ) if EMG_config is not None else None

        self.dropout = nn.Dropout(emb_dropout)

        num_position = 128
        self.EEG_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_position)
        ) if EEG_config is not None else None
        self.EOG_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_position)
        ) if EOG_config is not None else None
        self.ECG_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_position)
        ) if ECG_config is not None else None
        self.EMG_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_position)
        ) if EMG_config is not None else None

        embd_dim = 128
        self.EEG_Linear = nn.Linear(EEG_config.n_embd, embd_dim) if EEG_config is not None else None
        self.EOG_Linear = nn.Linear(EOG_config.n_embd, embd_dim) if EOG_config is not None else None
        self.ECG_Linear = nn.Linear(ECG_config.n_embd, embd_dim) if ECG_config is not None else None
        self.EMG_Linear = nn.Linear(EMG_config.n_embd, embd_dim) if EMG_config is not None else None
        
        self.regression = regression
        if regression:
            self.alignment_module = AlignmentModuleForRegression(n_embedings=n_embedings, num_position=num_position, dim=embd_dim)
        else:
            self.alignment_module = AlignmentModule(n_classes if n_classes > 1 else 2, num_position, embd_dim)
        
        transformer_args = dict(n_layer=12, n_head=8, n_embd=embd_dim, block_size=1024, patch_size=200, 
                            bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=8)
        transformer_conf = NTConfig(**transformer_args)
        self.X_transformer = Block(transformer_conf)

        self.lm_head = nn.Linear(embd_dim, n_classes)

        if regression:
            self.loss_fn = nn.MSELoss()
        elif n_classes > 1:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        self.lm_head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def cal_accuracy(self, logits, targets):
        _, preds = torch.max(logits, dim=-1)
        accuracy = torch.sum(preds == targets).float() / targets.size(0)
        return accuracy.item()
    
    def reorganization(self, feature, position):
        feature = torch.matmul(feature.transpose(1, 2).contiguous(), position)
        feature = feature.transpose(1, 2).contiguous()
        return feature

    def forward(self, batch, metrics=None, mask=None, return_all_tokens=True):
        EEG_X = batch['EEG_X']; EOG_X = batch['EOG_X']; ECG_X = batch['ECG_X']; EMG_X = batch['EMG_X']
        Y = batch['Y']
        EEG_input_chans = batch['EEG_input_chans']; EEG_input_time = batch['EEG_input_time']
        EOG_input_chans = batch['EOG_input_chans']; EOG_input_time = batch['EOG_input_time']
        ECG_input_chans = batch['ECG_input_chans']; ECG_input_time = batch['ECG_input_time']
        EMG_input_chans = batch['EMG_input_chans']; EMG_input_time = batch['EMG_input_time']

        EEG_X_inputs = self.EEG_encoder(EEG_X, EEG_input_chans, EEG_input_time, mask, return_all_tokens) if EEG_X is not None else None
        EOG_X_inputs = self.EOG_encoder(EOG_X, EOG_input_chans, EOG_input_time, mask, return_all_tokens) if EOG_X is not None else None
        ECG_X_inputs = self.ECG_encoder(ECG_X, ECG_input_chans, ECG_input_time, mask, return_all_tokens) if ECG_X is not None else None
        EMG_X_inputs = self.EMG_encoder(EMG_X, EMG_input_chans, EMG_input_time, mask, return_all_tokens) if EMG_X is not None else None
        
        EEG_X = self.dropout(self.EEG_embedding(EEG_X_inputs)) if EEG_X is not None else None #(B, seq_len_EEG, dim)
        EOG_X = self.dropout(self.EOG_embedding(EOG_X_inputs)) if EOG_X is not None else None #(B, seq_len_EOG, dim)
        ECG_X = self.dropout(self.ECG_embedding(ECG_X_inputs)) if ECG_X is not None else None #(B, seq_len_ECG, dim)
        EMG_X = self.dropout(self.EMG_embedding(EMG_X_inputs)) if EMG_X is not None else None #(B, seq_len_EMG, dim)

        EEG_X = self.EEG_head(EEG_X) if EEG_X is not None else None #(B, seq_len_EEG, num_position)
        EOG_X = self.EOG_head(EOG_X) if EOG_X is not None else None #(B, seq_len_EOG, num_position)
        ECG_X = self.ECG_head(ECG_X) if ECG_X is not None else None #(B, seq_len_ECG, num_position)
        EMG_X = self.EMG_head(EMG_X) if EMG_X is not None else None #(B, seq_len_ECG, num_position)

        EEG_X = torch.softmax(EEG_X, dim=-1) if EEG_X is not None else None #(B, seq_len_EEG, num_position)
        EOG_X = torch.softmax(EOG_X, dim=-1) if EOG_X is not None else None #(B, seq_len_EOG, num_position)
        ECG_X = torch.softmax(ECG_X, dim=-1) if ECG_X is not None else None #(B, seq_len_ECG, num_position)
        EMG_X = torch.softmax(EMG_X, dim=-1) if EMG_X is not None else None #(B, seq_len_EMG, num_position)

        EEG_X = self.reorganization(EEG_X_inputs, EEG_X) if EEG_X is not None else None #(B, num_position, EEG_config.n_embd)
        EOG_X = self.reorganization(EOG_X_inputs, EOG_X) if EOG_X is not None else None #(B, num_position, EOG_config.n_embd)
        ECG_X = self.reorganization(ECG_X_inputs, ECG_X) if ECG_X is not None else None #(B, num_position, ECG_config.n_embd)
        EMG_X = self.reorganization(EMG_X_inputs, EMG_X) if EMG_X is not None else None #(B, num_position, EMG_config.n_embd)

        EEG_X = self.EEG_Linear(EEG_X) if EEG_X is not None else None #(B, num_position, embd_dim)
        EOG_X = self.EOG_Linear(EOG_X) if EOG_X is not None else None #(B, num_position, embd_dim)
        ECG_X = self.ECG_Linear(ECG_X) if ECG_X is not None else None #(B, num_position, embd_dim)
        EMG_X = self.EMG_Linear(EMG_X) if EMG_X is not None else None #(B, num_position, embd_dim)

        EEG_X = self.X_transformer(EEG_X) if EEG_X is not None else None #(B, num_position, embd_dim)
        EOG_X = self.X_transformer(EOG_X) if EOG_X is not None else None #(B, num_position, embd_dim)
        ECG_X = self.X_transformer(ECG_X) if ECG_X is not None else None #(B, num_position, embd_dim)
        EMG_X = self.X_transformer(EMG_X) if EMG_X is not None else None #(B, num_position, embd_dim)

        if self.regression:
            EEG_X, alignment_loss_EEG, indices = self.alignment_module(EEG_X, Y) if EEG_X is not None else (None, None, None) #(B, 1, embd_dim)
            EOG_X, alignment_loss_EOG, _ = self.alignment_module(EOG_X, Y, indices) if EOG_X is not None else (None, None, None) #(B, 1, embd_dim)
            ECG_X, alignment_loss_ECG, _ = self.alignment_module(ECG_X, Y, indices) if ECG_X is not None else (None, None, None) #(B, 1, embd_dim)
            EMG_X, alignment_loss_EMG, _ = self.alignment_module(EMG_X, Y, indices) if EMG_X is not None else (None, None, None) #(B, 1, embd_dim)
        else:
            EEG_X, alignment_loss_EEG = self.alignment_module(EEG_X, Y) if EEG_X is not None else (None, None) #(B, 1, embd_dim)
            EOG_X, alignment_loss_EOG = self.alignment_module(EOG_X, Y) if EOG_X is not None else (None, None) #(B, 1, embd_dim)
            ECG_X, alignment_loss_ECG = self.alignment_module(ECG_X, Y) if ECG_X is not None else (None, None) #(B, 1, embd_dim)
            EMG_X, alignment_loss_EMG = self.alignment_module(EMG_X, Y) if EMG_X is not None else (None, None) #(B, 1, embd_dim)

        alignment_loss = sum([loss for loss in [alignment_loss_EEG, alignment_loss_EOG, alignment_loss_ECG, alignment_loss_EMG] if loss is not None])
        num_modality = sum([1 for loss in [EEG_X, EOG_X, ECG_X, EMG_X] if loss is not None])
        feature = 0
        specific_loss = 0
        for modality, ratio in zip([X for X in [EEG_X, EOG_X, ECG_X, EMG_X] if X is not None], self.loss_ratio[2:]):
            feature += modality
            if 'f1_weighted' in metrics or 'r2' in metrics:
                specific_loss += ratio * self.loss_fn(self.lm_head(modality.squeeze()), Y)
            else:
                specific_loss += ratio * self.loss_fn(self.lm_head(modality.squeeze()), Y.float().unsqueeze(-1))
            
        feature /= num_modality
        
        logits = self.lm_head(feature.squeeze())

        if 'f1_weighted' in metrics or 'r2' in metrics:
            loss = self.loss_fn(logits, Y)
        else:
            loss = self.loss_fn(logits, Y.float().unsqueeze(-1))

        total_loss = self.loss_ratio[0] * alignment_loss + self.loss_ratio[1] * loss + specific_loss

        if not self.training:
            return loss.item(), logits.to(torch.float32)

        with torch.amp.autocast('cuda', enabled=False):
            if 'r2' in metrics:
                results = get_metrics(logits.to(torch.float32).detach().cpu().numpy(), Y.cpu().numpy(), metrics, is_binary=False)
            elif 'f1_weighted' not in metrics:
                # binary classification
                results = get_metrics(torch.sigmoid(logits.to(torch.float32).detach()).cpu().numpy(), Y.cpu().numpy(), metrics, is_binary=True)
            else:
                # multi-class classification
                results = get_metrics(logits.to(torch.float32).detach().cpu().numpy(), Y.cpu().numpy(), metrics, is_binary=False)

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/total_loss'] = total_loss.item()
        log[f'{split}/loss'] = loss.item()
        log[f'{split}/alignment_loss'] = alignment_loss.item()
        log[f'{split}/specific_loss'] = specific_loss.item()
        for key, value in results.items():
            log[f'{split}/{key}'] = value

        return total_loss, log
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
