from torch import nn
from model.transformer import Block
from model.neural_transformer import TemporalConv
import math
import torch
from torch.functional import F
import inspect


class NeuralTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.num_classes = config.num_classes

        # To identify whether it is neural tokenizer or neural decoder. 
        # For the neural decoder, use linear projection (PatchEmbed) to project codebook dimension to hidden dimension.
        # Otherwise, use TemporalConv to extract temporal features from EEG signals.
        self.patch_embed = TemporalConv(out_chans=config.out_chans, config=config) if config.in_chans == 1 else nn.Linear(config.in_chans, config.n_embd)
        self.patch_size = config.patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))

        self.pos_embed = nn.Embedding(256, config.n_embd)
        self.time_embed = nn.Embedding(512, config.n_embd)

        self.rel_pos_bias = None

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.n_embd, eps=1e-6)
        self.fc_norm = nn.LayerNorm(config.n_embd, eps=1e-6) if config.use_mean_pooling else None
        self.head = nn.Linear(config.n_embd, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.pos_drop = nn.Dropout(p=config.dropout)

        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(config.init_scale)
            self.head.bias.data.mul_(config.init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.c_proj.weight.data, layer_id + 1)
            rescale(layer.mlp.w2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, input_chans=None, input_times=None, bool_masked_pos=None, mask=None, return_all_tokens=False, **kwargs):
        batch_size, n, t = x.shape
        x = self.patch_embed(x)

        if bool_masked_pos is not None:
            mask_token = self.mask_token.expand(batch_size, n, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        # add position and temporal embeddings
        pos_embed_used = self.pos_embed(input_chans)
        x = x + pos_embed_used
        time_embed = self.time_embed(input_times)
        x = x + time_embed

        x = torch.cat((self.cls_token.expand(batch_size, -1, -1), x), dim=1)

        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x, mask)
        
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            else:
                return self.fc_norm(x.mean(1))
        else:
            if return_all_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, input_times=None, bool_masked_pos=None, mask=None, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, sequence length, patch size]
        '''
        x = self.forward_features(x, input_chans, input_times, bool_masked_pos, mask, return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x[bool_masked_pos])
        return x
    

# model for masked signal modeling
class MSM(nn.Module):
    def __init__(self, EEG_config, EOG_config, ECG_config, EMG_config, n_embed=8192, **kwargs):
        super().__init__()
        self.EEG_encoder = NeuralTransformer(EEG_config)
        self.EOG_encoder = NeuralTransformer(EOG_config)
        self.ECG_encoder = NeuralTransformer(ECG_config)
        self.EMG_encoder = NeuralTransformer(EMG_config)

        self.EEG_shared_lm_head = nn.Sequential(
            nn.Linear(EEG_config.n_embd, EEG_config.n_embd),
            nn.Tanh(),
            nn.Linear(EEG_config.n_embd, n_embed)
        )
        self.EEG_private_lm_head = nn.Sequential(
            nn.Linear(EEG_config.n_embd, EEG_config.n_embd),
            nn.Tanh(),
            nn.Linear(EEG_config.n_embd, n_embed)
        )
        self.EOG_shared_lm_head = nn.Sequential(
            nn.Linear(EOG_config.n_embd, EOG_config.n_embd),
            nn.Tanh(),
            nn.Linear(EOG_config.n_embd, n_embed)
        )
        self.EOG_private_lm_head = nn.Sequential(
            nn.Linear(EOG_config.n_embd, EOG_config.n_embd),
            nn.Tanh(),
            nn.Linear(EOG_config.n_embd, n_embed)
        )
        self.ECG_shared_lm_head = nn.Sequential(
            nn.Linear(ECG_config.n_embd, ECG_config.n_embd),
            nn.Tanh(),
            nn.Linear(ECG_config.n_embd, n_embed)
        )
        self.ECG_private_lm_head = nn.Sequential(
            nn.Linear(ECG_config.n_embd, ECG_config.n_embd),
            nn.Tanh(),
            nn.Linear(ECG_config.n_embd, n_embed)
        )
        self.EMG_shared_lm_head = nn.Sequential(
            nn.Linear(EMG_config.n_embd, EMG_config.n_embd),
            nn.Tanh(),
            nn.Linear(EMG_config.n_embd, n_embed)
        )
        self.EMG_private_lm_head = nn.Sequential(
            nn.Linear(EMG_config.n_embd, EMG_config.n_embd),
            nn.Tanh(),
            nn.Linear(EMG_config.n_embd, n_embed)
        )

        self.loss_fn = F.cross_entropy

        self.EEG_shared_lm_head.apply(self._init_weights)
        self.EEG_private_lm_head.apply(self._init_weights)
        self.EOG_shared_lm_head.apply(self._init_weights)
        self.EOG_private_lm_head.apply(self._init_weights)
        self.ECG_shared_lm_head.apply(self._init_weights)
        self.ECG_private_lm_head.apply(self._init_weights)
        self.EMG_shared_lm_head.apply(self._init_weights)
        self.EMG_private_lm_head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio, continuous=-1):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, Len, D = x.shape  # batch, length, dim
        L = Len
        if continuous != -1:
            L = int(L / continuous)
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if continuous != -1:
            mask = mask[:, :, None].expand(-1, -1, continuous).flatten(1)

        return mask.to(torch.bool)
    
    def cal_accuracy(self, logits, targets):
        _, preds = torch.max(logits.view(-1, logits.size(-1)), dim=-1)
        accuracy = torch.sum(preds == targets).float() / targets.size(0)
        return accuracy.item()

    def forward(self, batch, EEG_mask_ratio=0.5, EOG_mask_ratio=0.7, ECG_mask_ratio=0.7, EMG_mask_ratio=0.5, mask=None, return_all_tokens=True):
        EEG_X = batch['EEG_X']; EOG_X = batch['EOG_X']; ECG_X = batch['ECG_X']; EMG_X = batch['EMG_X']
        EEG_bool_masked_pos = self.random_masking(EEG_X, mask_ratio=EEG_mask_ratio) if EEG_X is not None else None
        EOG_bool_masked_pos = self.random_masking(EOG_X, mask_ratio=EOG_mask_ratio, continuous=-1) if EOG_X is not None else None
        ECG_bool_masked_pos = self.random_masking(ECG_X, mask_ratio=ECG_mask_ratio, continuous=-1) if ECG_X is not None else None
        EMG_bool_masked_pos = self.random_masking(EMG_X, mask_ratio=EMG_mask_ratio, continuous=-1) if EMG_X is not None else None

        EEG_shared_Y = batch['EEG_shared_codebook_indices'][EEG_bool_masked_pos.flatten()] if EEG_X is not None else None
        EEG_private_Y = batch['EEG_private_codebook_indices'][EEG_bool_masked_pos.flatten()] if EEG_X is not None else None
        EOG_shared_Y = batch['EOG_shared_codebook_indices'][EOG_bool_masked_pos.flatten()] if EOG_X is not None else None
        EOG_private_Y = batch['EOG_private_codebook_indices'][EOG_bool_masked_pos.flatten()] if EOG_X is not None else None
        ECG_shared_Y = batch['ECG_shared_codebook_indices'][ECG_bool_masked_pos.flatten()] if ECG_X is not None else None
        ECG_private_Y = batch['ECG_private_codebook_indices'][ECG_bool_masked_pos.flatten()] if ECG_X is not None else None
        EMG_shared_Y = batch['EMG_shared_codebook_indices'][EMG_bool_masked_pos.flatten()] if EMG_X is not None else None
        EMG_private_Y = batch['EMG_private_codebook_indices'][EMG_bool_masked_pos.flatten()] if EMG_X is not None else None
        EEG_input_chans = batch['EEG_input_chans']; EEG_input_time = batch['EEG_input_time']
        EOG_input_chans = batch['EOG_input_chans']; EOG_input_time = batch['EOG_input_time']
        ECG_input_chans = batch['ECG_input_chans']; ECG_input_time = batch['ECG_input_time']
        EMG_input_chans = batch['EMG_input_chans']; EMG_input_time = batch['EMG_input_time']

        EEG_X = self.EEG_encoder(EEG_X, EEG_input_chans, EEG_input_time, EEG_bool_masked_pos, mask, return_all_tokens) if EEG_X is not None else None
        EOG_X = self.EOG_encoder(EOG_X, EOG_input_chans, EOG_input_time, EOG_bool_masked_pos, mask, return_all_tokens) if EOG_X is not None else None
        ECG_X = self.ECG_encoder(ECG_X, ECG_input_chans, ECG_input_time, ECG_bool_masked_pos, mask, return_all_tokens) if ECG_X is not None else None
        EMG_X = self.EMG_encoder(EMG_X, EMG_input_chans, EMG_input_time, EMG_bool_masked_pos, mask, return_all_tokens) if EMG_X is not None else None
        
        EEG_shared_logits = self.EEG_shared_lm_head(EEG_X) if EEG_X is not None else None
        EEG_private_logits = self.EEG_private_lm_head(EEG_X) if EEG_X is not None else None
        EOG_shared_logits = self.EOG_shared_lm_head(EOG_X) if EOG_X is not None else None
        EOG_private_logits = self.EOG_private_lm_head(EOG_X) if EOG_X is not None else None
        ECG_shared_logits = self.ECG_shared_lm_head(ECG_X) if ECG_X is not None else None
        ECG_private_logits = self.ECG_private_lm_head(ECG_X) if ECG_X is not None else None
        EMG_shared_logits = self.EMG_shared_lm_head(EMG_X) if EMG_X is not None else None
        EMG_private_logits = self.EMG_private_lm_head(EMG_X) if EMG_X is not None else None

        EEG_shared_loss = self.loss_fn(EEG_shared_logits.view(-1, EEG_shared_logits.size(-1)), EEG_shared_Y) if EEG_shared_logits is not None else 0
        EEG_private_loss = self.loss_fn(EEG_private_logits.view(-1, EEG_private_logits.size(-1)), EEG_private_Y) if EEG_private_logits is not None else 0
        EOG_shared_loss = self.loss_fn(EOG_shared_logits.view(-1, EOG_shared_logits.size(-1)), EOG_shared_Y) if EOG_shared_logits is not None else 0
        EOG_private_loss = self.loss_fn(EOG_private_logits.view(-1, EOG_private_logits.size(-1)), EOG_private_Y) if EOG_private_logits is not None else 0
        ECG_shared_loss = self.loss_fn(ECG_shared_logits.view(-1, ECG_shared_logits.size(-1)), ECG_shared_Y) if ECG_shared_logits is not None else 0
        ECG_private_loss = self.loss_fn(ECG_private_logits.view(-1, ECG_private_logits.size(-1)), ECG_private_Y) if ECG_private_logits is not None else 0
        EMG_shared_loss = self.loss_fn(EMG_shared_logits.view(-1, EMG_shared_logits.size(-1)), EMG_shared_Y) if EMG_shared_logits is not None else 0
        EMG_private_loss = self.loss_fn(EMG_private_logits.view(-1, EMG_private_logits.size(-1)), EMG_private_Y) if EMG_private_logits is not None else 0

        loss = EEG_shared_loss + EEG_private_loss + EOG_shared_loss + EOG_private_loss + ECG_shared_loss + ECG_private_loss + EMG_shared_loss + EMG_private_loss

        EEG_shared_accuracy = self.cal_accuracy(EEG_shared_logits, EEG_shared_Y) if EEG_shared_logits is not None else None
        EEG_private_accuracy = self.cal_accuracy(EEG_private_logits, EEG_private_Y) if EEG_private_logits is not None else None
        EOG_shared_accuracy = self.cal_accuracy(EOG_shared_logits, EOG_shared_Y) if EOG_shared_logits is not None else None
        EOG_private_accuracy = self.cal_accuracy(EOG_private_logits, EOG_private_Y) if EOG_private_logits is not None else None
        ECG_shared_accuracy = self.cal_accuracy(ECG_shared_logits, ECG_shared_Y) if ECG_shared_logits is not None else None
        ECG_private_accuracy = self.cal_accuracy(ECG_private_logits, ECG_private_Y) if ECG_private_logits is not None else None
        EMG_shared_accuracy = self.cal_accuracy(EMG_shared_logits, EMG_shared_Y) if EMG_shared_logits is not None else None
        EMG_private_accuracy = self.cal_accuracy(EMG_private_logits, EMG_private_Y) if EMG_private_logits is not None else None

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/total_loss'] = loss.item()
        if EEG_X is not None:
            log[f'{split}/EEG_shared_loss'] = EEG_shared_loss.item()
            log[f'{split}/EEG_private_loss'] = EEG_private_loss.item()
            log[f'{split}/EEG_shared_accuracy'] = EEG_shared_accuracy
            log[f'{split}/EEG_private_accuracy'] = EEG_private_accuracy
        if EOG_X is not None:
            log[f'{split}/EOG_shared_loss'] = EOG_shared_loss.item()
            log[f'{split}/EOG_private_loss'] = EOG_private_loss.item()
            log[f'{split}/EOG_shared_accuracy'] = EOG_shared_accuracy
            log[f'{split}/EOG_private_accuracy'] = EOG_private_accuracy
        if ECG_X is not None:
            log[f'{split}/ECG_shared_loss'] = ECG_shared_loss.item()
            log[f'{split}/ECG_private_loss'] = ECG_private_loss.item()
            log[f'{split}/ECG_shared_accuracy'] = ECG_shared_accuracy
            log[f'{split}/ECG_private_accuracy'] = ECG_private_accuracy
        if EMG_X is not None:
            log[f'{split}/EMG_shared_loss'] = EMG_shared_loss.item()
            log[f'{split}/EMG_private_loss'] = EMG_private_loss.item()
            log[f'{split}/EMG_shared_accuracy'] = EMG_shared_accuracy
            log[f'{split}/EMG_private_accuracy'] = EMG_private_accuracy

        return loss, log
    
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
