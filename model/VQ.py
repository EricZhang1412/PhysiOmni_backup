import torch
from torch import nn
import torch.nn.functional as F
import inspect
from model.neural_transformer import NeuralTransformer
from model.norm_ema_quantizer import NormEMAVectorQuantizer
from model.transformer import CrossAttention
from einops import rearrange


class VQ(nn.Module):
    def __init__(self,
                 EEG_encoder_config,
                 EOG_encoder_config,
                 ECG_encoder_config,
                 EMG_encoder_config,
                 EEG_decoder_config,
                 EOG_decoder_config,
                 ECG_decoder_config,
                 EMG_decoder_config,
                 n_embed=8192,
                 embed_dim=64,
                 decay=0.99,
                 quantize_kmeans_init=True,
                 EEG_decoder_out_dim=200,
                 EOG_decoder_out_dim=100,
                 ECG_decoder_out_dim=100,
                 EMG_decoder_out_dim=100,
                 smooth_l1_loss = False,
                 orth_ratio=0.1,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)

        # encoder & decode params
        self.EEG_encoder = NeuralTransformer(EEG_encoder_config)
        self.EOG_encoder = NeuralTransformer(EOG_encoder_config)
        self.ECG_encoder = NeuralTransformer(ECG_encoder_config)
        self.EMG_encoder = NeuralTransformer(EMG_encoder_config)

        self.EEG_decoder = NeuralTransformer(EEG_decoder_config)
        self.EOG_decoder = NeuralTransformer(EOG_decoder_config)
        self.ECG_decoder = NeuralTransformer(ECG_decoder_config)
        self.EMG_decoder = NeuralTransformer(EMG_decoder_config)

        self.EOG_crsatt = CrossAttention(EOG_encoder_config, embed_dim, 8)
        self.ECG_crsatt = CrossAttention(ECG_encoder_config, embed_dim, 8)
        self.EMG_crsatt = CrossAttention(EMG_encoder_config, embed_dim, 8)
        self.EEG2EOG_crsatt = CrossAttention(EOG_decoder_config, embed_dim, 8)
        self.EEG2ECG_crsatt = CrossAttention(ECG_decoder_config, embed_dim, 8)
        self.EEG2EMG_crsatt = CrossAttention(EMG_decoder_config, embed_dim, 8)
        
        self.shared_quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )
        self.EEG_quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )
        self.EOG_quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )
        self.ECG_quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )
        self.EMG_quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )

        self.EEG_decoder_out_dim = EEG_decoder_out_dim
        self.EOG_decoder_out_dim = EOG_decoder_out_dim
        self.ECG_decoder_out_dim = ECG_decoder_out_dim
        self.EMG_decoder_out_dim = EMG_decoder_out_dim
        self.embed_dim = embed_dim
        self.EOG_n_merge = 2
        self.ECG_n_merge = 5
        self.EMG_n_merge = 5
        self.orth_ratio = orth_ratio

        # task layer
        self.EEG_encode_task_layer = nn.Sequential(
            nn.Linear(EEG_encoder_config.n_embd, EEG_encoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(EEG_encoder_config.n_embd, embed_dim * 2) # for quantize
        )
        self.EEG_decode_task_layer = nn.Sequential(
            nn.Linear(EEG_decoder_config.n_embd, EEG_decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(EEG_decoder_config.n_embd, self.EEG_decoder_out_dim),
        )

        self.EOG_encode_task_layer = nn.Sequential(
            nn.Linear(EOG_encoder_config.n_embd, EOG_encoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(EOG_encoder_config.n_embd, embed_dim * 2) # for quantize
        )
        self.EOG_decode_task_layer = nn.Sequential(
            nn.Linear(EOG_decoder_config.n_embd, EOG_decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(EOG_decoder_config.n_embd, self.EOG_decoder_out_dim),
        )

        self.ECG_encode_task_layer = nn.Sequential(
            nn.Linear(ECG_encoder_config.n_embd, ECG_encoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(ECG_encoder_config.n_embd, embed_dim * 2) # for quantize
        )
        self.ECG_decode_task_layer = nn.Sequential(
            nn.Linear(ECG_decoder_config.n_embd, ECG_decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(ECG_decoder_config.n_embd, self.ECG_decoder_out_dim),
        )

        self.EMG_encode_task_layer = nn.Sequential(
            nn.Linear(EMG_encoder_config.n_embd, EMG_encoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(EMG_encoder_config.n_embd, embed_dim * 2) # for quantize
        )
        self.EMG_decode_task_layer = nn.Sequential(
            nn.Linear(EMG_decoder_config.n_embd, EMG_decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(EMG_decoder_config.n_embd, self.EMG_decoder_out_dim),
        )

        self.kwargs = kwargs
        
        self.EEG_encode_task_layer.apply(self._init_weights)
        self.EEG_decode_task_layer.apply(self._init_weights)
        self.EOG_encode_task_layer.apply(self._init_weights)
        self.EOG_decode_task_layer.apply(self._init_weights)
        self.ECG_encode_task_layer.apply(self._init_weights)
        self.ECG_decode_task_layer.apply(self._init_weights)
        self.EMG_encode_task_layer.apply(self._init_weights)
        self.EMG_decode_task_layer.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'quantize.embedding.weight', 'decoder.pos_embed', 'decoder.time_embed', 
    #             'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return self.EEG_decoder.cls_token.device
        
    def get_number_of_tokens(self):
        return self.shared_quantize.n_e

    def get_tokens(self, EEG_X, EOG_X, ECG_X, EMG_X, 
                   EEG_input_chans=None, EEG_input_time=None,
                   EOG_input_chans=None, EOG_input_time=None,
                   ECG_input_chans=None, ECG_input_time=None,
                   EMG_input_chans=None, EMG_input_time=None):
        _, EEG_private_embed_ind, _, \
        _, EOG_private_embed_ind, _, \
        _, ECG_private_embed_ind, _, \
        _, EMG_private_embed_ind, _, \
        _, EEG_shared_embed_ind, _, \
        _, EOG_shared_embed_ind, _, \
        _, ECG_shared_embed_ind, _, \
        _, EMG_shared_embed_ind, _ = self.encode(EEG_X, EOG_X, ECG_X, EMG_X,
                                                 EEG_input_chans, EEG_input_time,
                                                 EOG_input_chans, EOG_input_time,
                                                 ECG_input_chans, ECG_input_time,
                                                 EMG_input_chans, EMG_input_time, cal_loss=False)

        if EOG_private_embed_ind is not None:
            EOG_expand_num = EOG_private_embed_ind.size(1) // EOG_shared_embed_ind.size(1)
            EOG_shared_embed_ind = EOG_shared_embed_ind.unsqueeze(2).expand(-1, -1, EOG_expand_num).reshape(EOG_private_embed_ind.size())

        if ECG_private_embed_ind is not None:
            ECG_expand_num = ECG_private_embed_ind.size(1) // ECG_shared_embed_ind.size(1)
            ECG_shared_embed_ind = ECG_shared_embed_ind.unsqueeze(2).expand(-1, -1, ECG_expand_num).reshape(ECG_private_embed_ind.size())

        if EMG_private_embed_ind is not None:
            EMG_expand_num = EMG_private_embed_ind.size(1) // EMG_shared_embed_ind.size(1)
            EMG_shared_embed_ind = EMG_shared_embed_ind.unsqueeze(2).expand(-1, -1, EMG_expand_num).reshape(EMG_private_embed_ind.size())

        return EEG_private_embed_ind, EOG_private_embed_ind, ECG_private_embed_ind, EMG_private_embed_ind, \
            EEG_shared_embed_ind, EOG_shared_embed_ind, ECG_shared_embed_ind, EMG_shared_embed_ind

    def encode(self, EEG_X, EOG_X, ECG_X, EMG_X,
               EEG_input_chans=None, EEG_input_time=None,
               EOG_input_chans=None, EOG_input_time=None,
               ECG_input_chans=None, ECG_input_time=None,
               EMG_input_chans=None, EMG_input_time=None,
               cal_loss=True):
        EEG_encoder_features = self.EEG_encoder(EEG_X, EEG_input_chans, EEG_input_time, return_all_tokens=True) if EEG_X is not None else None
        EOG_encoder_features = self.EOG_encoder(EOG_X, EOG_input_chans, EOG_input_time, return_all_tokens=True) if EOG_X is not None else None
        ECG_encoder_features = self.ECG_encoder(ECG_X, ECG_input_chans, ECG_input_time, return_all_tokens=True) if ECG_X is not None else None
        EMG_encoder_features = self.EMG_encoder(EMG_X, EMG_input_chans, EMG_input_time, return_all_tokens=True) if EMG_X is not None else None

        with torch.amp.autocast('cuda', enabled=False):
            EEG_to_quantizer_features = self.EEG_encode_task_layer(EEG_encoder_features.type_as(self.EEG_encode_task_layer[-1].weight)) if EEG_encoder_features is not None else None
            EOG_to_quantizer_features = self.EOG_encode_task_layer(EOG_encoder_features.type_as(self.EOG_encode_task_layer[-1].weight)) if EOG_encoder_features is not None else None
            ECG_to_quantizer_features = self.ECG_encode_task_layer(ECG_encoder_features.type_as(self.ECG_encode_task_layer[-1].weight)) if ECG_encoder_features is not None else None
            EMG_to_quantizer_features = self.EMG_encode_task_layer(EMG_encoder_features.type_as(self.EMG_encode_task_layer[-1].weight)) if EMG_encoder_features is not None else None

            EEG_private_features, EEG_shared_features = EEG_to_quantizer_features.split(self.embed_dim, dim=2) if EEG_to_quantizer_features is not None else (None, None)
            EOG_private_features, EOG_shared_features = EOG_to_quantizer_features.split(self.embed_dim, dim=2) if EOG_to_quantizer_features is not None else (None, None)
            ECG_private_features, ECG_shared_features = ECG_to_quantizer_features.split(self.embed_dim, dim=2) if ECG_to_quantizer_features is not None else (None, None)
            EMG_private_features, EMG_shared_features = EMG_to_quantizer_features.split(self.embed_dim, dim=2) if EMG_to_quantizer_features is not None else (None, None)

            EOG_merged_features = self.EOG_crsatt(EOG_shared_features, self.EOG_n_merge) if EOG_shared_features is not None else None
            ECG_merged_features = self.ECG_crsatt(ECG_shared_features, self.ECG_n_merge) if ECG_shared_features is not None else None
            EMG_merged_features = self.EMG_crsatt(EMG_shared_features, self.EMG_n_merge) if EMG_shared_features is not None else None

        if cal_loss:
            EEG_orth_loss = self.calculate_orth_loss(EEG_private_features, EEG_shared_features) if EEG_shared_features is not None else None
            EOG_orth_loss = self.calculate_orth_loss(EOG_private_features, EOG_shared_features) if EOG_shared_features is not None else None
            ECG_orth_loss = self.calculate_orth_loss(ECG_private_features, ECG_shared_features) if ECG_shared_features is not None else None
            EMG_orth_loss = self.calculate_orth_loss(EMG_private_features, EMG_shared_features) if EMG_shared_features is not None else None

        EEG_private_quantize, EEG_private_loss, EEG_private_embed_ind = self.EEG_quantize(EEG_private_features) if EEG_private_features is not None else (None, None, None)
        EOG_private_quantize, EOG_private_loss, EOG_private_embed_ind = self.EOG_quantize(EOG_private_features) if EOG_private_features is not None else (None, None, None)
        ECG_private_quantize, ECG_private_loss, ECG_private_embed_ind = self.ECG_quantize(ECG_private_features) if ECG_private_features is not None else (None, None, None)
        EMG_private_quantize, EMG_private_loss, EMG_private_embed_ind = self.EMG_quantize(EMG_private_features) if EMG_private_features is not None else (None, None, None)
        EEG_shared_quantize, EEG_shared_loss, EEG_shared_embed_ind = self.shared_quantize(EEG_shared_features) if EEG_shared_features is not None else (None, None, None)
        EOG_shared_quantize, EOG_shared_loss, EOG_shared_embed_ind = self.shared_quantize(EOG_merged_features) if EOG_merged_features is not None else (None, None, None)
        ECG_shared_quantize, ECG_shared_loss, ECG_shared_embed_ind = self.shared_quantize(ECG_merged_features) if ECG_merged_features is not None else (None, None, None)
        EMG_shared_quantize, EMG_shared_loss, EMG_shared_embed_ind = self.shared_quantize(EMG_merged_features) if EMG_merged_features is not None else (None, None, None)

        if cal_loss:
            return EEG_private_quantize, EEG_private_embed_ind, EEG_private_loss, \
                EOG_private_quantize, EOG_private_embed_ind, EOG_private_loss, \
                ECG_private_quantize, ECG_private_embed_ind, ECG_private_loss, \
                EMG_private_quantize, EMG_private_embed_ind, EMG_private_loss, \
                EEG_shared_quantize, EEG_shared_embed_ind, EEG_shared_loss, \
                EOG_shared_quantize, EOG_shared_embed_ind, EOG_shared_loss, \
                ECG_shared_quantize, ECG_shared_embed_ind, ECG_shared_loss, \
                EMG_shared_quantize, EMG_shared_embed_ind, EMG_shared_loss, \
                EEG_orth_loss, EOG_orth_loss, ECG_orth_loss, EMG_orth_loss

        return EEG_private_quantize, EEG_private_embed_ind, EEG_private_loss, \
            EOG_private_quantize, EOG_private_embed_ind, EOG_private_loss, \
            ECG_private_quantize, ECG_private_embed_ind, ECG_private_loss, \
            EMG_private_quantize, EMG_private_embed_ind, EMG_private_loss, \
            EEG_shared_quantize, EEG_shared_embed_ind, EEG_shared_loss, \
            EOG_shared_quantize, EOG_shared_embed_ind, EOG_shared_loss, \
            ECG_shared_quantize, ECG_shared_embed_ind, ECG_shared_loss, \
            EMG_shared_quantize, EMG_shared_embed_ind, EMG_shared_loss
        
    def decode(self, EEG_private_quantize, EOG_private_quantize, ECG_private_quantize, EMG_private_quantize,
               EEG_shared_quantize, EOG_shared_quantize, ECG_shared_quantize, EMG_shared_quantize, 
               EEG_input_chans, EEG_input_time, EOG_input_chans, EOG_input_time, 
               ECG_input_chans, ECG_input_time, EMG_input_chans, EMG_input_time):
        EEG_rec, EOG_rec, ECG_rec, EMG_rec, EEG2EOG_rec, EEG2ECG_rec, EEG2EMG_rec = None, None, None, None, None, None, None
        # reshape tokens to feature maps for patch embed in decoder
        if EEG_private_quantize is not None:
            EEG_quantize = torch.cat((EEG_private_quantize, EEG_shared_quantize), dim=-1)
            EEG_decoder_features = self.EEG_decoder(EEG_quantize, EEG_input_chans, EEG_input_time, return_all_tokens=True)
            EEG_rec = self.EEG_decode_task_layer(EEG_decoder_features)

        if EOG_private_quantize is not None:
            EOG_expand_num = EOG_private_quantize.size(1) // EOG_shared_quantize.size(1)
            EOG_shared_quantize = EOG_shared_quantize.unsqueeze(2).expand(-1, -1, EOG_expand_num, -1).reshape(EOG_private_quantize.size())
            EOG_quantize = torch.cat((EOG_private_quantize, EOG_shared_quantize), dim=-1)
            EOG_decoder_features = self.EOG_decoder(EOG_quantize, EOG_input_chans, EOG_input_time, return_all_tokens=True)
            EOG_rec = self.EOG_decode_task_layer(EOG_decoder_features)

        if ECG_private_quantize is not None:
            ECG_expand_num = ECG_private_quantize.size(1) // ECG_shared_quantize.size(1)
            ECG_shared_quantize = ECG_shared_quantize.unsqueeze(2).expand(-1, -1, ECG_expand_num, -1).reshape(ECG_private_quantize.size())
            ECG_quantize = torch.cat((ECG_private_quantize, ECG_shared_quantize), dim=-1)
            ECG_decoder_features = self.ECG_decoder(ECG_quantize, ECG_input_chans, ECG_input_time, return_all_tokens=True)
            ECG_rec = self.EOG_decode_task_layer(ECG_decoder_features)

        if EMG_private_quantize is not None:
            EMG_expand_num = EMG_private_quantize.size(1) // EMG_shared_quantize.size(1)
            EMG_shared_quantize = EMG_shared_quantize.unsqueeze(2).expand(-1, -1, EMG_expand_num, -1).reshape(EMG_private_quantize.size())
            EMG_quantize = torch.cat((EMG_private_quantize, EMG_shared_quantize), dim=-1)
            EMG_decoder_features = self.EMG_decoder(EMG_quantize, EMG_input_chans, EMG_input_time, return_all_tokens=True)
            EMG_rec = self.EMG_decode_task_layer(EMG_decoder_features)

        if EOG_shared_quantize is not None:
            num_channels = EOG_private_quantize.size(1) // (EEG_input_time[-1, -1] + 1) // self.EOG_n_merge
            query = rearrange(EOG_private_quantize, 'B (N T A) C -> B N T A C', N=num_channels, A=self.EOG_n_merge).mean(dim=-2)
            query = rearrange(query, 'B N T C -> (B T) N C')
            EEG2EOG_shared_quantize = self.EEG2EOG_crsatt(EEG_shared_quantize, time=EEG_input_time[-1, -1] + 1, query=query)
            EEG2EOG_shared_quantize = EEG2EOG_shared_quantize.unsqueeze(1).expand(-1, EOG_expand_num, -1, -1).reshape(EOG_private_quantize.size())
            EEG2EOG_quantize = torch.cat((EOG_private_quantize, EEG2EOG_shared_quantize), dim=-1)
            EEG2EOG_decoder_features = self.EOG_decoder(EEG2EOG_quantize, EOG_input_chans, EOG_input_time, return_all_tokens=True)
            EEG2EOG_rec = self.EOG_decode_task_layer(EEG2EOG_decoder_features)

        if ECG_shared_quantize is not None:
            num_channels = ECG_private_quantize.size(1) // (EEG_input_time[-1, -1] + 1) // self.ECG_n_merge
            query = rearrange(ECG_private_quantize, 'B (N T A) C -> B N T A C', N=num_channels, A=self.ECG_n_merge).mean(dim=-2)
            query = rearrange(query, 'B N T C -> (B T) N C')
            EEG2ECG_shared_quantize = self.EEG2ECG_crsatt(EEG_shared_quantize, time=EEG_input_time[-1, -1] + 1, query=query)
            EEG2ECG_shared_quantize = EEG2ECG_shared_quantize.unsqueeze(1).expand(-1, ECG_expand_num, -1, -1).reshape(ECG_private_quantize.size())
            EEG2ECG_quantize = torch.cat((ECG_private_quantize, EEG2ECG_shared_quantize), dim=-1)
            EEG2ECG_decoder_features = self.ECG_decoder(EEG2ECG_quantize, ECG_input_chans, ECG_input_time, return_all_tokens=True)
            EEG2ECG_rec = self.ECG_decode_task_layer(EEG2ECG_decoder_features)

        if EMG_shared_quantize is not None:
            num_channels = EMG_private_quantize.size(1) // (EEG_input_time[-1, -1] + 1) // self.EMG_n_merge
            query = rearrange(EMG_private_quantize, 'B (N T A) C -> B N T A C', N=num_channels, A=self.EMG_n_merge).mean(dim=-2)
            query = rearrange(query, 'B N T C -> (B T) N C')
            EEG2EMG_shared_quantize = self.EEG2EMG_crsatt(EEG_shared_quantize, time=EEG_input_time[-1, -1] + 1, query=query)
            EEG2EMG_shared_quantize = EEG2EMG_shared_quantize.unsqueeze(1).expand(-1, EMG_expand_num, -1, -1).reshape(EMG_private_quantize.size())
            EEG2EMG_quantize = torch.cat((EMG_private_quantize, EEG2EMG_shared_quantize), dim=-1)
            EEG2EMG_decoder_features = self.EMG_decoder(EEG2EMG_quantize, EMG_input_chans, EMG_input_time, return_all_tokens=True)
            EEG2EMG_rec = self.EMG_decode_task_layer(EEG2EMG_decoder_features)

        return EEG_rec, EOG_rec, ECG_rec, EMG_rec, EEG2EOG_rec, EEG2ECG_rec, EEG2EMG_rec
    
    def get_codebook_indices(self, batch):
        EEG_X = batch['EEG_X']; EOG_X = batch['EOG_X']; ECG_X = batch['ECG_X']; EMG_X = batch['EMG_X']
        EEG_input_chans = batch['EEG_input_chans']; EEG_input_time = batch['EEG_input_time']
        EOG_input_chans = batch['EOG_input_chans']; EOG_input_time = batch['EOG_input_time']
        ECG_input_chans = batch['ECG_input_chans']; ECG_input_time = batch['ECG_input_time']
        EMG_input_chans = batch['EMG_input_chans']; EMG_input_time = batch['EMG_input_time']

        return self.get_tokens(EEG_X, EOG_X, ECG_X, EMG_X,
                               EEG_input_chans, EEG_input_time,
                               EOG_input_chans, EOG_input_time,
                               ECG_input_chans, ECG_input_time,
                               EMG_input_chans, EMG_input_time)
    
    def calculate_rec_loss(self, rec, target):
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def calculate_orth_loss(self, input1, input2):
        orth_loss = F.cosine_embedding_loss(input1.view(-1, input1.size(-1)), input2.view(-1, input2.size(-1)), target=torch.tensor([-1]).to(input1.device)).mean(0)
        return orth_loss

    def forward(self, batch):
        """
        x: shape [B, N, T]
        """
        EEG_X = batch['EEG_X']; EOG_X = batch['EOG_X']; ECG_X = batch['ECG_X']; EMG_X = batch['EMG_X']
        EEG_Y = batch['EEG_Y']; EOG_Y = batch['EOG_Y']; ECG_Y = batch['ECG_Y']; EMG_Y = batch['EMG_Y']
        EEG_input_chans = batch['EEG_input_chans']; EEG_input_time = batch['EEG_input_time']
        EOG_input_chans = batch['EOG_input_chans']; EOG_input_time = batch['EOG_input_time']
        ECG_input_chans = batch['ECG_input_chans']; ECG_input_time = batch['ECG_input_time']
        EMG_input_chans = batch['EMG_input_chans']; EMG_input_time = batch['EMG_input_time']
        
        EEG_private_quantize, _, EEG_private_loss, \
        EOG_private_quantize, _, EOG_private_loss, \
        ECG_private_quantize, _, ECG_private_loss, \
        EMG_private_quantize, _, EMG_private_loss, \
        EEG_shared_quantize, _, EEG_shared_loss, \
        EOG_shared_quantize, _, EOG_shared_loss, \
        ECG_shared_quantize, _, ECG_shared_loss, \
        EMG_shared_quantize, _, EMG_shared_loss, \
        EEG_orth_loss, EOG_orth_loss, ECG_orth_loss, EMG_orth_loss = self.encode(EEG_X, EOG_X, ECG_X, EMG_X,
                                                                                 EEG_input_chans, EEG_input_time,
                                                                                 EOG_input_chans, EOG_input_time,
                                                                                 ECG_input_chans, ECG_input_time,
                                                                                 EMG_input_chans, EMG_input_time)
        
        EEG_rec, EOG_rec, ECG_rec, EMG_rec, \
        EEG2EOG_rec, EEG2ECG_rec, EEG2EMG_rec = self.decode(EEG_private_quantize, EOG_private_quantize, ECG_private_quantize, EMG_private_quantize,
                                                            EEG_shared_quantize, EOG_shared_quantize, ECG_shared_quantize, EMG_shared_quantize,
                                                            EEG_input_chans, EEG_input_time, EOG_input_chans, EOG_input_time, 
                                                            ECG_input_chans, ECG_input_time, EMG_input_chans, EMG_input_time)

        EEG_rec_loss = self.calculate_rec_loss(EEG_rec, EEG_Y) if EEG_rec is not None else 0
        EOG_rec_loss = self.calculate_rec_loss(EOG_rec, EOG_Y) if EOG_rec is not None else 0
        ECG_rec_loss = self.calculate_rec_loss(ECG_rec, ECG_Y) if ECG_rec is not None else 0
        EMG_rec_loss = self.calculate_rec_loss(EMG_rec, EMG_Y) if EMG_rec is not None else 0
        EEG2EOG_rec_loss = self.calculate_rec_loss(EEG2EOG_rec, EOG_Y) if EEG2EOG_rec is not None else 0
        EEG2ECG_rec_loss = self.calculate_rec_loss(EEG2ECG_rec, ECG_Y) if EEG2ECG_rec is not None else 0
        EEG2EMG_rec_loss = self.calculate_rec_loss(EEG2EMG_rec, EMG_Y) if EEG2EMG_rec is not None else 0
        if EEG_private_loss is None:
            EEG_private_loss = 0
        if EOG_private_loss is None:
            EOG_private_loss = 0
        if ECG_private_loss is None:
            ECG_private_loss = 0
        if EMG_private_loss is None:
            EMG_private_loss = 0
        if EEG_shared_loss is None:
            EEG_shared_loss = 0
        if EOG_shared_loss is None:
            EOG_shared_loss = 0
        if ECG_shared_loss is None:
            ECG_shared_loss = 0
        if EMG_shared_loss is None:
            EMG_shared_loss = 0
        if EEG_orth_loss is None:
            EEG_orth_loss = 0
        if EOG_orth_loss is None:
            EOG_orth_loss = 0
        if ECG_orth_loss is None:
            ECG_orth_loss = 0
        if EMG_orth_loss is None:
            EMG_orth_loss = 0

        loss = (EEG_private_loss + EOG_private_loss + ECG_private_loss + EMG_private_loss + \
                EEG_shared_loss + EOG_shared_loss + ECG_shared_loss + EMG_shared_loss) + \
                (EEG_rec_loss + EOG_rec_loss + ECG_rec_loss + EEG2EOG_rec_loss + EEG2ECG_rec_loss + EEG2EMG_rec_loss) + \
                (EEG_orth_loss + EOG_orth_loss + ECG_orth_loss + EMG_orth_loss) * self.orth_ratio

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/total_loss'] = loss.item()
        if EEG_rec is not None:
            log[f'{split}/EEG_rec_loss'] = EEG_rec_loss.item()
            log[f'{split}/EEG_orth_loss'] = EEG_orth_loss.item()
        if EOG_rec is not None:
            log[f'{split}/EOG_rec_loss'] = EOG_rec_loss.item()
            log[f'{split}/EOG_orth_loss'] = EOG_orth_loss.item()
        if ECG_rec is not None:
            log[f'{split}/ECG_rec_loss'] = ECG_rec_loss.item()
            log[f'{split}/ECG_orth_loss'] = ECG_orth_loss.item()
        if EMG_rec is not None:
            log[f'{split}/EMG_rec_loss'] = EMG_rec_loss.item()
            log[f'{split}/EMG_orth_loss'] = EMG_orth_loss.item()
        if EEG2EOG_rec is not None:
            log[f'{split}/EEG2EOG_rec_loss'] = EEG2EOG_rec_loss.item()
        if EEG2ECG_rec is not None:
            log[f'{split}/EEG2ECG_rec_loss'] = EEG2ECG_rec_loss.item()
        if EEG2EMG_rec is not None:
            log[f'{split}/EEG2EMG_rec_loss'] = EEG2EMG_rec_loss.item()
        log[f'{split}/quant_loss'] = (EEG_private_loss + EOG_private_loss + ECG_private_loss + \
                                      EEG_shared_loss + EOG_shared_loss + ECG_shared_loss).item()
        
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
