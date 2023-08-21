'''
Network Architecture : SeamFormer 
A shared encoder followed 
by two decoder branches allocated 
for scribble and binarisation task.

'''
import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.vit import Transformer

# Global settings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_of_gpus = torch.cuda.device_count()

class SeamFormer(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        patch_size =8):

        super().__init__()
        # extract hyperparameters and functions from the ViT encoder.
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        # pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        pixel_values_per_patch = patch_size * patch_size

        # Binary Decoder 
        self.enc_to_dec_bin = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token_bin = nn.Parameter(torch.randn(decoder_dim))
        self.decoder_bin = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb_bin = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels_bin = nn.Linear(decoder_dim, pixel_values_per_patch)

        # Scribble Decoder 
        self.enc_to_dec_scr = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token_scr = nn.Parameter(torch.randn(decoder_dim))
        self.decoder_scr = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb_scr = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels_scr = nn.Linear(decoder_dim, pixel_values_per_patch)


    def forward(self,img,gt_bin_img=None,gt_scr_img=None,criterion=None,strain=True,btrain=True,mode='train'):

        scribbleloss=None
        gt_scr_patches=None
        binaryloss=None
        gt_bin_patches=None

        # get patches and their number
        patches = self.to_patch(img)
        _, num_patches, *_ = patches.shape
        # project pixel patches to tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # encode tokens by the encoder
        encoded_tokens = self.encoder.transformer(tokens)

        if btrain:
            decoder_tokens_bin = self.enc_to_dec_bin(encoded_tokens)
            # decode tokens with decoder
            decoded_tokens_bin = self.decoder_bin(decoder_tokens_bin)
            # project tokens to pixels
            pred_pixel_values_bin = self.to_pixels_bin(decoded_tokens_bin)
            ## --- Focal Loss ---
            if mode == 'train':
                # calculate the loss with gt
                if gt_bin_img is not None:
                    gt_bin_patches = self.to_patch(gt_bin_img)
                binaryloss = criterion(pred_pixel_values_bin,gt_bin_patches)
                pt = torch.exp(-binaryloss) 
                binaryloss = ((1-pt)**2) * binaryloss 
                binaryloss = torch.mean(binaryloss)
                return binaryloss,gt_bin_patches,pred_pixel_values_bin
        
        if strain:
            decoder_tokens_scr = self.enc_to_dec_scr(encoded_tokens)
            # decode tokens with decoder
            decoded_tokens_scr = self.decoder_scr(decoder_tokens_scr)
            # project tokens to pixels
            pred_pixel_values_scr = self.to_pixels_scr(decoded_tokens_scr)
            ## --- Focal Loss ---
            if mode == 'train':
                # calculate the loss with gt
                if gt_scr_img is not None:
                    gt_scr_patches = self.to_patch(gt_scr_img)
                ## ---  Weighted BCE Loss ---
                scribbleloss = criterion(pred_pixel_values_scr,gt_scr_patches)
                pt = torch.exp(-scribbleloss) 
                scribbleloss = ((1-pt)**2) *scribbleloss
                scribbleloss = torch.mean(scribbleloss)
                return scribbleloss,gt_scr_patches,pred_pixel_values_scr
            
        # Sending out only the pixel values for patching.
        if mode=='test':
            return pred_pixel_values_bin,pred_pixel_values_scr