import torch
from vae import VAEWorld
from outcome_heads_train_reg import VAEWithHeads
import pandas as pd

LATENT_DIM  = 64
HIDDEN_DIM  = 1024

def encode_child(x_tensor):
    with torch.no_grad():
        mu, logvar = vae.encode(x_tensor.unsqueeze(0))
        z = vae.reparameterize(mu, logvar)
    return z

df_scaled = pd.read_csv('../data/combined_imputed_scaled_large_nolip.tsv', sep='\t', index_col=0)
INPUT_DIM   = df_scaled.shape[1]
vae = VAEWorld(INPUT_DIM, LATENT_DIM, HIDDEN_DIM)
vae.load_state_dict(torch.load('vae_world_large_mam_nolip_mse.pt', map_location='cpu'))
vae.eval()

heads = VAEWithHeads(vae)
heads.load_state_dict(torch.load('../models/vae_heads_reg_20250814_160250.pt', map_location='cpu'))
heads.eval()

child_id = 'L1001'


x0 = torch.tensor(df_scaled.loc[child_id].values, dtype=torch.float32)
z0 = encode_child(x0)



latent_dim = 1 # column index in the latent space
delta = 0.1 # perturbation size
perturbed_z = z0.clone()
perturbed_z[0, latent_dim] += delta


with torch.no_grad():
    preds = heads(perturbed_z)
    cog = preds['cognitive_score_52'].item()
    voc = preds['vocalisation_52'].item()
    wlz = preds['WLZ_WHZ_52'].item()  


print("cognitive score:", cog, "vocalisation score:", voc, "WLZ/WHZ score:", wlz)