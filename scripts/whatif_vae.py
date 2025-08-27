#!/usr/bin/env python3
"""
what_if_vae.py
Simple counter-factual engine for the VAE + regression heads
"""

import torch
import pandas as pd
import numpy as np
from vae import VAEWorld
from outcome_heads_train_reg import VAEWithHeads

LATENT_DIM  = 64
HIDDEN_DIM  = 1024

df_scaled = pd.read_csv('../data/combined_imputed_scaled_large_nolip.tsv', sep='\t', index_col=0)

INPUT_DIM = df_scaled.shape[1]

vae = VAEWorld(INPUT_DIM, LATENT_DIM, HIDDEN_DIM)
vae.load_state_dict(torch.load('vae_world_large_mam_nolip_mse.pt', map_location='cpu'))
vae.eval()

heads = VAEWithHeads(vae)
checkpoint = torch.load('../models/vae_heads_nolip_mse_onecyclelr.pt', map_location='cpu')
heads.load_state_dict(checkpoint['state_dict'], strict=False)
heads.eval()

def encode_child(x_tensor):
    """Return latent vector (1, LATENT_DIM)"""
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.view(1, -1)  # Reshape to [1, 232]
    with torch.no_grad():
        mu, logvar = vae.encode(x_tensor)
        z = vae.reparameterize(mu, logvar)
    return z

def predict_from_latent(z):
    """Return dict of predictions given latent (1, LATENT_DIM)"""
    with torch.no_grad():
        # Use the regression heads directly instead of the full VAE
        preds = {
            k: head(z) for k, head in heads.reg_heads.items()
        }
    return {
        'cognitive_score_52': preds['cognitive_score_52'].item(),
        'vocalisation_52': preds['vocalisation_52'].item(),
        'WLZ_WHZ_52': preds['WLZ_WHZ_52'].item()
    }

def what_if(child_id: str, col: str, delta: float):
    """
    Perturb one column for one child and return original vs new predictions.
    """
    row_orig = df_scaled.loc[child_id].copy()
    x_orig   = torch.tensor(row_orig.values, dtype=torch.float32)

    # Original predictions
    z_orig = encode_child(x_orig)
    pred_orig = predict_from_latent(z_orig)

    # Perturbed row
    row_new = row_orig.copy()
    row_new[col] += delta
    x_new = torch.tensor(row_new.values, dtype=torch.float32)
    z_new = encode_child(x_new)
    pred_new = predict_from_latent(z_new)

    return {
        'child': child_id,
        'column': col,
        'delta': delta,
        'original': pred_orig,
        'perturbed': pred_new,
        'change': {k: pred_new[k] - pred_orig[k] for k in pred_orig}
    }

def population_curve(col: str, delta_grid: np.ndarray):
    """
    Sweep a delta across the whole cohort and return the average change.
    """
    results = []
    for d in delta_grid:
        rows = df_scaled.copy()
        rows[col] += d
        zs = []
        for idx in rows.index:
            x = torch.tensor(rows.loc[idx].values, dtype=torch.float32)
            zs.append(encode_child(x))
        zs = torch.cat(zs, dim=0)  # (N, 64)

        with torch.no_grad():
            # Use regression heads directly instead of full VAE
            preds = {
                k: head(zs) for k, head in heads.reg_heads.items()
            }
            
        mean_cog = preds['cognitive_score_52'].mean().item()
        mean_voc = preds['vocalisation_52'].mean().item()
        mean_wlz = preds['WLZ_WHZ_52'].mean().item()
        
        results.append({
            'delta': d, 
            'cognitive': mean_cog,
            'vocalisation': mean_voc, 
            'WLZ_WHZ': mean_wlz
        })
    return pd.DataFrame(results)

def population_fixed_values(col: str, value_grid: np.ndarray):
    """
    Set a column to fixed values across the whole cohort and return the average outcomes.
    
    Args:
        col (str): Column name to modify
        value_grid (np.ndarray): Array of values to set the column to
    
    Returns:
        pd.DataFrame: Results with fixed values and predicted outcomes
    """
    results = []
    for val in value_grid:
        rows = df_scaled.copy()
        rows[col] = val  # Set to absolute value instead of adding delta
        zs = []
        for idx in rows.index:
            x = torch.tensor(rows.loc[idx].values, dtype=torch.float32)
            zs.append(encode_child(x))
        zs = torch.cat(zs, dim=0)  # (N, 64)

        with torch.no_grad():
            preds = {
                k: head(zs) for k, head in heads.reg_heads.items()
            }
            
        mean_cog = preds['cognitive_score_52'].mean().item()
        mean_voc = preds['vocalisation_52'].mean().item()
        mean_wlz = preds['WLZ_WHZ_52'].mean().item()
        
        results.append({
            'value': val,  # Changed from 'delta' to 'value'
            'cognitive': mean_cog,
            'vocalisation': mean_voc, 
            'WLZ_WHZ': mean_wlz
        })
    return pd.DataFrame(results)

def population_fixed_values_dummy(col_values: dict):
    """
    Set multiple columns to fixed values across the whole cohort.
    
    Args:
        col_values (dict): Dictionary mapping column names to their desired fixed values
                          e.g., {'Feed_ERUSF': 1, 'Feed_Local RUSF': 0}
    
    Returns:
        dict: Results with predicted outcomes
    """
    rows = df_scaled.copy()
    # Set all specified columns to their fixed values
    for col, val in col_values.items():
        rows[col] = val
        
    zs = []
    for idx in rows.index:
        x = torch.tensor(rows.loc[idx].values, dtype=torch.float32)
        zs.append(encode_child(x))
    zs = torch.cat(zs, dim=0)  # (N, 64)

    with torch.no_grad():
        preds = {
            k: head(zs) for k, head in heads.reg_heads.items()
        }
        
    return {
        'cognitive': preds['cognitive_score_52'].mean().item(),
        'vocalisation': preds['vocalisation_52'].mean().item(),
        'WLZ_WHZ': preds['WLZ_WHZ_52'].mean().item()
    }

if __name__ == '__main__':
    # # single child
    # print(what_if('LCC1001', 'Weight', 1.0))

    # # population sweep assuming everyone can improve by delta
    curve = population_curve('Weight', np.linspace(-2, 2, 21))
    curve.to_csv('../Outcomes/whatif_population_weight.tsv', sep='\t', index=False)

    values = np.linspace(-2, 2, 21)  # Create range of values in standard deviation units
    curve = population_fixed_values('Weight', values)
    curve.to_csv('../Outcomes/whatif_population_weight_fixed.tsv', sep='\t', index=False)

    # Example usage
    result_ERUSF = population_fixed_values_dummy({
        'Feed_ERUSF (B)': 1, 
        'Feed_Local RUSF (A)': 0
    })

    result_LocalRUSF = population_fixed_values_dummy({
        'Feed_ERUSF (B)': 0, 
        'Feed_Local RUSF (A)': 1
    })

    print("Feed ERUSF", result_ERUSF)
    print("Feed Local RUSF", result_LocalRUSF)