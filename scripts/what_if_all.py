#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
from vae import VAEWorld
from outcome_heads_train_reg import VAEWithHeads
import joblib

LATENT_DIM  = 64
HIDDEN_DIM  = 1024
BATCH_SIZE  = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load inputs (already z-scored)
df_scaled = pd.read_csv('../data/combined_imputed_scaled_large_nolip.tsv', sep='\t', index_col=0)

# Match the training cohort (MAM only), same as outcome_heads_train_reg.py
meta_data = pd.read_csv('../data/meta.tsv', sep='\t')
mam_ids = meta_data[meta_data['Condition'] == 'MAM']['subjectID']
df_scaled = df_scaled.loc[df_scaled.index.intersection(mam_ids)]

INPUT_DIM = df_scaled.shape[1]

# Load VAE + heads
vae = VAEWorld(INPUT_DIM, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
vae.load_state_dict(torch.load('vae_world_large_mam_nolip_mse.pt', map_location=DEVICE))
vae.eval()

heads = VAEWithHeads(vae).to(DEVICE)
ckpt = torch.load('../models/vae_heads_nolip_mse_onecyclelr.pt', map_location=DEVICE)
heads.load_state_dict(ckpt['state_dict'], strict=False)
heads.eval()

# Optional: target scaler for inverse-transform (if present)
target_scaler = ckpt.get('target_scaler', None)
regression_cols = ckpt.get('regression_cols', ['cognitive_score_52', 'vocalisation_52', 'WLZ_WHZ_52'])


# Detect dummy (one-hot) columns to help enforce valid interventions
def detect_dummy_cols(df: pd.DataFrame):
    # Binary 0/1 columns
    dummy_cols = [
        col for col in df.columns
        if df[col].dropna().isin([0, 1]).all()
    ]
    return dummy_cols

DUMMY_COLS = detect_dummy_cols(df_scaled)

def group_onehots_by_prefix(cols, prefix):
    # All one-hot columns that start with prefix_
    return [c for c in cols if c.startswith(prefix)]

# Encode a whole DataFrame in batches
@torch.no_grad()
def encode_df(df: pd.DataFrame) -> torch.Tensor:
    zs = []
    vae.eval()
    for start in range(0, len(df), BATCH_SIZE):
        end = start + BATCH_SIZE
        x = torch.tensor(df.iloc[start:end].values, dtype=torch.float32, device=DEVICE)
        mu, logvar = vae.encode(x)
        z = mu  # deterministic: use mean only
        zs.append(z)
    return torch.cat(zs, dim=0)

# Predict from latent (batched)
@torch.no_grad()
def predict_from_latent(z: torch.Tensor):
    preds = {k: head(z) for k, head in heads.reg_heads.items()}
    # Stack to DataFrame
    out = pd.DataFrame({
        'cognitive_score_52': preds['cognitive_score_52'].squeeze(1).cpu().numpy(),
        'vocalisation_52':    preds['vocalisation_52'].squeeze(1).cpu().numpy(),
        'WLZ_WHZ_52':         preds['WLZ_WHZ_52'].squeeze(1).cpu().numpy(),
    }, index=df_scaled.index)
    return out

def inverse_transform_targets(df_preds_scaled: pd.DataFrame) -> pd.DataFrame:
    if target_scaler is None:
        return df_preds_scaled
    # keep column order
    cols = ['cognitive_score_52', 'vocalisation_52', 'WLZ_WHZ_52']
    arr_inv = target_scaler.inverse_transform(df_preds_scaled[cols].values)
    return pd.DataFrame(arr_inv, index=df_preds_scaled.index, columns=cols)

def predict_df(df: pd.DataFrame):
    z = encode_df(df)
    preds_scaled = predict_from_latent(z)
    preds_inv = inverse_transform_targets(preds_scaled)
    return preds_scaled, preds_inv

def apply_intervention(base_df: pd.DataFrame,
                       add: dict | None = None,
                       set_: dict | None = None,
                       onehot_sets: dict | None = None) -> pd.DataFrame:
    """
    add:  dict of {col: delta_in_scaled_units} to add to existing values
    set_: dict of {col: absolute_value_in_scaled_units} to set
    onehot_sets: dict of {prefix: chosen_col_name} to enforce one-hot groups
    Returns new DataFrame (copy).
    """
    df = base_df.copy()
    # continuous add/set
    if add:
        for c, d in add.items():
            if c in df.columns:
                df[c] = df[c] + d
    if set_:
        for c, v in set_.items():
            if c in df.columns:
                df[c] = v
    # one-hot groups: zero all group cols, set chosen to 1
    if onehot_sets:
        for prefix, chosen in onehot_sets.items():
            group = group_onehots_by_prefix(DUMMY_COLS, prefix)
            if not group:
                continue
            for g in group:
                if g in df.columns:
                    df[g] = 0.0
            if chosen in df.columns:
                df[chosen] = 1.0
    return df

def population_effect(base_df: pd.DataFrame,
                      add=None, set_=None, onehot_sets=None):
    """
    Apply intervention to all rows and return mean predictions (scaled + inverse).
    """
    df_int = apply_intervention(base_df, add=add, set_=set_, onehot_sets=onehot_sets)
    preds_s, preds = predict_df(df_int)
    return preds_s.mean().to_dict(), preds.mean().to_dict()

def grid_search_continuous(base_df: pd.DataFrame, sweep: dict):
    """
    sweep: dict {col_name: np.array of values to set (absolute, scaled units)}
    Returns DataFrame of average outcomes for each grid point.
    """
    from itertools import product
    keys = list(sweep.keys())
    grids = list(product(*[sweep[k] for k in keys]))
    rows = []
    for values in grids:
        set_dict = {k: v for k, v in zip(keys, values)}
        _, mean_inv = population_effect(base_df, set_=set_dict)
        rows.append({**{'param_'+k: v for k, v in set_dict.items()}, **mean_inv})
    return pd.DataFrame(rows)

def grid_search_mixed(base_df: pd.DataFrame,
                      continuous_set: dict[str, np.ndarray] | None,
                      onehot_choices: dict[str, list[str]] | None):
    """
    continuous_set: {col: np.array of absolute values (scaled units)}
    onehot_choices: {prefix: [candidate one-hot column names]}
    """
    from itertools import product
    cont_items = list((continuous_set or {}).items())
    cat_items  = list((onehot_choices or {}).items())
    cont_keys  = [k for k,_ in cont_items]
    cont_grids = [v for _,v in cont_items]
    cat_keys   = [k for k,_ in cat_items]
    cat_grids  = [v for _,v in cat_items]

    rows = []
    for cont_vals in product(*cont_grids or [()]):
        for cat_vals in product(*cat_grids or [()]):
            set_dict = {k: v for k, v in zip(cont_keys, cont_vals)}
            onehot_sets = {k: v for k, v in zip(cat_keys, cat_vals)}
            _, mean_inv = population_effect(base_df, set_=set_dict, onehot_sets=onehot_sets)
            rows.append({**{'param_'+k: v for k,v in set_dict.items()},
                         **{'choice_'+k: v for k,v in onehot_sets.items()},
                         **mean_inv})
    return pd.DataFrame(rows)

if __name__ == '__main__':

    with torch.no_grad():
        x = torch.tensor(df_scaled.values[:1024], dtype=torch.float32, device=DEVICE)
        mu, logvar = vae.encode(x)
        z = mu

        # Perturb the first latent dimension
        z_perturbed = z.clone()
        z_perturbed[:, 0] += 2.0  # Add 2 to the first latent dimension

        # Predict WLZ for original and perturbed latent
        wlz_out = heads.reg_heads['WLZ_WHZ_52'](z).squeeze(1).cpu().numpy()
        wlz_out_perturbed = heads.reg_heads['WLZ_WHZ_52'](z_perturbed).squeeze(1).cpu().numpy()

        print("WLZ mean delta after latent perturbation:",
              float((wlz_out_perturbed - wlz_out).mean()))
    # # Example 1: single-variable sweep (scaled units)
    # deltas = np.linspace(-2, 2, 21)
    # sweep_df = grid_search_continuous(df_scaled, {'Weight': deltas})
    # sweep_df.to_csv('../Outcomes/whatif_weight_sweep.tsv', sep='\t', index=False)

    # # Example 2: 2D sweep over Weight and Height (if present)
    # if 'Length' in df_scaled.columns:
    #     grid2 = grid_search_continuous(df_scaled, {'Weight': deltas, 'Length': deltas})
    #     grid2.to_csv('../Outcomes/whatif_weight_height_sweep.tsv', sep='\t', index=False)

    # # Example 3: mixed – set continuous and toggle one-hot Feed group
    # feed_group = [c for c in DUMMY_COLS if c.startswith('Feed_')]
    # # pick two example choices if they exist
    # feed_choices = [c for c in feed_group if c in df_scaled.columns]
    # if feed_choices:
    #     mixed = grid_search_mixed(
    #         df_scaled,
    #         continuous_set={'Weight': np.linspace(-1, 1, 5)},
    #         onehot_choices={'Feed': feed_choices}
    #     )
    #     mixed.to_csv('../Outcomes/whatif_mixed_weight_feed.tsv', sep='\t', index=False)

    # # Example 4: absolute “policy” toggles applied to everyone
    # # e.g., force ERUSF=1 and Local RUSF=0
    # _, mean_out = population_effect(
    #     df_scaled,
    #     onehot_sets={'Feed': 'Feed_ERUSF (B)'}
    # )
    # print("Mean outcomes if Feed=ERUSF(B):", mean_out)