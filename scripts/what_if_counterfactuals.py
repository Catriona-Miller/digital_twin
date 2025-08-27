#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
from vae import VAEWorld
from outcome_heads_train_reg import VAEWithHeads

LATENT_DIM  = 64
HIDDEN_DIM  = 1024
BATCH_SIZE  = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load inputs (already z-scored)
# ----------------------------
df_scaled = pd.read_csv('../data/combined_imputed_scaled_large_nolip.tsv', sep='\t', index_col=0)
meta_data = pd.read_csv('../data/meta.tsv', sep='\t')
mam_ids = meta_data[meta_data['Condition'] == 'MAM']['subjectID']
df_scaled = df_scaled.loc[df_scaled.index.intersection(mam_ids)]

INPUT_DIM = df_scaled.shape[1]

# ----------------------------
# Load VAE + Heads
# ----------------------------
vae = VAEWorld(INPUT_DIM, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
vae.load_state_dict(torch.load('vae_world_large_mam_nolip_mse.pt', map_location=DEVICE))
vae.eval()

heads = VAEWithHeads(vae).to(DEVICE)
ckpt = torch.load('../models/vae_heads_nolip_mse_onecyclelr.pt', map_location=DEVICE)
state = ckpt.get('state_dict', ckpt)
missing, unexpected = heads.load_state_dict(state, strict=False)
print("Heads load_state_dict missing keys:", missing)
print("Heads load_state_dict unexpected keys:", unexpected)
heads.eval()

# Optional: target scaler for inverse-transform (if saved)
target_scaler = ckpt.get('target_scaler', None)
regression_cols = ckpt.get('regression_cols', ['cognitive_score_52', 'vocalisation_52', 'WLZ_WHZ_52'])

# ----------------------------
# One-hot detection (binary columns stay 0/1)
# ----------------------------
def detect_dummy_cols(df: pd.DataFrame):
    return [c for c in df.columns if df[c].dropna().isin([0, 1]).all()]

DUMMY_COLS = detect_dummy_cols(df_scaled)

def group_onehots_by_prefix(dummy_cols, prefix: str):
    # require prefix match safely; allow 'Feed' to match 'Feed_...'
    pfx = prefix if prefix.endswith('_') else prefix + '_'
    return [c for c in dummy_cols if c.startswith(pfx)]

# ----------------------------
# Encoding and prediction
# ----------------------------
@torch.no_grad()
def encode_df(df: pd.DataFrame) -> torch.Tensor:
    vae.eval()
    zs = []
    for start in range(0, len(df), BATCH_SIZE):
        x = torch.tensor(df.iloc[start:start+BATCH_SIZE].values, dtype=torch.float32, device=DEVICE)
        mu, _ = vae.encode(x)
        zs.append(mu)  # deterministic: use mean only
    return torch.cat(zs, dim=0)

@torch.no_grad()
def predict_df(df: pd.DataFrame):
    vae.eval(); heads.eval()
    frames = []
    for start in range(0, len(df), BATCH_SIZE):
        idx = df.index[start:start+BATCH_SIZE]
        x = torch.tensor(df.iloc[start:start+BATCH_SIZE].values, dtype=torch.float32, device=DEVICE)
        mu, _ = vae.encode(x)
        z = mu
        out = {
            'cognitive_score_52': heads.reg_heads['cognitive_score_52'](z).squeeze(1).cpu().numpy(),
            'vocalisation_52':    heads.reg_heads['vocalisation_52'](z).squeeze(1).cpu().numpy(),
            'WLZ_WHZ_52':         heads.reg_heads['WLZ_WHZ_52'](z).squeeze(1).cpu().numpy(),
        }
        frames.append(pd.DataFrame(out, index=idx))
    preds_scaled = pd.concat(frames, axis=0).loc[df.index]

    if target_scaler is not None:
        cols = ['cognitive_score_52', 'vocalisation_52', 'WLZ_WHZ_52']
        arr_inv = target_scaler.inverse_transform(preds_scaled[cols].values)
        preds_inv = pd.DataFrame(arr_inv, index=preds_scaled.index, columns=cols)
    else:
        preds_inv = preds_scaled.copy()

    return preds_scaled, preds_inv

# ----------------------------
# Interventions
# ----------------------------
def apply_intervention(base_df: pd.DataFrame,
                       add: dict | None = None,
                       set_: dict | None = None,
                       onehot_sets: dict | None = None) -> pd.DataFrame:
    """
    add:  {col: delta_in_scaled_units}
    set_: {col: absolute_in_scaled_units}
    onehot_sets: {prefix: chosen_col_name} (e.g., {'Feed': 'Feed_ERUSF (B)'})
    """
    df = base_df.copy()
    if add:
        for c, d in add.items():
            if c in df.columns:
                df[c] = df[c] + d
    if set_:
        for c, v in set_.items():
            if c in df.columns:
                df[c] = v
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

def population_effect(base_df: pd.DataFrame, add=None, set_=None, onehot_sets=None):
    df_int = apply_intervention(base_df, add=add, set_=set_, onehot_sets=onehot_sets)
    preds_s, preds = predict_df(df_int)
    return preds_s.mean().to_dict(), preds.mean().to_dict()

def grid_search_continuous(base_df: pd.DataFrame, sweep: dict):
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

# ----------------------------
# Diagnostics
# ----------------------------
def input_sensitivity(df: pd.DataFrame, top_k=20) -> pd.Series:
    # Computes average |d WLZ / d x| over a mini-batch
    vae.eval(); heads.eval()
    x = torch.tensor(df.iloc[:min(len(df), BATCH_SIZE)].values, dtype=torch.float32, device=DEVICE, requires_grad=True)
    mu, _ = vae.encode(x)
    z = mu
    wlz_mean = heads.reg_heads['WLZ_WHZ_52'](z).squeeze(1).mean()
    vae.zero_grad(set_to_none=True); heads.zero_grad(set_to_none=True)
    wlz_mean.backward()
    grad = x.grad.detach().abs().mean(0).cpu().numpy()
    idx = np.argsort(-grad)[:top_k]
    return pd.Series(grad[idx], index=df.columns[idx])

def diagnose_intervention(base_df: pd.DataFrame, set_: dict | None = None, add: dict | None = None):
    df_int = base_df.copy()
    for c, v in (set_ or {}).items():
        if c in df_int.columns:
            df_int[c] = v
    for c, d in (add or {}).items():
        if c in df_int.columns:
            df_int[c] = df_int[c] + d

    z0 = encode_df(base_df)
    z1 = encode_df(df_int)
    dz = (z1 - z0).abs().mean(0).cpu().numpy()
    top = np.argsort(-dz)[:10]
    preds0_s, preds0 = predict_df(base_df)
    preds1_s, preds1 = predict_df(df_int)
    wlz_delta = float((preds1['WLZ_WHZ_52'] - preds0['WLZ_WHZ_52']).mean())

    print("Top latent |Δz| dims:", list(zip(top.tolist(), dz[top].round(4).tolist())))
    print("WLZ mean delta:", wlz_delta)
    return df_int

def individual_effects(base_df: pd.DataFrame, add=None, set_=None, onehot_sets=None, top_k=10):
    df_int = apply_intervention(base_df, add=add, set_=set_, onehot_sets=onehot_sets)
    _, base = predict_df(base_df)
    _, cf   = predict_df(df_int)
    delta = (cf['WLZ_WHZ_52'] - base['WLZ_WHZ_52'])
    rank = delta.abs().sort_values(ascending=False)
    top_idx = rank.index[:top_k]
    out = pd.DataFrame({
        'WLZ_base': base.loc[top_idx, 'WLZ_WHZ_52'],
        'WLZ_cf':   cf.loc[top_idx,   'WLZ_WHZ_52'],
        'WLZ_delta': delta.loc[top_idx]
    }, index=top_idx)
    return out

def _project_binaries_inplace(x_tensor: torch.Tensor, cols: list[int]):
    # clamp to [0,1] using .data to avoid autograd error
    if len(cols) == 0: return
    x_tensor.data[:, cols] = x_tensor.data[:, cols].clamp(0.0, 1.0)

def counterfactual_optimize_x(row: pd.Series,
                              target_delta=0.3,
                              lam=1e-2,
                              steps=300,
                              lr=0.05,
                              dummy_cols: list[str] | None = None):
    """
    Find x_cf close to x that changes WLZ by target_delta (orig units).
    Works in scaled input space; returns df of x, x_cf, and prediction change.
    """
    vae.eval(); heads.eval()
    # Prep tensors
    x0_np = row.values.astype(np.float32, copy=True)
    x0 = torch.tensor(x0_np[None, :], device=DEVICE)
    x_cf = torch.nn.Parameter(x0.clone())

    # Which indices are binary/dummies
    dummy_indices = [row.index.get_loc(c) for c in (dummy_cols or []) if c in row.index]

    # Compute current WLZ (orig units)
    with torch.no_grad():
        mu0, _ = vae.encode(x0)
        z0 = mu0
        y0_s = heads.reg_heads['WLZ_WHZ_52'](z0)
        y0 = y0_s.clone()
        # If you saved a target scaler, inverse-transform here if needed.
        # Assuming head already outputs in orig units per your setup.

    target = (y0 + target_delta).detach()

    opt = torch.optim.Adam([x_cf], lr=lr)
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        mu, _ = vae.encode(x_cf)
        z = mu
        y = heads.reg_heads['WLZ_WHZ_52'](z)
        loss = (y - target).pow(2).mean() + lam * (x_cf - x0).pow(2).mean()
        loss.backward()
        opt.step()
        # keep binaries in [0,1]
        _project_binaries_inplace(x_cf, dummy_indices)

    with torch.no_grad():
        mu, _ = vae.encode(x_cf)
        y_cf = heads.reg_heads['WLZ_WHZ_52'](mu)
        delta = float((y_cf - y0).mean().item())

    x_cf_np = x_cf.detach().cpu().numpy().squeeze(0)
    df_changes = pd.DataFrame({
        'feature': row.index,
        'x0': x0_np,
        'x_cf': x_cf_np,
        'delta': x_cf_np - x0_np
    }).set_index('feature').sort_values('delta', key=np.abs, ascending=False)

    return {
        'wlz_base': float(y0.item()),
        'wlz_cf': float(y_cf.item()),
        'wlz_delta': delta,
        'changes': df_changes
    }


# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    # Baseline predictions
    preds_s, preds = predict_df(df_scaled)
    print("Baseline WLZ mean/std (orig units):",
          float(preds['WLZ_WHZ_52'].mean()),
          float(preds['WLZ_WHZ_52'].std()))

    # Strong change on a continuous feature (if present)
    some_cont = None
    for c in df_scaled.columns:
        if c not in DUMMY_COLS:
            some_cont = c
            break
    if some_cont is not None:
        df_jitter = df_scaled.copy()
        df_jitter[some_cont] = df_jitter[some_cont] + 2.0
        _, preds_j = predict_df(df_jitter)
        print(f"WLZ mean delta after +2 on {some_cont}:",
              float((preds_j['WLZ_WHZ_52'] - preds['WLZ_WHZ_52']).mean()))

    # Latent perturbation sensitivity (first latent dim)
    with torch.no_grad():
        x = torch.tensor(df_scaled.values[:min(1024, len(df_scaled))], dtype=torch.float32, device=DEVICE)
        mu, _ = vae.encode(x)
        z = mu
        z_pert = z.clone()
        z_pert[:, 0] += 2.0
        wlz0 = heads.reg_heads['WLZ_WHZ_52'](z).squeeze(1).cpu().numpy()
        wlz1 = heads.reg_heads['WLZ_WHZ_52'](z_pert).squeeze(1).cpu().numpy()
        print("WLZ mean delta after latent[0] += 2:",
              float((wlz1 - wlz0).mean()))

    # Input sensitivity ranking
    print("\nTop input sensitivities (|d WLZ / d x|):")
    print(input_sensitivity(df_scaled, top_k=20))

    # Diagnose a named intervention (example: set Weight to +2 from its mean, if exists)
    if 'Weight' in df_scaled.columns:
        _ = diagnose_intervention(df_scaled, set_={'Weight': df_scaled['Weight'].mean() + 2.0})

    # Example: enforce a one-hot group toggle (Feed) if present
    feed_group = group_onehots_by_prefix(DUMMY_COLS, 'Feed')
    if feed_group:
        chosen = feed_group[0]  # choose any available feed category
        df_feed = apply_intervention(df_scaled, onehot_sets={'Feed': chosen})
        _, preds_feed = predict_df(df_feed)
        delta = preds_feed['WLZ_WHZ_52'] - preds['WLZ_WHZ_52']
        print(f"WLZ mean delta after forcing {chosen}:", float(delta.mean()), "std:", float(delta.std()))

    # 1) Heterogeneity: who responds most to a given intervention
    if 'Weight' in df_scaled.columns:
        top = individual_effects(df_scaled, set_={'Weight': df_scaled['Weight'].mean() + 2.0}, top_k=10)
        print("\nTop 10 individual WLZ deltas for Weight+2 (orig units):")
        print(top)

    # 2) Counterfactual optimization on a single person
    # pick a random individual
    ix = np.random.choice(df_scaled.index)
    person = df_scaled.loc[ix]
    # which columns are binary
    DUMMY_COLS = [c for c in df_scaled.columns if df_scaled[c].dropna().isin([0,1]).all()]
    cf = counterfactual_optimize_x(person, target_delta=1, lam=5e-3, steps=400, lr=0.05, dummy_cols=DUMMY_COLS)
    print(f"\nCounterfactual for {ix}: WLZ {cf['wlz_base']:.3f} -> {cf['wlz_cf']:.3f} (Δ {cf['wlz_delta']:.3f})")
    print("Top changed inputs (scaled units):")
    print(cf['changes'].head(15))
    # Example grid over a continuous var (save optional)
    # if 'Weight' in df_scaled.columns:
    #     deltas = np.linspace(-2, 2, 21)
    #     grid = grid_search_continuous(df_scaled, {'Weight': deltas})
    #     grid.to_csv('../Outcomes/whatif_weight_sweep.tsv', sep='\t', index=False)