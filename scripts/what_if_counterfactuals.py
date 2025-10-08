#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from vae import VAEWorld
from outcome_heads_nobrain import VAEWithHeads
import joblib
from itertools import product

LATENT_DIM  = 64
HIDDEN_DIM  = 1024
BATCH_SIZE  = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load inputs
df_scaled = pd.read_csv('../data/combined_imputed_scaled_large_nolip_psd.tsv', sep='\t', index_col=0)
meta_data = pd.read_csv('../data/meta.tsv', sep='\t')
mam_ids = meta_data[meta_data['Condition'] == 'MAM']['subjectID']
df_scaled = df_scaled.loc[df_scaled.index.intersection(mam_ids)]

INPUT_DIM = df_scaled.shape[1]

editable_cols = pd.read_csv('../data/combined_imputed_scaled_large_nolip_editable.tsv', sep='\t', index_col=0).columns.tolist()
# remove Members_in_household, Number of rooms in household
editable_cols = [c for c in editable_cols if c != 'Members_in_household' and c != 'Number_of_rooms_in_current_household']
# remove microbiome diversity scores (anything that ends in 0 or 52) from editable columns
editable_cols = [c for c in editable_cols if not (c.endswith('_0') or c.endswith('_52'))]

# Load VAE + Heads
vae = VAEWorld(INPUT_DIM, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
vae.load_state_dict(torch.load('vae_world_large_mam_nolip_psd_mse.pt', map_location=DEVICE))
vae.eval()

heads = VAEWithHeads(vae).to(DEVICE)
ckpt = torch.load('../models/vae_heads_nolip_psd_mse.pt', map_location=DEVICE)
state = ckpt.get('state_dict', ckpt)
missing, unexpected = heads.load_state_dict(state, strict=False)
print("Heads load_state_dict missing keys:", missing)
print("Heads load_state_dict unexpected keys:", unexpected)
heads.eval()

# target scaler for inverse-transform
target_scaler = ckpt.get('target_scaler', None)
regression_cols = ckpt.get('regression_cols', ['cognitive_score_52', 'vocalisation_52', 'WLZ_WHZ_52'])

# One-hot detection (binary columns stay 0/1)
def detect_dummy_cols(df: pd.DataFrame):
    return [c for c in df.columns if df[c].dropna().isin([0, 1]).all()]

DUMMY_COLS = detect_dummy_cols(df_scaled)

def group_onehots_by_prefix(dummy_cols, prefix: str):
    # require prefix match safely and e.g. allow 'Feed' to match 'Feed_...'
    pfx = prefix if prefix.endswith('_') else prefix + '_'
    return [c for c in dummy_cols if c.startswith(pfx)]

# Encoding and prediction
@torch.no_grad()
def encode_df(df: pd.DataFrame) -> torch.Tensor:
    ''' Encode entire df in batches, return latent means only.'''
    vae.eval()
    zs = []
    for start in range(0, len(df), BATCH_SIZE):
        x = torch.tensor(df.iloc[start:start+BATCH_SIZE].values, dtype=torch.float32, device=DEVICE)
        mu, _ = vae.encode(x)
        zs.append(mu)  # use mean only (don't sample here so don't need noise and sd)
    return torch.cat(zs, dim=0)

@torch.no_grad()
def predict_df(df: pd.DataFrame):
    ''' Predict outcomes for entire df in batches, return scaled and inv-scaled (og units) DataFrames.
    If no target_scaler was saved, inv-scaled will be same as scaled.'''
    vae.eval()
    heads.eval()
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

# Interventions
def apply_intervention(base_df: pd.DataFrame,
                       add: dict | None = None,
                       set_: dict | None = None,
                       onehot_sets: dict | None = None) -> pd.DataFrame:
    """
    This either adds an amount to or sets continuous features, and/or sets one-hot groups.
    add:  dict of col: delta_in_scaled_units
    set_: dict of col: absolute_in_scaled_units
    onehot_sets: dict of prefix: chosen_col_name (e.g., {'Feed': 'Feed_ERUSF (B)'})

    It then outputs a new DataFrame with the intervention applied.
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
    '''
    Estimate the population effect of an intervention by applying it to the base DataFrame
    and predicting the outcomes.
    Returns mean predicted outcomes (scaled and inv-scaled).
    '''
    df_int = apply_intervention(base_df, add=add, set_=set_, onehot_sets=onehot_sets)
    preds_s, preds_og = predict_df(df_int)
    return preds_s.mean().to_dict(), preds_og.mean().to_dict()

def grid_search_continuous(base_df: pd.DataFrame, sweep: dict):
    '''
    For a grid of continuous parameter values, estimate population effects.
    sweep: dict of col: list_of_values_in_scaled_units
    Returns a DataFrame with the results.
    '''
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
    '''
    Same as grid_search_continuous but also sweeps over one-hot group choices.
    continuous_set: dict of col: list_of_values_in_scaled_units
    onehot_choices: dict of prefix: list_of_chosen_col_names to oneshot between
    '''
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


# Diagnostics
def input_sensitivity(df: pd.DataFrame, top_k=20) -> pd.Series:
    '''Computes average abs gradient over a mini-batch of inputs (batch_size).
    Returns top_k features with highest abs grad.'''
    vae.eval()
    heads.eval()
    x = torch.tensor(df.iloc[:min(len(df), BATCH_SIZE)].values, dtype=torch.float32, device=DEVICE, requires_grad=True)
    mu, _ = vae.encode(x)
    z = mu
    wlz_mean = heads.reg_heads['WLZ_WHZ_52'](z).squeeze(1).mean()
    vae.zero_grad(set_to_none=True); heads.zero_grad(set_to_none=True)
    wlz_mean.backward()
    grad = x.grad.detach().abs().mean(0).cpu().numpy()
    idx = np.argsort(-grad)[:top_k]
    return pd.Series(grad[idx], index=df.columns[idx])

def input_sensitivity_directional(df: pd.DataFrame, top_k=20) -> pd.Series:
    '''Computes average gradient over a mini-batch of inputs (batch_size).
    Returns top_k features with highest grad. (i.e. same as input_sensitivity but not abs)'''
    vae.eval()
    heads.eval()
    x = torch.tensor(df.iloc[:min(len(df), BATCH_SIZE)].values, dtype=torch.float32, device=DEVICE, requires_grad=True)
    mu, _ = vae.encode(x)
    z = mu
    wlz_mean = heads.reg_heads['WLZ_WHZ_52'](z).squeeze(1).mean()
    vae.zero_grad(set_to_none=True); heads.zero_grad(set_to_none=True)
    wlz_mean.backward()
    grad = x.grad.detach().mean(0).cpu().numpy()  
    idx = np.argsort(-np.abs(grad))[:top_k]
    return pd.Series(grad[idx], index=df.columns[idx])

def diagnose_intervention(base_df: pd.DataFrame, set_: dict | None = None, add: dict | None = None):
    '''Applies an intervention and reports (via print statement):
    - Top 10 latent dimensions changed (mean |Δz|)
    - Mean WLZ change (orig units)
    Intervention can be 'set_' and/or 'add'  as in either add to a feature or set as a value. Can be multiple of either.
    Returns the intervened DataFrame.
    '''
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
    # change in WLZ from prediction based on og features to prediction based on intervened features
    wlz_delta = float((preds1['WLZ_WHZ_52'] - preds0['WLZ_WHZ_52']).mean())

    print("Top latent |Δz| dims:", list(zip(top.tolist(), dz[top].round(4).tolist())))
    print("WLZ mean delta:", wlz_delta)
    return df_int

def individual_effects(base_df: pd.DataFrame, add=None, set_=None, onehot_sets=None, top_k=10):
    '''
    Applies an intervention and reports the top_k individuals with largest |WLZ delta| (orig units) from that intervention.
    Returns a DataFrame with their baseline WLZ, counterfactual WLZ, and delta.
    Intervention can be 'set_' and/or 'add' as in either add to a feature or set as a value. Can be multiple of either.
    '''
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
    '''Clamp binary/dummy columns to [0,1] in-place so that they remain valid.
    '''
    if len(cols) == 0: 
        return
    x_tensor.data[:, cols] = x_tensor.data[:, cols].clamp(0.0, 1.0)

def _freeze_disallowed_inplace(x_tensor: torch.Tensor, x0_tensor: torch.Tensor, allowed_idx: list[int]):
    '''For indices not in allowed_idx, set x_tensor to x0_tensor values in-place. These are the ones that aren't modifiable
    '''
    all_idx = torch.arange(x_tensor.shape[1], device=x_tensor.device)
    mask = torch.ones_like(all_idx, dtype=torch.bool)
    mask[allowed_idx] = False
    disallowed_idx = all_idx[mask]
    if disallowed_idx.numel() > 0:
        x_tensor.data[:, disallowed_idx] = x0_tensor.data[:, disallowed_idx]

def counterfactual_optimize_x(row: pd.Series,
                              target_delta=0.3,
                              lam=1e-2,
                              steps=300,
                              lr=0.05,
                              dummy_cols: list[str] | None = None,
                              allowed_cols: list[str] | None = None):
    """
    Find x_cf close to x that changes WLZ by target_delta.
    Works in scaled input space; returns df of x, x_cf, and prediction change.
    """
    vae.eval(); heads.eval()
    # Prep tensors
    x0_np = row.values.astype(np.float32, copy=True)
    x0 = torch.tensor(x0_np[None, :], device=DEVICE)
    x_cf = torch.nn.Parameter(x0.clone())

    # Which indices are binary/dummies
    dummy_indices = [row.index.get_loc(c) for c in (dummy_cols or []) if c in row.index]
    # which allowed to change
    allowed_indices = [row.index.get_loc(c) for c in (allowed_cols or []) if c in row.index]

    # Compute current WLZ
    with torch.no_grad():
        mu0, _ = vae.encode(x0)
        z0 = mu0
        y0_s = heads.reg_heads['WLZ_WHZ_52'](z0)
        y0 = y0_s.clone()

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
        # freeze non-modifiable features
        if allowed_cols is not None:
            _freeze_disallowed_inplace(x_cf, x0, allowed_indices)

    with torch.no_grad():
        mu, _ = vae.encode(x_cf)
        y_cf = heads.reg_heads['WLZ_WHZ_52'](mu)
        delta = float((y_cf - y0).mean().item())

    x_cf_np = x_cf.detach().cpu().numpy().squeeze(0)

        # Round the binary features in the final result
    # binary_col_names = [c for c in (dummy_cols or []) if c in row.index]
    # for col_name in binary_col_names:
    #     loc = row.index.get_loc(col_name)
    #     x_cf_np[loc] = round(x_cf_np[loc])

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

def wlz_std_original_units():
    # Return std of WLZ in original units
    try:
        if target_scaler is None or not hasattr(target_scaler, 'scale_'):
            return None
        idx = regression_cols.index('WLZ_WHZ_52')
        return float(target_scaler.scale_[idx])
    except Exception:
        return None

def generate_population_optimization_results(df: pd.DataFrame,
                                             target_wlz=-1,
                                             lam=5e-3,
                                             steps=400,
                                             lr=0.05,
                                             dummy_cols: list[str] | None = None,
                                             allowed_cols: list[str] | None = None):
    """
    Runs counterfactual optimization for every individual in the dataframe to reach a target WLZ.
    Takes all individuals below target_wlz and tries to optimize them to reach target_wlz.
    Returns a DataFrame with all changes made for all individuals. This df gets fed into make_figs.py functions
    """
    all_changes_list = []

    wlz_idx = regression_cols.index('WLZ_WHZ_52')
    wlz_mean = target_scaler.mean_[wlz_idx]
    wlz_std = target_scaler.scale_[wlz_idx]

    # Get baseline predictions for everyone first to calculate target_delta efficiently
    _, preds_baseline_orig = predict_df(df)
    
    for ix, person in tqdm(df.iterrows(), total=len(df)):
        # Get the individual's baseline WLZ in original units
        wlz_base_orig = preds_baseline_orig.loc[ix, 'WLZ_WHZ_52']

        # Don't run optimization if they already meet the target
        if wlz_base_orig >= target_wlz:
            continue

        # Convert everything to scaled units for the optimizer
        wlz_base_scaled = (wlz_base_orig - wlz_mean) / wlz_std 
        target_wlz_scaled = (target_wlz - wlz_mean) / wlz_std
        target_delta_scaled = target_wlz_scaled - wlz_base_scaled

        # Run optimization
        cf = counterfactual_optimize_x(
            person,
            target_delta=target_delta_scaled,
            lam=lam,
            steps=steps,
            lr=lr,
            dummy_cols=dummy_cols,
            allowed_cols=allowed_cols
        )

        changes_df = cf['changes'].copy()
        changes_df['subjectID'] = ix
        all_changes_list.append(changes_df.reset_index())

    full_changes_df = pd.concat(all_changes_list, ignore_index=True)

    return full_changes_df

def run_single_subject_cf(subject_id: str, target_wlz_gain: float = 1.0):
    """
    purpose is just so I can easily call the counterfactuals from the figure script
    """
    person = df_scaled.loc[subject_id]
    wlz_std = wlz_std_original_units() 

    target_delta_scaled = target_wlz_gain / wlz_std
    cf = counterfactual_optimize_x(person, target_delta=target_delta_scaled, lam=5e-3, steps=400, lr=0.05, dummy_cols=DUMMY_COLS, allowed_cols=editable_cols)

    # inverse scale results back to og units
    wlz_idx = regression_cols.index('WLZ_WHZ_52')
    wlz_mean = target_scaler.mean_[wlz_idx]
    wlz_std = target_scaler.scale_[wlz_idx]

    wlz_base_orig = cf['wlz_base'] * wlz_std + wlz_mean
    wlz_cf_orig = cf['wlz_cf'] * wlz_std + wlz_mean
    
    return {'subject_id': subject_id,
        'changes_df': cf['changes'],
        'wlz_base': wlz_base_orig,
        'wlz_cf': wlz_cf_orig
    }

def unscale_intervention_data(intervention_data: pd.DataFrame,
                              combined_matrix_path: str = '../data/combined_matrix_large.tsv',
                              scaler_path: str = '../data/scaler_large_nolip_psd.save') -> pd.DataFrame:
    """
    Convert x0/x_cf/delta from scaled units back to original units for non-categorical features.
    Categorical are left unchanged.

    intervention_data: DataFrame from generate_population_optimization_results() with columns ['subjectID','feature','x0','x_cf','delta'].
    combined_matrix_path: Path to original (pre-impute/scale) combined matrix (tsv file).
    scaler_path: Path to joblib'd StandardScaler fitted on continuous columns.

    Returns df with added columns for og units ['x0_orig','x_cf_orig','delta_orig'].
    """
    combined_df = pd.read_csv(combined_matrix_path, sep='\t')
    dummy_cols = [
        col for col in combined_df.columns
        if col != 'subjectID'
        and combined_df[col].dropna().nunique() == 2
        and set(combined_df[col].dropna().unique()) <= {0, 1}
    ]
    categorical_cols = ['Number_of_rooms_in_current_household'] + dummy_cols
    # Preserve original column order used for fit_transform (see format_matrix_og.py)
    continuous_cols = [c for c in combined_df.columns if c not in categorical_cols and c != 'subjectID']

    # Load the fitted scaler and build feature (mean, std) maps
    scaler = joblib.load(scaler_path)
    means = dict(zip(continuous_cols, scaler.mean_))
    scales = dict(zip(continuous_cols, scaler.scale_))

    # Apply inverse transform per-row to get unscaled values
    def _unscale_row(r: pd.Series) -> pd.Series:
        feat = r['feature']
        if feat in means:  # was scaled
            m, s = means[feat], scales[feat]
            return pd.Series({
                'x0_orig':   r['x0'] * s + m,
                'x_cf_orig': r['x_cf'] * s + m,
                'delta_orig': r['delta'] * s
            })
        else:
            # categorical/unscaled: pass-through
            return pd.Series({
                'x0_orig':   r['x0'],
                'x_cf_orig': r['x_cf'],
                'delta_orig': r['delta']
            })

    out = intervention_data.copy()
    out[['x0_orig', 'x_cf_orig', 'delta_orig']] = out.apply(_unscale_row, axis=1)
    return out

# Main
if __name__ == '__main__':
    # Baseline predictions
    # preds_s, preds = predict_df(df_scaled)
    # print("Baseline WLZ mean/std (orig units):",
    #       float(preds['WLZ_WHZ_52'].mean()),
    #       float(preds['WLZ_WHZ_52'].std()))

    # # Strong change on a continuous feature
    # some_cont = None
    # for c in df_scaled.columns:
    #     if c not in DUMMY_COLS:
    #         some_cont = c
    #         break
    # if some_cont is not None:
    #     df_jitter = df_scaled.copy()
    #     df_jitter[some_cont] = df_jitter[some_cont] + 2.0
    #     _, preds_j = predict_df(df_jitter)
    #     print(f"WLZ mean delta after +2 on {some_cont}:",
    #           float((preds_j['WLZ_WHZ_52'] - preds['WLZ_WHZ_52']).mean()))

    # # Latent perturbation sensitivity
    # with torch.no_grad():
    #     x = torch.tensor(df_scaled.values[:min(1024, len(df_scaled))], dtype=torch.float32, device=DEVICE)
    #     mu, _ = vae.encode(x)
    #     z = mu
    #     z_pert = z.clone()
    #     z_pert[:, 0] += 2.0
    #     wlz0 = heads.reg_heads['WLZ_WHZ_52'](z).squeeze(1).cpu().numpy()
    #     wlz1 = heads.reg_heads['WLZ_WHZ_52'](z_pert).squeeze(1).cpu().numpy()
    #     print("WLZ mean delta after latent[0] += 2:",
    #           float((wlz1 - wlz0).mean()))


    # # Input sensitivity ranking
    # print("\nTop input sensitivities (|d WLZ / d x|):")
    # print(input_sensitivity_directional(df_scaled, top_k=20))

    # # Diagnose a named intervention
    # if 'Weight' in df_scaled.columns:
    #     _ = diagnose_intervention(df_scaled, set_={'Weight': df_scaled['Weight'].mean() + 2.0})

    # # enforce a one-hot group toggle (Feed) if present
    # feed_group = group_onehots_by_prefix(DUMMY_COLS, 'Feed')
    # if feed_group:
    #     chosen = feed_group[0]  # choose any available feed category
    #     df_feed = apply_intervention(df_scaled, onehot_sets={'Feed': chosen})
    #     _, preds_feed = predict_df(df_feed)
    #     delta = preds_feed['WLZ_WHZ_52'] - preds['WLZ_WHZ_52']
    #     print(f"WLZ mean delta after forcing {chosen}:", float(delta.mean()), "std:", float(delta.std()))

    # #  who responds most to a given intervention
    # if 'Weight' in df_scaled.columns:
    #     top = individual_effects(df_scaled, set_={'Weight': df_scaled['Weight'].mean() + 2.0}, top_k=10)
    #     print("\nTop 10 individual WLZ deltas for Weight+2 (orig units):")
    #     print(top)

    # # Counterfactual optimization on a single person
    # ix = np.random.choice(df_scaled.index)
    # person = df_scaled.loc[ix]
    # DUMMY_COLS = [c for c in df_scaled.columns if df_scaled[c].dropna().isin([0,1]).all()]

    # og_unit_change = 1.0
    # target_delta_scaled = og_unit_change / (wlz_std_original_units() or 1.0)

    # cf = counterfactual_optimize_x(person, target_delta=target_delta_scaled, lam=5e-3, steps=400, lr=0.05, dummy_cols=DUMMY_COLS, allowed_cols=editable_cols)

    # if target_scaler is not None:
    #     try:
    #         wlz_idx = regression_cols.index('WLZ_WHZ_52')
    #         wlz_mean = target_scaler.mean_[wlz_idx]
    #         wlz_std = target_scaler.scale_[wlz_idx]
    #         wlz_base_orig = cf['wlz_base'] * wlz_std + wlz_mean
    #         wlz_cf_orig = cf['wlz_cf'] * wlz_std + wlz_mean
    #         wlz_delta_orig = cf['wlz_delta'] * wlz_std
    #     except (ValueError, IndexError):
    #         print("Warning: Could not find 'WLZ_WHZ_52' in target_scaler columns. Using scaled values.")
    #         wlz_base_orig = cf['wlz_base']
    #         wlz_cf_orig = cf['wlz_cf']
    #         wlz_delta_orig = cf['wlz_delta']

    # print(f"\nCounterfactual for {ix}: WLZ {wlz_base_orig:.3f} -> {wlz_cf_orig:.3f} (Δ {wlz_delta_orig:.3f}) [original units]")
    # print("Top changed inputs (scaled units):")
    # print(cf['changes'].head(15))

    # Run the full population analysis
    intervention_data = generate_population_optimization_results(
        df=df_scaled,
        target_wlz=-1,
        dummy_cols=DUMMY_COLS,
        allowed_cols=editable_cols
    )
    # for outputting intervention_data, want only feature, delta, subjectID for make_figs.py input
    intervention_data_out = intervention_data[['feature', 'delta', 'subjectID']]
    #intervention_data_out.to_csv('../Outcomes/population_wlz_deltas_neg1.tsv', sep='\t', header=True)
    intervention_data_orig = unscale_intervention_data(intervention_data)
    #intervention_data_orig.to_csv('../Outcomes/test_input_original_units.tsv', sep='\t', index=False)

    mean_abs_delta = intervention_data.groupby('feature')['delta'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)

    mean_delta = intervention_data.groupby('feature')['delta'].mean()

    top_feature_summary = pd.DataFrame({
        'mean_abs_delta': mean_abs_delta,
        'mean_delta': mean_delta
    }).loc[mean_abs_delta.index]

    print('Top 20 features by mean abs change')
    print(top_feature_summary.head(20))

    # apply average changes to population and see WLZ effect
    # id those below target wlz
    _, preds_base = predict_df(df_scaled)
    base_below = df_scaled.loc[preds_base['WLZ_WHZ_52'] < -1.0]
    # apply mean changes to editable features for these indivs
    df_below_int = base_below.copy()
    for feat, row in mean_delta.items():
        if feat in df_below_int.columns:
            if feat in DUMMY_COLS:
                # round to nearest int and clamp to [0,1]
                df_below_int[feat] = (df_below_int[feat] + row).round().clip(0, 1)
            else:
                df_below_int[feat] = df_below_int[feat] + row
    # predict new WLZ
    _, preds_below_int = predict_df(df_below_int)

    # count how many now above -1.0
    new_above = (preds_below_int['WLZ_WHZ_52'] >= -1.0).sum()
    total_below = len(base_below)
    print(f"\nAfter applying mean changes to those below -1.0 WLZ, {new_above} out of {total_below} are now above -1.0.")

    # Example grid over a continuous var
    # if 'Weight' in df_scaled.columns:
    #     deltas = np.linspace(-2, 2, 21)
    #     grid = grid_search_continuous(df_scaled, {'Weight': deltas})
    #     grid.to_csv('../Outcomes/whatif_weight_sweep.tsv', sep='\t', index=False)