from sklearn.model_selection import GroupShuffleSplit
import torch, pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from vae import VAEWorld
from torch import nn
import joblib
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
save = False

def combine_df(df):
    # separate into two dataframes for timepoints 0 and 52
    df_0 = df[df['sampleID'].str.endswith('_0')].copy()
    df_52 = df[df['sampleID'].str.endswith('_52')].copy()
    # remove the timepoint from sampleID
    df_0['sampleID'] = df_0['sampleID'].str.replace('_0', '', regex=False)
    df_52['sampleID'] = df_52['sampleID'].str.replace('_52', '', regex=False)
    # rename feature columns to include timepoint
    df_0.columns = [col + '_0' if col != 'sampleID' else col for col in df_0.columns]
    df_52.columns = [col + '_52' if col != 'sampleID' else col for col in df_52.columns]
    # when merging, if there are any subjects not in both timepoints, have NaN for that timepoint
    merged_df = pd.merge(df_0, df_52, on='sampleID', how='outer')
    merged_df.rename(columns={'sampleID': 'subjectID'}, inplace=True)
    return merged_df

df_scaled = pd.read_csv('../data/combined_imputed_scaled_large_nolip.tsv', sep='\t', index_col=0)
wolkes_data = pd.read_csv('../data/wolkes.tsv', sep='\t')
bayley_data = pd.read_csv('../data/bayley.tsv', sep='\t')
anthro_data = pd.read_csv('../data/anthro.tsv', sep='\t')
meta_data = pd.read_csv('../data/meta.tsv', sep='\t')
# get just the mam kids
df_scaled = df_scaled[df_scaled.index.isin(meta_data[meta_data['Condition'] == 'MAM']['subjectID'])]

wolkes_data = combine_df(wolkes_data)
bayley_data = combine_df(bayley_data)
anthro_data = combine_df(anthro_data)
comb = wolkes_data.merge(bayley_data, on='subjectID', how='outer')
comb = comb.merge(anthro_data, on='subjectID', how='outer')
comb = comb.set_index('subjectID')
regression_cols = ["cognitive_score_52", "vocalisation_52", "WLZ_WHZ_52"]
targets = comb[regression_cols].copy()

target_scaler = StandardScaler()
targets_array = target_scaler.fit_transform(targets)
targets = pd.DataFrame(targets_array, index=targets.index, columns=targets.columns)

print("\nTarget Statistics after scaling:")
for col in regression_cols:
    print(f"\n{col}:")
    print(f"  Mean: {targets[col].mean():.3f}") 
    print(f"  Std: {targets[col].std():.3f}") 
    print(f"  Min: {targets[col].min():.3f}")
    print(f"  Max: {targets[col].max():.3f}")

#targets_scaled.to_csv('../data/targets_scaled.tsv', sep='\t', index=True)

common_idx = targets.index.intersection(df_scaled.index)
targets_scaled = targets.loc[common_idx]
df_scaled = df_scaled.loc[common_idx]

joined = df_scaled.join(targets)
joined = joined.dropna(subset=regression_cols)


groups = joined.index
# First split: 70 % train vs 30 % (temp)
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=10)
train_idx, temp_idx = next(gss1.split(groups, groups=groups))

# Second split: split the 30 % temp into val / test (50 % each)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=10)
val_idx, test_idx = next(gss2.split(groups[temp_idx], groups=groups[temp_idx]))

# Map back to the original indices
train_idx = train_idx                # already global
val_idx   = temp_idx[val_idx]        # translate relative → absolute
test_idx  = temp_idx[test_idx]

def make_tensors(idx):
    X   = torch.tensor(joined.iloc[idx][df_scaled.columns].values, dtype=torch.float32)
    y_dict = {
        k: torch.tensor(joined[k].iloc[idx].values, dtype=torch.float32).unsqueeze(1)
        for k in regression_cols
    }
    return X, y_dict

X_train, y_train = make_tensors(train_idx)
X_val,   y_val   = make_tensors(val_idx)
X_test,  y_test  = make_tensors(test_idx)

def make_loader(X, y_dict, shuffle):
    ds = TensorDataset(X, *(y_dict[k] for k in regression_cols))
    return DataLoader(ds, batch_size=256, shuffle=shuffle, num_workers=4)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   shuffle=False)
test_loader  = make_loader(X_test,  y_test,  shuffle=False)

class VAEWithHeads(pl.LightningModule):
    def __init__(self, vae, important_dims=None):
        super().__init__()
        self.vae   = vae
        self.important_dims = important_dims
        effective_dims = len(important_dims) if important_dims is not None else LATENT_DIM
        # Regression heads
        self.reg_heads = nn.ModuleDict({
            "cognitive_score_52": nn.Sequential(
                nn.Linear(effective_dims, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            ),
            "vocalisation_52": nn.Sequential(
                nn.Linear(effective_dims, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            ),
            "WLZ_WHZ_52": nn.Sequential(
                nn.Linear(effective_dims, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
        })
        self.log_vars = nn.ParameterDict({
            'reg': nn.Parameter(torch.tensor(0.0))
        })
        self.task_log_vars = nn.ParameterDict({
            "cog": nn.Parameter(torch.zeros(1)),
            "voc": nn.Parameter(torch.zeros(1)),
            "wlz": nn.Parameter(torch.zeros(1)),
        })

    def forward(self, x):
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        if self.important_dims is not None:
            z = z[:, self.important_dims]
        preds = {k: head(z) for k, head in self.reg_heads.items()}
        return preds
    
    def step_common(self, batch):
        x, cog52, voc52, recovery = batch
        preds = self(x)

        #L1 reg
        l1_lambda = 1e-6
        wlz_l1 = l1_lambda * sum(p.abs().sum() for p in self.reg_heads["WLZ_WHZ_52"].parameters())
    

        # Separate losses for each target
        loss_cog = nn.functional.mse_loss(preds["cognitive_score_52"], cog52)
        loss_voc = nn.functional.mse_loss(preds["vocalisation_52"], voc52)
        loss_wlz = nn.functional.mse_loss(preds["WLZ_WHZ_52"], recovery) + wlz_l1
        
        # Total loss
        loss_reg = loss_cog + loss_voc + loss_wlz
        
        return loss_reg, loss_cog, loss_voc, loss_wlz, preds, recovery

    def training_step(self, batch, _):
        x, cog52, voc52, recovery = batch
        
        # Add noise augmentation
        if self.training:
            noise = torch.randn_like(x) * 0.1
            x = x + noise
        
        loss_reg, loss_cog, loss_voc, loss_wlz, _, _ = self.step_common((x, cog52, voc52, recovery))
        
        self.log_dict({
            "loss_reg_total": loss_reg,
            "loss_cognitive": loss_cog,
            "loss_vocalisation": loss_voc,
            "loss_wlz": loss_wlz
        })
        return loss_reg
    
    def validation_step(self, batch, _):
        loss_reg, loss_cog, loss_voc, loss_wlz, _, _ = self.step_common(batch)

        self.log_dict({
            "val_loss_reg_total": loss_reg,
            "val_loss_cognitive": loss_cog,
            "val_loss_vocalisation": loss_voc,
            "val_loss_wlz": loss_wlz
        })

    def configure_optimizers(self):
        vae_params = list(self.vae.parameters())
        head_params = list(self.reg_heads.parameters())
        optimizer = torch.optim.AdamW([
            {'params': vae_params,  'lr': 1e-5, 'weight_decay': 1e-6},
            {'params': head_params, 'lr': 3e-4, 'weight_decay': 1e-4}
        ])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-4,
            steps_per_epoch=len(train_loader),
            epochs=400
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # update every batch
                "frequency": 1
            }
        }
    

scaler = joblib.load('../data/scaler_large_nolip.save')
X = torch.tensor(df_scaled.values, dtype=torch.float32)    
INPUT_DIM   = X.shape[1]
LATENT_DIM  = 64
HIDDEN_DIM  = 1024
model = VAEWorld(INPUT_DIM, LATENT_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load('vae_world_large_mam_nolip_mse.pt', map_location='cpu'))
#model.eval()

model_heads = VAEWithHeads(model, important_dims=None)
# Freeze the VAE encoder / decoder
for p in model_heads.vae.parameters():
    p.requires_grad = False


trainer = pl.Trainer(
    max_epochs=400,
    log_every_n_steps=1,
    callbacks=[
        pl.callbacks.EarlyStopping(
            monitor='val_loss_wlz',
            mode='min',
            patience=15,
            min_delta=0.001
        )
    ],
    gradient_clip_val=0.5,
    gradient_clip_algorithm='norm',
    accumulate_grad_batches=2
)
trainer.fit(model_heads, train_loader, val_loader)

trainer.validate(model_heads, test_loader)

# Save the model with additional metadata
save_dict = {
    'reg_heads': model_heads.reg_heads.state_dict(),
    'log_vars': model_heads.log_vars.state_dict(),
    'input_dim': INPUT_DIM,
    'latent_dim': LATENT_DIM,
    'hidden_dim': HIDDEN_DIM,
    'target_scaler': target_scaler,
    'regression_cols': regression_cols,
    'validation_metrics': trainer.callback_metrics
}


# Save with timestamp and metrics
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# model_path = f'../models/vae_heads_reg_{timestamp}.pt'

# torch.save(save_dict, model_path)
# print(f"Model saved to: {model_path}")

if save:
    torch.save({
        'state_dict': model_heads.state_dict(),   # full VAE + heads
        'target_scaler': target_scaler,
        'regression_cols': regression_cols,
        'input_dim': INPUT_DIM,
        'latent_dim': LATENT_DIM,
        'hidden_dim': HIDDEN_DIM,
    }, '../models/vae_heads_nolip_mse_onecyclelr.pt')



# # Now do feature selection using only training data
# X_train_features = joined.iloc[train_idx][df_scaled.columns]
# y_train_wlz = joined.iloc[train_idx]['WLZ_WHZ_52']

# # Train RF on original features from training data only
# rf = RandomForestRegressor(n_estimators=100, random_state=10)
# rf.fit(X_train_features, y_train_wlz)

# # Get feature importance
# feature_importances = pd.Series(
#     rf.feature_importances_,
#     index=df_scaled.columns
# )

# print("\nTop 20 most important features for WLZ prediction:")
# print(feature_importances.nlargest(20))

# # Keep only features with importance > mean importance
# mean_importance = feature_importances.mean()
# important_features = feature_importances[feature_importances > mean_importance]
# print(f"\nKeeping {len(important_features)} features above mean importance")

# # Create new VAE model with reduced input dimension
# INPUT_DIM_SELECTED = len(important_features)

# # Create new datasets with only important features
# X_train_selected = torch.tensor(X_train_features[important_features.index].values, dtype=torch.float32)
# X_val_selected = torch.tensor(joined.iloc[val_idx][important_features.index].values, dtype=torch.float32)
# X_test_selected = torch.tensor(joined.iloc[test_idx][important_features.index].values, dtype=torch.float32)

# # Create new dataloaders with selected features
# train_loader_selected = make_loader(X_train_selected, y_train, shuffle=True)
# val_loader_selected = make_loader(X_val_selected, y_val, shuffle=False)
# test_loader_selected = make_loader(X_test_selected, y_test, shuffle=False)

# dataset_selected = TensorDataset(X_train_selected, X_train_selected)
# train_loader_vae = DataLoader(
#     dataset_selected,
#     batch_size=256,
#     shuffle=True,
#     num_workers=4
# )

# # Train the VAE
# model_selected = VAEWorld(INPUT_DIM_SELECTED, LATENT_DIM, HIDDEN_DIM)
# trainer_vae = pl.Trainer(
#     max_epochs=400,
#     accumulate_grad_batches=4,
#     gradient_clip_val=1.0,
#     gradient_clip_algorithm='norm'
# )
# trainer_vae.fit(model_selected, train_loader_vae)

# # Train new model with selected features
# model_heads_selected = VAEWithHeads(model_selected, important_dims=None)


# trainer_selected = pl.Trainer(
#     max_epochs=400,
#     log_every_n_steps=1,
#     callbacks=[
#         pl.callbacks.EarlyStopping(
#             monitor='val_loss_wlz',
#             mode='min',
#             patience=15,
#             min_delta=0.001
#         )
#     ],
#     gradient_clip_val=0.5,
#     gradient_clip_algorithm='norm',
#     accumulate_grad_batches=2
# )

# trainer_selected.fit(model_heads_selected, train_loader_selected, val_loader_selected)
# trainer_selected.validate(model_heads_selected, test_loader_selected)