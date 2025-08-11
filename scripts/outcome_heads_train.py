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

df_scaled = pd.read_csv('../data/combined_imputed_scaled_large.tsv', sep='\t', index_col=0)
wolkes_data = pd.read_csv('../data/wolkes.tsv', sep='\t')
bayley_data = pd.read_csv('../data/bayley.tsv', sep='\t')
meta_data = pd.read_csv('../data/meta.tsv', sep='\t')
# get just the mam kids
df_scaled = df_scaled[df_scaled.index.isin(meta_data[meta_data['Condition'] == 'MAM']['subjectID'])]

wolkes_data = combine_df(wolkes_data)
bayley_data = combine_df(bayley_data)
comb = wolkes_data.merge(bayley_data, on='subjectID', how='outer')
comb = comb.merge(meta_data, on='subjectID', how='outer')
comb = comb.set_index('subjectID')
regression_cols = ["cognitive_score_52", "cognitive_score_0"]
classification_col = ["Recovery"]

# Encode Condition as 0/1 before scaling
targets = comb[regression_cols + classification_col].copy()
#targets["Condition"] = targets["Condition"].apply(lambda x: 1 if x == "MAM" else 0)
targets["Recovery"] = targets["Recovery"].apply(lambda x: 0 if x == "No recovery" else 1) 

# Scale only the regression columns
target_scaler = StandardScaler()
targets[regression_cols] = target_scaler.fit_transform(targets[regression_cols])

#targets_scaled.to_csv('../data/targets_scaled.tsv', sep='\t', index=True)

common_idx = targets.index.intersection(df_scaled.index)
targets_scaled = targets.loc[common_idx]
df_scaled = df_scaled.loc[common_idx]

joined = df_scaled.join(targets)
joined = joined.dropna(subset=regression_cols + classification_col)


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
        for k in regression_cols + classification_col
    }
    return X, y_dict

X_train, y_train = make_tensors(train_idx)
X_val,   y_val   = make_tensors(val_idx)
X_test,  y_test  = make_tensors(test_idx)

def make_loader(X, y_dict, shuffle):
    ds = TensorDataset(X, *(y_dict[k] for k in regression_cols + classification_col))
    return DataLoader(ds, batch_size=256, shuffle=shuffle, num_workers=4)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   shuffle=False)
test_loader  = make_loader(X_test,  y_test,  shuffle=False)

cls_weight = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=joined["Recovery"].values)
cls_weight = torch.tensor(cls_weight, dtype=torch.float32)


class VAEWithHeads(pl.LightningModule):
    def __init__(self, vae):
        super().__init__()
        self.vae   = vae
        # Regression heads
        self.reg_heads = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(LATENT_DIM, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
            for k in ["cognitive_score_52", "cognitive_score_0"]
        })

        # Binary classification head
        self.cls_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # logits
        )
        self.log_vars = nn.ParameterDict({
            'reg': nn.Parameter(torch.tensor(0.0)),
            'cls': nn.Parameter(torch.tensor(0.0))
        })

    def forward(self, x):
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        preds = {k: head(z) for k, head in self.reg_heads.items()}
        preds["Recovery"] = self.cls_head(z)
        return preds
    
    def step_common(self, batch):
        x, cog52, cog0, recovery = batch
        preds = self(x)

        # Regression loss
        loss_reg = (
            nn.functional.mse_loss(preds["cognitive_score_52"], cog52) +
            nn.functional.mse_loss(preds["cognitive_score_0"],  cog0)
        )
        loss_cls = nn.functional.binary_cross_entropy_with_logits(
            preds["Recovery"], 
            recovery.float(),
            pos_weight=cls_weight[1] 
        )

        total = (
            0.3 * loss_reg * torch.exp(-self.log_vars['reg']) + self.log_vars['reg'] +
            0.7 * loss_cls * torch.exp(-self.log_vars['cls']) + self.log_vars['cls']
        )
        return total, loss_reg, loss_cls, preds, recovery

    def training_step(self, batch, _):
        loss, loss_reg, loss_cls, _, _ = self.step_common(batch)
        self.log("loss_total", loss)
        self.log("loss_reg", loss_reg)
        self.log("loss_cls", loss_cls)
        return loss
    
    def validation_step(self, batch, _):
        loss, loss_reg, loss_cls, preds, recovery = self.step_common(batch)

        # Compute predictions
        probs = torch.sigmoid(preds["Recovery"])
        pred_labels = (probs > 0.5).long()
        
        # Multiple metrics
        acc = (pred_labels.squeeze() == recovery).float().mean()
        tp = ((pred_labels.squeeze() == 1) & (recovery == 1)).float().sum()
        fp = ((pred_labels.squeeze() == 1) & (recovery == 0)).float().sum()
        fn = ((pred_labels.squeeze() == 0) & (recovery == 1)).float().sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        self.log_dict({
            "val_loss": loss,
            "val_loss_reg": loss_reg,
            "val_loss_cls": loss_cls,
            "val_acc": acc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1
        })

    def configure_optimizers(self):
        vae_params = list(self.vae.parameters())
        head_params = list(self.reg_heads.parameters()) + list(self.cls_head.parameters())
        optimizer = torch.optim.Adam([
            {'params': vae_params,  'lr': 1e-5},
            {'params': head_params, 'lr': 1e-3, 'weight_decay': 1e-4}
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True, threshold=0.001
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_acc"}
        }
    

scaler = joblib.load('../data/scaler_large.save')
X = torch.tensor(df_scaled.values, dtype=torch.float32)    
INPUT_DIM   = X.shape[1]
LATENT_DIM  = 64 
HIDDEN_DIM  = 1024
model = VAEWorld(INPUT_DIM, LATENT_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load('vae_world_large_mam.pt', map_location='cpu'))
#model.eval()

model_heads = VAEWithHeads(model)

# 5.  Train
trainer = pl.Trainer(
    max_epochs=400,
    log_every_n_steps=1,
    callbacks=[
        pl.callbacks.EarlyStopping(
            monitor='val_f1',
            mode='max',
            patience=20,
            min_delta=0.001
        )
    ],
    gradient_clip_val=1.0,
    gradient_clip_algorithm='norm'
)
trainer.fit(model_heads, train_loader, val_loader)

# 6.  Evaluate
trainer.validate(model_heads, test_loader)
# 7.  Save the model
# torch.save({
#     'reg_heads': model_heads.reg_heads.state_dict(),
#     'cls_head':  model_heads.cls_head.state_dict(),
#     'log_vars':  model_heads.log_vars.state_dict()
# }, 'vae_heads_train_large.pt')