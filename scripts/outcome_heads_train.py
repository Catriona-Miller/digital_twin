from sklearn.model_selection import GroupShuffleSplit
import torch, pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from format_matrix import combine_df
from sklearn.preprocessing import StandardScaler
from vae import VAEWorld
from torch import nn
import joblib
from pytorch_lightning.callbacks import EarlyStopping

df_scaled = pd.read_csv('../data/combined_imputed_scaled.tsv', sep='\t', index_col=0)
wolkes_data = pd.read_csv('../data/wolkes.tsv', sep='\t')
bayley_data = pd.read_csv('../data/bayley.tsv', sep='\t')

wolkes_data = combine_df(wolkes_data)
bayley_data = combine_df(bayley_data)
comb = wolkes_data.merge(bayley_data, on='subjectID', how='outer')
comb = comb.set_index('subjectID')
target_cols = ['cognitive_score_52', 'cognitive_score_0', 'vocalisation_0', 'vocalisation_52']
targets = comb[target_cols].copy()

target_scaler = StandardScaler()
targets_scaled = pd.DataFrame(
    target_scaler.fit_transform(targets),
    index=targets.index,
    columns=targets.columns
)

targets_scaled.to_csv('../data/targets_scaled.tsv', sep='\t', index=True)

common_idx = targets_scaled.index.intersection(df_scaled.index)
targets_scaled = targets_scaled.loc[common_idx]
df_scaled = df_scaled.loc[common_idx]

joined = df_scaled.join(targets_scaled)

# 2. drop any row with a missing target
joined = joined.dropna(subset=target_cols)

# 3. split (the groups are now the filtered subject IDs)
groups = joined.index
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
train_idx, test_idx = next(gss.split(groups, groups=groups))

# 4. build tensors
X_train = torch.tensor(joined.iloc[train_idx][df_scaled.columns].values, dtype=torch.float32)
X_test  = torch.tensor(joined.iloc[test_idx][df_scaled.columns].values,  dtype=torch.float32)

y_train = {k: torch.tensor(joined[k].iloc[train_idx].values, dtype=torch.float32).unsqueeze(1)
           for k in target_cols}
y_test  = {k: torch.tensor(joined[k].iloc[test_idx].values,  dtype=torch.float32).unsqueeze(1)
           for k in target_cols}

train_ds = TensorDataset(X_train, *(y_train[k] for k in target_cols))
test_ds  = TensorDataset(X_test,  *(y_test[k]  for k in target_cols))

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4)

class VAEWithHeads(pl.LightningModule):
    def __init__(self, vae):
        super().__init__()
        self.vae   = vae
        self.heads = nn.ModuleDict({
            k: nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
            ) for k in target_cols
                })

    def forward(self, x):
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        return {k: h(z) for k, h in self.heads.items()}

    def training_step(self, batch, _):
        x, cog52, cog0, voc0, voc52 = batch
        preds = self(x)
        loss  = (nn.functional.mse_loss(preds["cognitive_score_52"], cog52) +
                nn.functional.mse_loss(preds["cognitive_score_0"],  cog0) +
                nn.functional.mse_loss(preds["vocalisation_0"],     voc0) +
                nn.functional.mse_loss(preds["vocalisation_52"],    voc52))
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, _):
        x, cog52, cog0, voc0, voc52 = batch
        preds = self(x)
        loss  = (
            nn.functional.mse_loss(preds["cognitive_score_52"], cog52) +
            nn.functional.mse_loss(preds["cognitive_score_0"],  cog0) +
            nn.functional.mse_loss(preds["vocalisation_0"],     voc0) +
            nn.functional.mse_loss(preds["vocalisation_52"],    voc52)
        )
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.heads.parameters(), lr=3e-4, weight_decay=1e-5)
    

scaler = joblib.load('../data/scaler.save')
X = torch.tensor(df_scaled.values, dtype=torch.float32)    
INPUT_DIM   = X.shape[1]
LATENT_DIM  = 32 
HIDDEN_DIM  = 512
model = VAEWorld(INPUT_DIM, LATENT_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load('vae_world.pt', map_location='cpu'))
#model.eval()
model.mu.weight.requires_grad     = True
model.mu.bias.requires_grad       = True
model.logvar.weight.requires_grad = True
model.logvar.bias.requires_grad   = True

model_heads = VAEWithHeads(model)

# 5.  Train
trainer = pl.Trainer(max_epochs=400)
trainer.fit(model_heads, train_loader)

# 6.  Evaluate
trainer.validate(model_heads, test_loader)
# 7.  Save the model
torch.save(model_heads.heads.state_dict(), 'vae_heads_train.pt')