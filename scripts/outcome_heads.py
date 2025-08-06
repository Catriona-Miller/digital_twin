import joblib
import torch
import pytorch_lightning as pl
from vae import VAEWorld
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from format_matrix import combine_df
from sklearn.preprocessing import StandardScaler
from torch import nn


df_scaled = pd.read_csv('../data/combined_imputed_scaled.tsv', sep='\t', index_col=0)
scaler = joblib.load('../data/scaler.save')
X = torch.tensor(df_scaled.values, dtype=torch.float32)
dataset = TensorDataset(X, X)

INPUT_DIM   = X.shape[1]
LATENT_DIM  = 32 
HIDDEN_DIM  = 512
model = VAEWorld(INPUT_DIM, LATENT_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load('vae_world.pt', map_location='cpu'))
model.eval()

class VAEWithHeads(pl.LightningModule):
    def __init__(self, vae):
        super().__init__()
        self.vae   = vae
        self.heads = nn.ModuleDict({
            "cognitive_score_52": nn.Linear(LATENT_DIM, 1),
            "cognitive_score_0":      nn.Linear(LATENT_DIM, 1),
            "vocalisation_0":      nn.Linear(LATENT_DIM, 1),
            "vocalisation_52":      nn.Linear(LATENT_DIM, 1)
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.heads.parameters(), lr=1e-3)
    
wolkes_data = pd.read_csv('../data/wolkes.tsv', sep='\t')
bayley_data = pd.read_csv('../data/bayley.tsv', sep='\t')

wolkes_data = combine_df(wolkes_data)
bayley_data = combine_df(bayley_data)
comb = wolkes_data.merge(bayley_data, on='subjectID', how='outer')
comb = comb.set_index('subjectID')
targets = comb[['cognitive_score_52', 'cognitive_score_0', 'vocalisation_0', 'vocalisation_52']].copy()

target_scaler = StandardScaler()
targets_scaled = pd.DataFrame(
    target_scaler.fit_transform(targets),
    index=targets.index,
    columns=targets.columns
)

targets_scaled.to_csv('../data/targets_scaled.tsv', sep='\t', index=True)
joblib.dump(target_scaler, 'target_scaler.save')

common_idx = targets_scaled.index.intersection(df_scaled.index)
X = torch.tensor(df_scaled.loc[common_idx].values, dtype=torch.float32)
y_dict = {
    k: torch.tensor(targets_scaled[k].loc[common_idx].values, dtype=torch.float32).unsqueeze(1)
    for k in targets_scaled.columns
}

dataset = TensorDataset(X, *(y_dict[k] for k in targets_scaled.columns))
loader = DataLoader(dataset, batch_size=256, shuffle=True)
model_with_heads = VAEWithHeads(model)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model_with_heads, loader)

torch.save(model_with_heads.heads.state_dict(), 'vae_heads.pt')

