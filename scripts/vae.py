import pandas as pd
import torch, pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
pl.seed_everything(10)


# Load the data
df_scaled = pd.read_csv('../data/combined_imputed_scaled_large_nolip.tsv', sep='\t', index_col=0)
# get just the mam kids
meta_data = pd.read_csv('../data/meta.tsv', sep='\t')
df_scaled = df_scaled[df_scaled.index.isin(meta_data[meta_data['Condition'] == 'MAM']['subjectID'])]

X = torch.tensor(df_scaled.values, dtype=torch.float32)
dataset = TensorDataset(X, X)

INPUT_DIM   = X.shape[1]
LATENT_DIM  = 64
HIDDEN_DIM  = 1024
BETA        = 1e-5
BATCH_SIZE  = 256
MAX_EPOCHS  = 400

class VAEWorld(pl.LightningModule):
    def __init__(self, in_dim, latent, hidden):
        super().__init__()
        self.save_hyperparameters()
        # encoder
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.mu     = nn.Linear(hidden//2, latent)
        self.logvar = nn.Linear(hidden//2, latent)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, in_dim)
        )

    def encode(self, x):
        h = self.enc(x)
        mu    = self.mu(h)
        logvar = torch.clamp(self.logvar(h), min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_fn(self, x, x_hat, mu, logvar):
        recon = nn.functional.mse_loss(x_hat, x, reduction='mean')
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return recon + BETA * kl, {'recon': recon, 'kl': kl}

    def training_step(self, batch, _):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss, logs = self.loss_fn(x, x_hat, mu, logvar)
        self.log_dict(logs, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
# BATCH_SIZE = min(256, len(dataset))
# train_loader = DataLoader(dataset,
#                           batch_size=BATCH_SIZE,
#                           shuffle=True,
#                           drop_last=False,
#                           num_workers=0)    

# print(len(train_loader.dataset), BATCH_SIZE)
# assert BATCH_SIZE >= 4

# model = VAEWorld(INPUT_DIM, LATENT_DIM, HIDDEN_DIM)
# trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accumulate_grad_batches=4,
#                      accelerator='auto', devices='auto',
#                      enable_progress_bar=True, gradient_clip_val=1.0, gradient_clip_algorithm='norm',)
# trainer.fit(model, train_loader)

# model.eval()
# with torch.no_grad():
#     x_sample = X[:8]
#     x_rec, _, _ = model(x_sample)
#     print("Reconstruction MSE:", nn.functional.mse_loss(x_rec, x_sample).item())

# torch.save(model.state_dict(), "vae_world_large_mam_nolip.pt")