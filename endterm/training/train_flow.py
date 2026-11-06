import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
chk_path = PROJECT_ROOT / "checkpoints"
chk_path.mkdir(parents=True, exist_ok=True)
    
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t is shape [Batch, 1]
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class FlowMLP(nn.Module):
    def __init__(self, x_dim=2560, c_dim=512, hidden_dim=4096, t_emb_dim=256):
        super().__init__()
        
        # Proper diffusion-style time embeddings
        self.time_embed = SinusoidalTimeEmbedding(t_emb_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 2, t_emb_dim)
        )
        
        # No more bottleneck! hidden_dim is 4096 > 2560
        in_dim = x_dim + c_dim + t_emb_dim
        
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, x_dim) # Output is 2560
        
        self.act = nn.SiLU()

    def forward(self, x_t, t, c):
        # Get proper time embeddings
        t_emb = self.t_mlp(self.time_embed(t))
        
        # Concatenate x_t, condition c, and time embedding
        h = torch.cat([x_t, c, t_emb], dim=-1)
        
        # Forward pass with residual/skip connection to help gradients flow
        h = self.act(self.layer1(h))
        
        h_res = self.act(self.layer2(h))
        h = h + h_res  # Skip connection
        
        h_res2 = self.act(self.layer3(h))
        h = h + h_res2 # Skip connection
        
        v_pred = self.layer4(h)
        return v_pred

# # The Flow Matching Model
# class FlowMLP(nn.Module):
#     # takes the flattened vector of the sliced w vector and also the farl features 512 dim
#     def __init__(self, x_dim=2560, c_dim=512, hidden_dim=4096, t_emb_dim=256):
#         super().__init__()
        
#         # Simple time embedding -> Project scalar t [0, 1] into a higher dimensional space
#         self.t_mlp = nn.Sequential(
#             nn.Linear(1, t_emb_dim),
#             nn.SiLU(),
#             nn.Linear(t_emb_dim, t_emb_dim)
#         )
        
#         # Concatenating x_t, condition c, and time embedding
#         in_dim = x_dim + c_dim + t_emb_dim  # simple concatenation
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, x_dim) # Outputs the predicted velocity
#         )

#     def forward(self, x_t, t, c):
#         # x_t: [Batch, 2560]
#         # t: [Batch, 1]
#         # c: [Batch, 512]
        
#         t_emb = self.t_mlp(t)
        
#         # Concatenate everything together
#         h = torch.cat([x_t, c, t_emb], dim=-1)
        
#         # Predict velocity
#         v_pred = self.net(h)
#         return v_pred


# Dataset Loader wrapper, using the Dataset class
class FlowDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading data from {data_path}...")
        data = torch.load(data_path)
        self.c = data['c'].float()
        self.x1 = data['x1'].float()
        print(f"Loaded {len(self.c)} pairs.")

    def __len__(self):
        return len(self.c)

    def __getitem__(self, idx):
        return self.c[idx], self.x1[idx]


# Training Loop
def train(data_path:str):
    # take in the path to the data file .pt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 128
    epochs = 100
    lr = 1e-3
    
    # Setup Data
    dataset = FlowDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # Setup Model & Optimizer
    model = FlowMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    print("Starting Flow Training...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # tqdm for a clean progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for c, x1 in pbar:
            c = c.to(device)
            x1 = x1.to(device)
            B = x1.shape[0]
            #print(x1.shape)
            #print(c.shape)
            
            # Sample Noise (x0)
            x0 = torch.randn_like(x1)
            
            # Sample random time steps t ~ Uniform[0, 1]
            t = torch.rand((B, 1), device=device)
            
            # Construct the straight-line path x_t
            # Equation => x_t = t * x1 + (1 - t) * x0
            x_t = t * x1 + (1.0 - t) * x0
            
            # The ground truth velocity v
            # Equation: v = x1 - x0
            v_target = x1 - x0
            
            # Predict velocity with the network
            v_pred = model(x_t, t, c)
            
            # Get loss
            loss = criterion(v_pred, v_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Completed | Avg MSE Loss: {avg_loss:.4f}")
        
        # Save checkpoints periodically
        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            ckpt_path = chk_path / f"flow_model_ep{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train()