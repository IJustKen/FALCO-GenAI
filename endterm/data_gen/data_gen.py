import torch
import os
from tqdm import tqdm
from lib import AttrLoss
from models.load_generator import load_generator
import sys

def generate_flow_dataset(save_path, num_samples=10, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading StyleGAN2...")
    # load stylegan2
    G = load_generator(model_name='stylegan2_ffhq1024', latent_is_w=True, verbose=False).eval().to(device)

    print("Loading FaRL Module...")
    # load farl model
    farl_module = AttrLoss(feat_ext='farl', use_cuda=True).eval()

    all_c = []
    all_x1 = []

    print(f"Generating {num_samples} training pairs...")
    with torch.no_grad():
        for _ in tqdm(range(0, num_samples, batch_size)):
            current_batch_size = min(batch_size, num_samples - len(all_c) * batch_size)
            # when remaining points are less than batch size
            
            # Generate random Z noise, 512 dim for the batch
            z = torch.randn([current_batch_size, 512], device=device)
            
            # Map Z to W space (18 vectors each of 512 dim)
            # their code gives a get_w method to get the w
            wp = G.get_w(z, truncation=1.0) 
            
            # Extract the target x1 (layers 3 to 7) and flatten it
            w_slice = wp[:, 3:8, :]
            x1_flat = w_slice.reshape(current_batch_size, -1)
            
            # Generate the synthetic image 
            # Directly call the synthesis module using the W+ latents
            # synthesis returns a dictionary, so we get ['image']
            img = G.synthesis(wp=wp, randomize_noise=False)['image']
            
            # Apply the exact transform defined inside their AttrLoss class
            img_transformed = farl_module.feat_ext_transform(img)
            
            # Extract the FaRL condition c (512 dims)
            c, _ = farl_module.extract_visual(img_transformed)
            
            # Move to CPU immediately to prevent GPU OOM errors
            all_c.append(c.cpu())
            all_x1.append(x1_flat.cpu())

    # Concatenate all batches into large tensors
    tensor_c = torch.cat(all_c, dim=0)
    tensor_x1 = torch.cat(all_x1, dim=0)

    print(f"Dataset generated. Condition shape: {tensor_c.shape}, Target shape: {tensor_x1.shape}")
    
    # Save as a single dictionary
    torch.save({'c': tensor_c, 'x1': tensor_x1}, save_path)
    print(f"Successfully saved to {save_path}")

if __name__ == "__main__":
    generate_flow_dataset(num_samples=10000, batch_size=8, save_path='endterm/data/flow_train_data.pt')