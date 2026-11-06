import torch
import torch.nn as nn

from train_flow import FlowMLP 

class FlowPriorSampler:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.model = FlowMLP().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def sample(self, condition_c, num_steps=20):
        """
        Performs and solves the Flow ODE using the Euler method.
        condition_c: FaRL feature tensor of shape (Batch, 512) to guide
        Returns: StyleGAN2 latent slice of shape (Batch, 5, 512)
        """
        B = condition_c.shape[0]    # num batches
        
        # Start from pure Gaussian noise (x_0)
        x = torch.randn((B, 2560), device=self.device)
        
        dt = 1.0 / num_steps
        
        # Euler ODE Integration
        for i in range(num_steps):
            # Current time t
            t_val = i * dt
            t_tensor = torch.full((B, 1), t_val, device=self.device)
            
            # Predict velocity
            v_pred = self.model(x, t_tensor, condition_c)
            
            # Step forward
            x = x + v_pred * dt
            
        # Reshape the final flattened vector back into the 5-layer slice
        # Target shape -> [Batch size, 5 layers, 512 dimensions]
        w_slice = x.reshape(B, 5, 512)
        
        return w_slice