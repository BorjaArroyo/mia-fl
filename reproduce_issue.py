
import torch
import numpy as np
import sys, os
from src.models.vae import VAE, vae_loss

def check_slope():
    print("--- Checking Error Slope ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # tiny model for speed
    model = VAE(num_channels=3, latent_dim=8, num_classes=2).to(device)
    model.eval()
    
    x = torch.randn(1, 3, 28, 28).to(device)
    y = torch.as_tensor([0]).to(device)
    
    # 1. Compute Gradient (with random seed A - implicit)
    model.zero_grad()
    # Implicit seed A
    recon_x, mu, logvar = model(x, y)
    loss_val, _, _ = vae_loss(recon_x, x, mu, logvar)
    loss_val.backward()
    
    grads = []
    for p in model.parameters():
        grads.append(p.grad.view(-1)) 
    grad_vec = torch.cat(grads)
    norm_grad = torch.norm(grad_vec)
    print(f"Gradient Norm: {norm_grad.item()}")
    
    u_dir = grad_vec / norm_grad
    linear_term_base = torch.dot(grad_vec, u_dir).item()
    
    params = []
    for p in model.parameters():
        params.append(p.view(-1))
    W_base_vec = torch.cat(params)
    
    def get_loss_at(w_vec, seed):
        rng_devices = [device] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed)
            offset = 0
            for p in model.parameters():
                num = p.numel()
                p.data.copy_(w_vec[offset:offset+num].view_as(p))
                offset += num
            recon_x, mu, logvar = model(x, y)
            loss, _, _ = vae_loss(recon_x, x, mu, logvar)
        return loss
        
    # 2. Check errors at different scales with DIFFERENT seed B
    scales = [1e-5, 1e-4, 1e-3, 1e-2]
    errors = []
    
    # Pick a fixed seed for the "Different Seed" case
    seed_B = 12345
    
    print("\n--- Testing Mismatched Seed (Current Bug) ---")
    for s in scales:
        u = u_dir * s
        
        # Calculate Delta with Seed B
        l_base = get_loss_at(W_base_vec, seed_B)
        l_pert = get_loss_at(W_base_vec + u, seed_B)
        delta = l_base.item() - l_pert.item() # Note: gain definition
        
        # Actually logic in code: delta_true = l_pert - loss_val_fixed
        delta_true = l_pert.item() - l_base.item()
        
        linear_approx = s * linear_term_base
        err = abs(delta_true - linear_approx)
        errors.append(err)
        print(f"Scale {s:.1e}: Error {err:.2e}")

    # Calculate slope
    log_scales = np.log(scales)
    log_errors = np.log(errors)
    slope, intercept = np.polyfit(log_scales, log_errors, 1)
    print(f"Slope (Mismatched): {slope:.4f} (Expected ~1.0)")
    
    # 3. Check errors with SAME seed (Fix)
    print("\n--- Testing Matched Seed (Proposed Fix) ---")
    
    # Reset weights to base!!
    offset = 0
    for p in model.parameters():
        num = p.numel()
        p.data.copy_(W_base_vec[offset:offset+num].view_as(p))
        offset += num
        
    # Recalculate gradient with Seed B
    model.zero_grad()
    rng_devices = [device] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed_B)
        # We need to reload weights to be sure? No, weights are same.
        # But we need to ensure the forward pass uses seed B
        recon_x, mu, logvar = model(x, y)
        loss_val_B, _, _ = vae_loss(recon_x, x, mu, logvar)
        loss_val_B.backward()
        
    grads_B = []
    for p in model.parameters():
        grads_B.append(p.grad.view(-1))
    grad_vec_B = torch.cat(grads_B)
    
    linear_term_base_B = torch.dot(grad_vec_B, u_dir).item()
    
    errors_fixed = []
    for s in scales:
        u = u_dir * s
        
        l_base = get_loss_at(W_base_vec, seed_B)
        l_pert = get_loss_at(W_base_vec + u, seed_B)
        delta_true = l_pert.item() - l_base.item()
        
        linear_approx = s * linear_term_base_B # Using correlated gradient!
        
        err = abs(delta_true - linear_approx)
        errors_fixed.append(err)
        print(f"Scale {s:.1e}: Error {err:.2e}")
        
    log_errors_fixed = np.log(errors_fixed)
    slope_fixed, _ = np.polyfit(log_scales, log_errors_fixed, 1)
    print(f"Slope (Matched): {slope_fixed:.4f} (Expected ~2.0)")

if __name__ == "__main__":
    check_slope()
