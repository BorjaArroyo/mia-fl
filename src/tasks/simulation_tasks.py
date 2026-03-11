import ray
import os
import sys

# @ray.remote
def verify_stage_one(n_trials=8000, d=100):
    import numpy as np
    import sys, os
    # Ensure working directory is in sys.path
    root_dir = os.getcwd()
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    print(f"--- Verifying Stage I: Signal Survival (CPU) --- (CWD: {root_dir})")
    
    def _random_unit_vector(d):
        v = np.random.randn(d)
        return v / np.linalg.norm(v)

    data_points = []
    errors_energy = []
    errors_sur = []
    
    # Use grid sampling for better heatmap visualization
    # We want to cover (lambda, r) space densely
    # Expanded grid limits per user feedback ("edges bias")
    r_vals = np.linspace(0.01, 16.0, 80)
    lambda_vals = np.linspace(-1.0, 1.0, 80)
    
    # Create meshgrid
    rr, ll = np.meshgrid(r_vals, lambda_vals)
    r_flat = rr.flatten()
    l_flat = ll.flatten()
    
    # If n_trials is small, we might subsample, but let's try to use the grid
    # If the user passed a specific n_trials that is small, we respect it by random sampling
    # But for "verification" and plotting, grid is better.
    # Let's fallback to random if n_trials is significantly smaller than grid size
    use_grid = len(r_flat) <= n_trials * 2 
    
    if use_grid:
        print(f"Using Grid Sampling for Stage I ({len(r_flat)} points)")
        iterations = range(len(r_flat))
    else:
        print(f"Using Random Sampling for Stage I ({n_trials} trials)")
        iterations = range(n_trials)
    
    for i in iterations:
        if use_grid:
            r = r_flat[i]
            target_lambda = l_flat[i]
        else:
            r = np.random.uniform(0.01, 16.0) 
            target_lambda = np.random.uniform(-1.0, 1.0)
        
        u_dir = _random_unit_vector(d)
        v_rand = np.random.randn(d)
        v_orth = v_rand - np.dot(v_rand, u_dir) * u_dir
        v_orth /= np.linalg.norm(v_orth)
        
        B_dir = target_lambda * u_dir + np.sqrt(1 - target_lambda**2) * v_orth
        
        u = u_dir
        B = B_dir * r
        
        actual_lambda = np.dot(u, B) / (np.linalg.norm(u) * np.linalg.norm(B))
        
        DeltaW = u + B
        energy_lhs = np.linalg.norm(DeltaW)**2
        energy_rhs = np.linalg.norm(u)**2 + np.linalg.norm(B)**2 + 2 * np.linalg.norm(u) * np.linalg.norm(B) * actual_lambda
        errors_energy.append(abs(energy_lhs - energy_rhs))
        
        if abs(np.linalg.norm(DeltaW)) > 1e-9:
            sur_lhs_sq = np.linalg.norm(u)**2 / np.linalg.norm(DeltaW)**2
            sur_rhs_sq = 1 / (1 + r**2 + 2*r*actual_lambda)
            errors_sur.append(abs(sur_lhs_sq - sur_rhs_sq))
            
            data_points.append({
                'lambda': float(actual_lambda),
                'r': float(r),
                'sur_sq': float(sur_lhs_sq)
            })
        
    print(f"Stage I Max Energy Error: {max(errors_energy):.2e}")
    if errors_sur:
        print(f"Stage I Max SUR Error: {max(errors_sur):.2e}")
    
    return {
        'max_energy_error': float(max(errors_energy)),
        'max_sur_error': float(max(errors_sur)) if errors_sur else 0.0,
        'data_sample': data_points # Return all points for heatmap
    }

# @ray.remote(num_gpus=1)
def verify_stage_two(n_trials=50, in_channels=3, latent_dim=32, num_classes=10):
    import torch
    import numpy as np
    import sys, os
    from src.models.vae import VAE, vae_loss

    root_dir = os.getcwd()
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    print(f"--- Verifying Stage II: Signal Attribution (GPU) --- (CWD: {root_dir})")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Stage II running on {device}")

    try:
        from src.data.datasets import _load_medmnist_data
        # Increase batch size to have more diverse samples
        BATCH_SIZE = max(128, n_trials * 2) 
        loader, _ = _load_medmnist_data('pathmnist', batch_size=BATCH_SIZE, shuffle=True, train=True, num_workers=0)
        # Iterate to get a full batch
        X_batch, y_batch = next(iter(loader))
        X_batch = X_batch.float() / 255.0
        y_batch = y_batch.flatten().long()
    except Exception as e:
        raise RuntimeError(f"MedMNIST data loading failed in Stage II: {e}. Cannot proceed without real data.") from e

    model = VAE(num_channels=in_channels, latent_dim=latent_dim, num_classes=num_classes).to(device)
    model.eval() # Ensure deterministic behavior (disable dropout)
    
    # --- Calibration/Test split approach for L estimation ---
    # For each trial (data point x), we:
    #   1. Compute errors at N_SCALES perturbation scales
    #   2. Split into calibration (odd indices) and test (even indices)
    #   3. Estimate L_emp from calibration: L_cal = max(2*err/||u||^2)
    #   4. Verify bound holds on test set using L_cal
    
    N_SCALES = 30
    trial_summaries = []  # per-trial summary for bootstrapping
    all_points = []       # all individual data points for plotting
    
    # Ensure we don't exceed batch size
    n_trials = min(n_trials, len(X_batch))
    
    for trial_idx in range(n_trials):
        # Use different image for each trial
        # Use double precision for verification to avoid numerical noise floor at small scales
        x = X_batch[trial_idx].unsqueeze(0).to(device).double()
        y = y_batch[trial_idx].unsqueeze(0).to(device)
        
        # Cast model to double
        model.double()
        
        params = []
        for p in model.parameters():
            params.append(p.view(-1))
        W_base_vec = torch.cat(params)
        
        # Use a fixed seed for this trial's perturbations AND gradient
        trial_seed = int(torch.randint(0, 1000000, (1,)).item())
        
        # 1. Compute Gradient with FIXED seed
        rng_devices = [device] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(trial_seed)
            model.zero_grad()
            recon_x, mu, logvar = model(x, y)
            loss_val, _, _ = vae_loss(recon_x, x, mu, logvar)
            loss_val.backward()
        
        grads = []
        for p in model.parameters():
            grads.append(p.grad.view(-1)) 
        grad_vec = torch.cat(grads)
        
        norm_grad = torch.norm(grad_vec)
        if norm_grad < 1e-9: continue
        
        u_dir = grad_vec / norm_grad
        
        def get_loss_at(w_vec, seed):
            # Enforce same VAE noise by setting seed
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
        
        # Note: We interpret the generative score as log-likelihood, i.e., score = -loss.
        # grad_vec is gradient of loss; gradient of score is -grad_vec.
        linear_term_base = torch.dot(grad_vec, u_dir).item()
        
        def safe_phi(z):
            if z >= 0: return 1.0 / (1.0 + np.exp(-z)) - 0.5
            else: return np.exp(z) / (1.0 + np.exp(z)) - 0.5
        
        # Compute errors at N_SCALES perturbation scales
        # Scale relative to 1/||grad|| so target_gain ≈ linear response magnitude
        grad_scale = 1.0 / (norm_grad.item() + 1e-9)
        # Randomize target gains to avoid vertical clustering
        target_gains = np.sort(np.random.uniform(0.01, 2.0, N_SCALES))
        
        scale_data = []  # (scale, err, err/scale^2) for this trial
        for i, target_gain in enumerate(target_gains):
            scale = target_gain * grad_scale
            u = u_dir * scale
            
            # Recalculate base loss with the fixed seed to be consistent
            try:
                loss_val_fixed = get_loss_at(W_base_vec, trial_seed)
                l_pert = get_loss_at(W_base_vec + u, trial_seed)
            except Exception as e:
                print(f"Error in loss calc: {e}")
                continue

            delta_loss = l_pert.item() - loss_val_fixed.item()
            # Score increment uses -loss, so flip sign for consistency with the report.
            delta_true = -delta_loss
            linear_approx = -scale * linear_term_base
            err = abs(delta_true - linear_approx)
            
            phi_delta = safe_phi(delta_true)
            phi_linear = safe_phi(linear_approx)
            
            scale_data.append({
                'index': i,
                'scale': scale,
                'err': err,
                'ratio': 2.0 * err / (scale**2) if scale > 0 else 0.0,  # = empirical L
                'diff_phi': abs(phi_delta - phi_linear),
            })
        
        # Restore model weights (seed doesn't matter here)
        get_loss_at(W_base_vec, 0)
        
        # Split: odd indices = calibration, even indices = test
        # Filter out points with tiny linear approximation (numerical noise regime) from calibration
        cal_set = [d for d in scale_data if d['index'] % 2 == 1 and abs(d['err']) > 1e-8]
        test_set = [d for d in scale_data if d['index'] % 2 == 0]
        
        # Estimate L from calibration set
        L_cal = max(d['ratio'] for d in cal_set) if cal_set else 0.0
        
        # Verify bound on test set
        n_test = len(test_set)
        n_violations = 0
        for d in test_set:
            bound_lemma = 0.5 * L_cal * (d['scale']**2)        # Lemma 3.1: L/2 ||u||^2
            bound_thm = 0.125 * L_cal * (d['scale']**2)        # Thm 3.2:  L/8 ||u||^2
            violated = d['err'] > bound_lemma
            if violated:
                n_violations += 1
            
            all_points.append({
                'norm_u': float(d['scale']),
                'error_quad': float(d['err']),
                'bound_quad': float(bound_lemma),
                'diff_phi': float(d['diff_phi']),
                'bound_phi': float(bound_thm),
                'trial': trial_idx,
                'split': 'test',
            })
        
        # Also record calibration points (with their own bound for plotting)
        for d in cal_set:
            bound_lemma = 0.5 * L_cal * (d['scale']**2)
            bound_thm = 0.125 * L_cal * (d['scale']**2)
            all_points.append({
                'norm_u': float(d['scale']),
                'error_quad': float(d['err']),
                'bound_quad': float(bound_lemma),
                'diff_phi': float(d['diff_phi']),
                'bound_phi': float(bound_thm),
                'trial': trial_idx,
                'split': 'calibration',
            })
        
        trial_summaries.append({
            'trial': trial_idx,
            'L_empirical': float(L_cal),
            'n_test': n_test,
            'n_violations': n_violations,
            'violation_rate': float(n_violations / n_test) if n_test > 0 else 0.0,
        })
    
    overall_violations = sum(t['n_violations'] for t in trial_summaries)
    overall_total = sum(t['n_test'] for t in trial_summaries)

    # Wilson score interval (binomial) for the pooled test-point violation rate
    if overall_total > 0:
        z = 1.96
        p_hat = overall_violations / overall_total
        denom = 1.0 + (z**2) / overall_total
        center = (p_hat + (z**2) / (2.0 * overall_total)) / denom
        half = (z * np.sqrt((p_hat * (1 - p_hat) / overall_total) + (z**2) / (4.0 * overall_total**2))) / denom
        violation_ci_95 = [float(max(0.0, center - half)), float(min(1.0, center + half))]
    else:
        violation_ci_95 = [0.0, 0.0]
    
    summary = {
        'method': 'calibration_test_split',
        'n_trials': int(len(trial_summaries)),
        'n_scales': int(N_SCALES),
        'overall_violation_rate': float(overall_violations / overall_total) if overall_total > 0 else 0.0,
        'violation_ci_95': violation_ci_95,
        'median_L_empirical': float(np.median([t['L_empirical'] for t in trial_summaries])) if trial_summaries else 0.0,
    }
    
    print(f"Stage II verification: {overall_violations}/{overall_total} test violations "
          f"(rate={summary['overall_violation_rate']:.1%}, 95% CI=[{summary['violation_ci_95'][0]:.1%}, {summary['violation_ci_95'][1]:.1%}])")
    print(f"Median empirical L: {summary['median_L_empirical']:.2e}")
    
    # Convert all_points to pure python types if they aren't already
    # They seem to be constructed with float(), but let's be safe
    
    # Convert model back to float to save memory/compute if needed elsewhere (though we are done)
    # model.float()

    return {
        'points': all_points,
        'trial_summaries': trial_summaries,
        'summary': summary,
    }

# @ray.remote(num_gpus=1)
def generate_stage_two_geometry(n_points=50, in_channels=3, latent_dim=32, num_classes=10):
    import torch
    import numpy as np
    import sys, os
    from src.models.vae import VAE, vae_loss
    from src.data.partitions import get_fl_partitioned_data

    root_dir = os.getcwd()
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    print(f"--- Generating Stage II Geometry Data (GPU) --- (CWD: {root_dir})")
    print(f"Directory listing of CWD: {os.listdir(root_dir)}")
    data_dir = os.path.join(root_dir, 'data')
    if os.path.exists(data_dir):
        print(f"Data dir exists at {data_dir}. Contents: {os.listdir(data_dir)}")
    else:
        print(f"Data dir DOES NOT EXIST at {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created data dir at {data_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    try:
        X_ref, _, clients = get_fl_partitioned_data('pathmnist', n_clients=1, scenario='iid', num_classes=num_classes)
        X_batch, y_batch = clients[0]
        x = X_batch[:1] 
        y = y_batch[:1]
    except Exception as e:
        print(f"Error loading data for Geometry: {e}")
        return []
    
    model = VAE(num_channels=in_channels, latent_dim=latent_dim, num_classes=num_classes).to(device)
    model.eval() # Ensure deterministic behavior (disable dropout)
    
    model.zero_grad()
    x = x.to(device)
    y = y.to(device)
    recon_x, mu, logvar = model(x, y)
    loss_val, _, _ = vae_loss(recon_x, x, mu, logvar)
    loss_val.backward()
    
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    g = torch.cat(grads)
    norm_g = torch.norm(g)
    if norm_g < 1e-9:
        print("Warning: Gradient is zero, skipping geometry plot.")
        return []

    g_unit = g / norm_g
    
    v_rand = torch.randn_like(g)
    v_orth = v_rand - torch.dot(v_rand, g_unit) * g_unit
    v_orth = v_orth / torch.norm(v_orth)
    
    thetas = np.linspace(0, np.pi, n_points)
    alignments = np.cos(thetas)
    
    # Adaptive magnitudes: choose ||u|| so that the linear predictor
    # z = ||g|| * ||u|| * cos(theta) spans the sigmoid transition region.
    scale = 1.0 / (norm_g.item() + 1e-9)
    target_logits = [0.1, 0.5, 1.0, 2.0, 4.0]
    magnitudes = sorted([t * scale for t in target_logits])
    
    geometry_data = []
    
    params = []
    for p in model.parameters():
        params.append(p.view(-1))
    W_base_vec = torch.cat(params)
    
    def get_loss_at(w_vec, seed):
        # Enforce same VAE noise
        rng_devices = [device] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed)
            
            offset = 0
            for p in model.parameters():
                num = p.numel()
                p.data.copy_(w_vec[offset:offset+num].view_as(p))
                offset += num
            with torch.no_grad():
                recon_x, mu, logvar = model(x, y)
                loss, _, _ = vae_loss(recon_x, x, mu, logvar)
        return loss.item()
        
    # Fixed seed for geometry plot
    fixed_seed = 42
    base_loss = get_loss_at(W_base_vec, fixed_seed)
    
    for mag in magnitudes:
        for theta, align in zip(thetas, alignments):
            u_dir = np.cos(theta) * g_unit + np.sin(theta) * v_orth
            u = u_dir * mag
            
            loss_pert = get_loss_at(W_base_vec + u, fixed_seed)
            delta = loss_pert - base_loss
            gain = -delta
            
            if gain >= 0:
                phi = 1.0 / (1.0 + np.exp(-gain)) - 0.5
            else:
                phi = np.exp(gain) / (1.0 + np.exp(gain)) - 0.5
                
                phi = np.exp(gain) / (1.0 + np.exp(gain)) - 0.5
            
            linear_pred = -mag * norm_g.item() * align
            geometry_data.append({
                'magnitude': float(mag),
                'alignment': float(align),
                'theta': float(theta),
                'J': float(phi),
                'linear_pred': float(linear_pred),
                'error': float(abs(phi - (1.0/(1.0+np.exp(-linear_pred)) - 0.5))) # Error vs sigmoid(linear_pred)
            })
            
    return geometry_data

# @ray.remote(num_gpus=1)
def simulate_trajectory(in_channels=3, latent_dim=32, num_classes=10, limit=2000, run_seed=None, tau_noise=0.1):
    import torch
    import torch.optim as optim
    import numpy as np
    import random
    import time
    import sys, os
    from src.models.vae import VAE, vae_loss
    from src.data.partitions import get_fl_partitioned_data

    root_dir = os.getcwd()
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    print(f"--- Simulating Trajectory (VAE + Local Epochs) (GPU) --- (CWD: {root_dir})")
    print(f"Directory listing of CWD: {os.listdir(root_dir)}")
    data_dir = os.path.join(root_dir, 'data')
    if os.path.exists(data_dir):
        print(f"Data dir exists at {data_dir}. Contents: {os.listdir(data_dir)}")
    else:
        print(f"Data dir DOES NOT EXIST at {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created data dir at {data_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Trajectory Sim running on {device}")
    
    scenarios = ['iid', 'non-iid', 'inverted']
    rounds = 20
    n_clients = 5
    lr = 1e-3
    local_epochs = 3 
    n_seeds = 5 # Run multiple seeds for CI

    if run_seed is None:
        env_seed = os.getenv('TS_MIA_RUN_SEED') or os.getenv('TS_MIA_SEED')
        if env_seed is not None:
            run_seed = int(env_seed)
            seed_source = "env"
        else:
            run_seed = int(time.time_ns() % (2**32))
            seed_source = "time_ns"
    else:
        seed_source = "arg"

    # Allow tau_noise override via env for reproducibility in plots
    env_tau = os.getenv('TS_MIA_TAU_NOISE')
    if env_tau is not None:
        try:
            tau_noise = float(env_tau)
        except ValueError:
            pass

    print(f"Trajectory run seed: {run_seed} (source: {seed_source})")
    print(f"Trajectory tau_noise: {tau_noise}")
    seed_seq = np.random.SeedSequence(run_seed)
    scenario_seqs = seed_seq.spawn(len(scenarios))
    
    all_risks = {}
    all_surs = {}
    all_js = {}
    all_seeds = {}
    
    for scenario_idx, scenario in enumerate(scenarios):
        print(f"Running Scenario: {scenario}, E={local_epochs}, Seeds={n_seeds}")
        scenario_risks = []
        scenario_surs = []
        scenario_js = []
        scenario_seed_seq = scenario_seqs[scenario_idx]
        scenario_seeds = scenario_seed_seq.generate_state(n_seeds, dtype=np.uint32)
        
        for seed_idx, seed in enumerate(scenario_seeds):
            seed = int(seed)
            print(f"  Seed {seed_idx+1}/{n_seeds}: {seed}")
            # Set seed for reproducibility of data partition
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            X_all, num_classes_actual, clients = get_fl_partitioned_data('pathmnist', n_clients, scenario, num_classes, limit=limit)
            
            model = VAE(num_channels=in_channels, latent_dim=latent_dim, num_classes=num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            risks_trace = []
            surs_trace = []
            js_trace = []
        
            for t in range(rounds):
                updates = []
                global_state = {k: v.clone() for k, v in model.state_dict().items()}
                
                for X_c, y_c in clients:
                    model.load_state_dict(global_state)
                    start_params = [p.clone() for p in model.parameters()]
                    
                    for _ in range(local_epochs):
                        model.zero_grad()
                        X_c_dev = X_c.to(device)
                        y_c_dev = y_c.to(device)
                        recon, mu, logvar = model(X_c_dev, y_c_dev)
                        loss, _, _ = vae_loss(recon, X_c_dev, mu, logvar)
                        loss.backward()
                        optimizer.step()
                        
                    end_params = [p for p in model.parameters()]
                    u_k_vec = torch.cat([(ep - sp).view(-1) for sp, ep in zip(start_params, end_params)])
                    updates.append(u_k_vec.detach())
                
                Delta_W = torch.mean(torch.stack(updates), dim=0)
                
                round_scores = []
                round_surs = []
                round_js = []
                
                for k in range(n_clients):
                    u_k = updates[k] 
                    u_target_in_avg = u_k / n_clients
                    B = Delta_W - u_target_in_avg
                    
                    sur = 0.0
                    norm_dw = torch.norm(Delta_W)
                    if norm_dw > 1e-9:
                        sur = (torch.norm(u_target_in_avg) / norm_dw).item()
                        
                    def get_scores_at(state_dict, X, y):
                        model.load_state_dict(state_dict)
                        with torch.no_grad():
                            X_dev = X.to(device)
                            y_dev = y.to(device)
                            recon, mu, logvar = model(X_dev, y_dev)
                            loss_vec, _, _ = vae_loss(recon, X_dev, mu, logvar, reduction='none')
                        return -loss_vec.cpu().numpy()

                    W_base_vec = {}
                    offset = 0
                    for name, p in model.named_parameters():
                        num = p.numel()
                        b_part = B[offset:offset+num].view_as(p)
                        W_base_vec[name] = global_state[name] + b_part
                        offset += num
                        
                    W_full_vec = {}
                    offset = 0
                    for name, p in model.named_parameters():
                        num = p.numel()
                        dw_part = Delta_W[offset:offset+num].view_as(p)
                        W_full_vec[name] = global_state[name] + dw_part
                        offset += num
                        
                    X_target, y_target = clients[k]
                    scores_base = get_scores_at(W_base_vec, X_target, y_target)
                    scores_full = get_scores_at(W_full_vec, X_target, y_target)
                    
                    diffs = scores_full - scores_base 
                    def phi_np(z):
                        out = np.zeros_like(z)
                        pos = z >= 0
                        neg = ~pos
                        if np.any(pos):
                            out[pos] = 1.0 / (1.0 + np.exp(-z[pos])) - 0.5
                        if np.any(neg):
                            out[neg] = np.exp(z[neg]) / (1.0 + np.exp(z[neg])) - 0.5
                        return out
                    
                    phis = phi_np(diffs)
                    J = np.mean(phis)
                    
                    detected = 1.0 if (sur > tau_noise and J > 0) else 0.0
                    round_scores.append(detected)
                    round_surs.append(float(sur))
                    round_js.append(float(J))
                
                risks_trace.append(np.mean(round_scores))
                surs_trace.append(float(np.mean(round_surs)))
                js_trace.append(float(np.mean(round_js)))
                
                # Apply Global Update
                offset = 0
                new_state = {}
                for name, p in model.named_parameters():
                    num = p.numel()
                    update_vec = Delta_W[offset:offset+num].view_as(p)
                    new_state[name] = global_state[name] + update_vec
                    offset += num
                model.load_state_dict(new_state, strict=False)
            
            scenario_risks.append(risks_trace) # list of per-seed traces
            scenario_surs.append(surs_trace)
            scenario_js.append(js_trace)
        
        all_risks[scenario] = scenario_risks # list of [seed1_trace, seed2_trace, ...]
        all_surs[scenario] = scenario_surs
        all_js[scenario] = scenario_js
        all_seeds[scenario] = [int(s) for s in scenario_seeds]
    
    # Convert numpy arrays/scalars to pure python types for safe serialization
    all_risks_py = {k: [[float(x) for x in trace] for trace in v] for k, v in all_risks.items()}
    all_surs_py = {k: [[float(x) for x in trace] for trace in v] for k, v in all_surs.items()}
    all_js_py = {k: [[float(x) for x in trace] for trace in v] for k, v in all_js.items()}
    
    # Calculate average risk for reporting
    avg_risks = {k: float(np.mean(v)) for k, v in all_risks.items()}
    print(f"Average Risks (across all seeds/rounds): {avg_risks}")
    
    return {
        'scenarios': list(all_risks_py.keys()),
        'rounds': list(range(rounds)),
        'risks': all_risks_py,
        'surs': all_surs_py,
        'Js': all_js_py,
        'avg_risks': avg_risks,
        'run_seed': int(run_seed),
        'scenario_seeds': all_seeds,
        'tau_noise': float(tau_noise),
    }
