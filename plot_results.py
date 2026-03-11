import json
import matplotlib.pyplot as plt
import numpy as np
import os
# from scipy.interpolate import griddata # Removed unused dependency
from matplotlib.colors import LogNorm

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

SAVE_DPI = 600

def _bootstrap_ci(y_matrix, n_boot=1000, alpha=0.05, seed=0):
    """Return (low, high) CI bands for mean across seeds via bootstrap."""
    n, t = y_matrix.shape
    if n <= 1:
        mean = y_matrix.mean(axis=0)
        return mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = y_matrix[idx].mean(axis=1)
    lo = np.percentile(boot_means, 100 * (alpha / 2), axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return lo, hi

def _load_non_iid_sample(
    dataset_name='pathmnist',
    limit=2000,
    n_clients=5,
    client_idx=0,
    sample_idx=0,
):
    """Load a deterministic example image from the Non-IID partition."""
    try:
        import torch
        from src.data.partitions import get_fl_partitioned_data
    except Exception as e:
        print(f"Skipping Non-IID sample image (dependency error): {e}")
        return None

    try:
        _, _, clients = get_fl_partitioned_data(
            dataset_name=dataset_name,
            n_clients=n_clients,
            scenario='non-iid',
            num_classes=10,
            limit=limit,
        )
    except Exception as e:
        print(f"Skipping Non-IID sample image (data load error): {e}")
        return None

    if not clients:
        print("Skipping Non-IID sample image (no clients returned).")
        return None

    if client_idx < 0 or client_idx >= len(clients):
        client_idx = 0

    X_c, y_c = clients[client_idx]
    if len(X_c) == 0:
        print("Skipping Non-IID sample image (empty client dataset).")
        return None

    if sample_idx < 0 or sample_idx >= len(X_c):
        sample_idx = 0

    img = X_c[sample_idx].detach().cpu().numpy()
    label = y_c[sample_idx]
    try:
        label = int(label.item())
    except Exception:
        label = int(label)

    return img, label

def load_results(path='results/theory_simulation.json'):
    with open(path, 'r') as f:
        return json.load(f)

def plot_sur_geometry(data, output_path='figs/verify_sur_geometry.png', cancel_eps=0.3):
    print("Plotting SUR Geometry...")
    # data is list of dicts: {'lambda': ..., 'r': ..., 'sur_sq': ...}
    
    lambdas = np.array([d['lambda'] for d in data])
    rs = np.array([d['r'] for d in data])
    surs = np.sqrt(np.array([d['sur_sq'] for d in data]))

    # Restrict r-range for clearer visualization
    r_max = 6.0
    keep = rs <= r_max
    lambdas = lambdas[keep]
    rs = rs[keep]
    surs = surs[keep]
    
    plt.figure(figsize=(10, 6))
    
    # Try to detect if it's a grid
    unique_l = np.unique(lambdas.round(6)) # Rounding to merge floats
    unique_r = np.unique(rs.round(6))
    
    if len(lambdas) > 100 and len(unique_l) * len(unique_r) >= len(lambdas) * 0.95:
        # It's a grid (or close to it)
        from scipy.interpolate import griddata
        
        # Use valid unique values sorted
        unique_l.sort()
        unique_r.sort()
        
        # If we have a massive number of unique values (e.g. random sampling), fallback to linspace
        if len(unique_l) > 200 or len(unique_r) > 200:
             li = np.linspace(min(lambdas), max(lambdas), 100)
             ri = np.linspace(min(rs), max(rs), 100)
        else:
             # Use the exact grid
             li = unique_l
             ri = unique_r
             
        LL, RR = np.meshgrid(li, ri)
        
        # Interpolate using 'nearest' to avoid NaNs at edges (since points match exactly or are very close)
        # 'linear' can fail due to floating point epsilon being "outside" hull
        SUR_grid = griddata((lambdas, rs), surs, (LL, RR), method='nearest')
        
        # Handle NaNs (should be none with nearest, but safe to keep)
        # Also ensure strict positivity for LogNorm
        min_val = np.nanmin(SUR_grid) if not np.isnan(SUR_grid).all() else 1e-4
        min_val = max(min_val, 1e-4)
        SUR_grid = np.nan_to_num(SUR_grid, nan=min_val)
        SUR_grid = np.maximum(SUR_grid, min_val)
        
        # Use pcolormesh
        # Note: pcolormesh expects corners or centers. If we give X, Y, Z of same shape, it infers centers.
        plt.pcolormesh(LL, RR, SUR_grid, cmap='viridis_r', shading='auto',
                       norm=LogNorm(vmin=SUR_grid.min(), vmax=SUR_grid.max()))
        plt.colorbar(label='SUR (Signal-to-Update Ratio) [Log Scale]')

        # Overlay cancellation band where ||Delta W|| is near zero.
        # For unit ||u||, ||Delta W|| = sqrt(1 + r^2 + 2 r lambda).
        dW_norm = np.sqrt(1.0 + RR**2 + 2.0 * RR * LL)
        cancel_mask = (dW_norm < cancel_eps).astype(float)
        try:
            # Color cancellation pixels in red (no contour lines)
            plt.contourf(LL, RR, cancel_mask, levels=[0.5, 1.5], colors=['#ff4d4d'], alpha=1.)
        except Exception:
            pass
        
        # Add contour lines
        try:
            cs = plt.contour(LL, RR, SUR_grid, levels=[0.1, 0.5, 1.0, 2.0, 5.0], colors='white', linewidths=0.8)
            plt.clabel(cs, inline=1, fontsize=10)
        except:
            pass # Contour might fail if grid is weird
        
    else:
        # Fallback to scatter
        sc = plt.scatter(lambdas, rs, c=surs, cmap='viridis_r', s=20, alpha=0.8)
        plt.colorbar(sc, label='SUR (Signal-to-Update Ratio)')
        # Mark near-cancellation points
        dW_norm = np.sqrt(1.0 + rs**2 + 2.0 * rs * lambdas)
        cancel_idx = dW_norm < cancel_eps
        if np.any(cancel_idx):
            plt.scatter(lambdas[cancel_idx], rs[cancel_idx], facecolors='none',
                        edgecolors='red', s=60, linewidths=1.2, label=r'$\|\Delta W\|<\epsilon$')

    plt.xlabel(r'Alignment $\lambda$ (Target vs Background)')
    plt.ylabel(r'Magnitude Ratio $r = \|B\| / \|u\|$')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=SAVE_DPI)
    plt.close()

def plot_attribution_bound(data, output_path='figs/verify_attribution_bound.png', min_norm_u=1e-4):
    print("Plotting Attribution Bounds...")
    # New format: data = {'points': [...], 'trial_summaries': [...], 'summary': {...}}
    
    points = data['points']
    summary = data['summary']

    # Filter out small-norm points to avoid numerical noise floor artifacts
    filtered_points = [p for p in points if p['norm_u'] >= min_norm_u]
    if not filtered_points:
        print(f"Warning: All points filtered out by min_norm_u={min_norm_u:.2e}. "
              "Falling back to unfiltered points.")
        filtered_points = points
    
    test_pts = [p for p in filtered_points if p['split'] == 'test']
    cal_pts = [p for p in filtered_points if p['split'] == 'calibration']
    
    plt.figure(figsize=(10, 6))
    
    # Plot calibration points (used to estimate L)
    if cal_pts:
        cal_norms = np.array([d['norm_u'] for d in cal_pts])
        cal_errors = np.array([d['error_quad'] for d in cal_pts])
        cal_bounds = np.array([d['bound_quad'] for d in cal_pts])
        plt.scatter(cal_norms, cal_errors, c='gray', alpha=0.3, s=15, label='Calibration (fit L)', zorder=1)
    
    # Plot test points (held out for verification)
    test_norms = np.array([d['norm_u'] for d in test_pts])
    test_errors = np.array([d['error_quad'] for d in test_pts])
    test_bounds = np.array([d['bound_quad'] for d in test_pts])
    plt.scatter(test_norms, test_errors, c='blue', alpha=0.5, s=20, label='Test (verify bound)', zorder=2)
    
    # Plot the bound curve (use all points sorted by norm)
    all_norms = np.array([d['norm_u'] for d in filtered_points])
    # Use global median L for a smooth bound curve
    L_med = summary.get('median_L_empirical', 0.0)
    if L_med > 0:
        # Create a smooth range of norms for plotting
        x_smooth = np.logspace(np.log10(all_norms.min() + 1e-9), np.log10(all_norms.max()), 200)
        y_smooth = 0.5 * L_med * (x_smooth**2)
        plt.plot(x_smooth, y_smooth, 'r--', 
                 label=fr'Theoretical Bound ($L \approx {L_med:.2e}$)', linewidth=2, zorder=3)
    else:
        # Fallback to old jagged line if L_med missing (shouldn't happen)
        all_bounds = np.array([d['bound_quad'] for d in filtered_points])
        idx = np.argsort(all_norms)
        plt.plot(all_norms[idx], all_bounds[idx], 'r--', label='Empirical Bound', linewidth=2)
    
    plt.yscale('log')
    plt.xscale('log') # Log-log makes quadratic look linear with slope 2
    plt.xlabel(r'Update Norm $\|u\|$')
    plt.ylabel(r'Approximation Error $|\delta - \mathrm{linear}|$')
    # plt.title('Stage II: Smoothness Bound (Lemma 3.1) — Cal/Test Split') # Removed per user request
    
    # Add summary stats as text annotation
    vr = summary['overall_violation_rate']
    if 'violation_ci_95' in summary:
        ci = summary['violation_ci_95']
        ci_label = "95% CI (Wilson)"
    else:
        ci = summary.get('boot_ci_95', [0.0, 0.0])
        ci_label = "95% CI (bootstrap)"
    L_med = summary['median_L_empirical']
    stats_text = (f"Test violation rate: {vr:.1%}\n"
                  f"{ci_label}: [{ci[0]:.1%}, {ci[1]:.1%}]\n"
                  fr"Median $\hat{{L}}$: {L_med:.2e}")
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=13, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))
    
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=SAVE_DPI)
    plt.close()

def plot_trajectories(
    data,
    output_path='figs/risk_trajectory.png',
    ci_method='bootstrap',
    ci_alpha=0.05,
    ci_boot=1000,
    ci_seed=0,
):
    print("Plotting Risk Trajectories...")
    # data: {'scenarios': [...], 'rounds': [...], 'risks': {scenario: [[seed1], [seed2]...]}}
    
    scenarios = data['scenarios']
    rounds = np.array(data['rounds'])
    risks = data['risks']
    tau_noise = data.get('tau_noise', None)
    
    plt.figure(figsize=(10, 6))
    
    # Okabe-Ito colour-blind-safe palette
    colors = {'iid': '#0072B2', 'non-iid': '#E69F00', 'inverted': '#D55E00'}
    styles = {'iid': '-', 'non-iid': '--', 'inverted': '-.'}
    labels = {
        'iid': 'IID (Baseline)',
        'non-iid': 'Non-IID (Label Skew)',
        'inverted': r'Inverted (Adversarial $\lambda \approx -1$)'
    }
    
    for sc in scenarios:
        # Check if we have multiple seeds (list of lists)
        y_data = risks[sc]
        
        # If it's old format (single list), wrap it
        if isinstance(y_data[0], (float, int)):
            y_data = [y_data]
            
        y_matrix = np.array(y_data) # (n_seeds, n_rounds)
        
        mean_risk = np.mean(y_matrix, axis=0)

        n = y_matrix.shape[0]
        print(f"Scenario '{sc}': Found {n} seeds/runs.")
        ci_low = ci_high = None
        if n > 1:
            if ci_method == 'bootstrap':
                ci_low, ci_high = _bootstrap_ci(
                    y_matrix,
                    n_boot=ci_boot,
                    alpha=ci_alpha,
                    seed=ci_seed,
                )
            else:
                std_risk = np.std(y_matrix, axis=0)
                z = 1.96
                ci = z * std_risk / np.sqrt(n)
                ci_low = mean_risk - ci
                ci_high = mean_risk + ci
        
        plt.plot(rounds, mean_risk, color=colors.get(sc, 'gray'), linestyle=styles.get(sc, '-'), 
                 linewidth=2.5, label=labels.get(sc, sc))
        
        if n > 1 and ci_low is not None:
            ci_low = np.clip(ci_low, 0.0, 1.0)
            ci_high = np.clip(ci_high, 0.0, 1.0)
            plt.fill_between(rounds, ci_low, ci_high, color=colors.get(sc, 'gray'), alpha=0.25)
        
    plt.xlabel('Federated Round')
    if tau_noise is None:
        ylabel = r'TS-MIA Detection Rate ($\mathrm{SUR} > \tau_{\mathrm{noise}} \wedge J > 0$)'
    else:
        ylabel = fr'TS-MIA Detection Rate ($\mathrm{{SUR}} > {tau_noise:g} \wedge J > 0$)'
    plt.ylabel(ylabel)
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=SAVE_DPI)
    plt.close()

def plot_non_iid_image_and_risk(
    trajectory,
    output_path='figs/non_iid_image_risk.png',
    dataset_name='pathmnist',
    limit=2000,
    n_clients=5,
    client_idx=0,
    sample_idx=0,
    ci_method='bootstrap',
    ci_alpha=0.05,
    ci_boot=1000,
    ci_seed=0,
):
    print("Plotting Non-IID image + risk curve...")
    scenario = 'non-iid'
    if scenario not in trajectory.get('risks', {}):
        print("Non-IID scenario not found in trajectory data.")
        return

    rounds = np.array(trajectory['rounds'])
    def _get_metric_matrix(key):
        if key not in trajectory:
            return None
        if scenario not in trajectory[key]:
            return None
        data = trajectory[key][scenario]
        if isinstance(data[0], (float, int)):
            data = [data]
        return np.array(data)

    risk_matrix = _get_metric_matrix('risks')
    sur_matrix = _get_metric_matrix('surs')
    j_matrix = _get_metric_matrix('Js')

    if risk_matrix is None:
        print("Non-IID risks missing in trajectory data.")
        return
    if sur_matrix is None or j_matrix is None:
        print("Non-IID SUR/J missing in trajectory data. Re-run simulate_results.py to include them.")

    def _mean_and_ci(y_matrix):
        mean = np.mean(y_matrix, axis=0)
        n = y_matrix.shape[0]
        ci_low = ci_high = None
        if n > 1:
            if ci_method == 'bootstrap':
                ci_low, ci_high = _bootstrap_ci(
                    y_matrix,
                    n_boot=ci_boot,
                    alpha=ci_alpha,
                    seed=ci_seed,
                )
            else:
                std = np.std(y_matrix, axis=0)
                z = 1.96
                ci = z * std / np.sqrt(n)
                ci_low = mean - ci
                ci_high = mean + ci
        return mean, ci_low, ci_high

    mean_risk, ci_risk_low, ci_risk_high = _mean_and_ci(risk_matrix)
    mean_sur = ci_sur_low = ci_sur_high = None
    mean_j = ci_j_low = ci_j_high = None
    if sur_matrix is not None:
        mean_sur, ci_sur_low, ci_sur_high = _mean_and_ci(sur_matrix)
    if j_matrix is not None:
        mean_j, ci_j_low, ci_j_high = _mean_and_ci(j_matrix)

    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3.5])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])

    sample = _load_non_iid_sample(
        dataset_name=dataset_name,
        limit=limit,
        n_clients=n_clients,
        client_idx=client_idx,
        sample_idx=sample_idx,
    )

    if sample is None:
        ax_img.text(0.5, 0.5, 'Image unavailable', ha='center', va='center')
    else:
        img, label = sample
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img_plot = np.transpose(img, (1, 2, 0))
            if img_plot.shape[2] == 1:
                img_plot = img_plot.squeeze(-1)
        elif img.ndim == 2:
            img_plot = img
        else:
            img_plot = img
        img_plot = img_plot.astype(np.float32, copy=False)
        max_val = float(np.max(img_plot))
        min_val = float(np.min(img_plot))
        # If data was scaled twice (e.g., /255 after ToTensor), rescale for display.
        if max_val <= (1.0 / 255.0 + 1e-6):
            img_plot = img_plot * 255.0
            max_val = float(np.max(img_plot))
            min_val = float(np.min(img_plot))
        # Normalize 0-255 to 0-1 for imshow if needed.
        if max_val > 1.0:
            img_plot = img_plot / 255.0
            max_val = float(np.max(img_plot))
            min_val = float(np.min(img_plot))
        # If contrast is still tiny, stretch slightly for visibility.
        if max_val - min_val > 1e-6:
            img_plot = np.clip((img_plot - min_val) / (max_val - min_val), 0.0, 1.0)
        ax_img.imshow(img_plot, cmap='gray' if img_plot.ndim == 2 else None)

    ax_img.axis('off')

    n = risk_matrix.shape[0]
    color_risk = 'orange'
    color_sur = '#1f77b4'
    color_j = '#2ca02c'

    ax_curve.plot(rounds, mean_risk, color=color_risk, linestyle='--', linewidth=2.5, label=r'Risk ($\mathrm{SUR} > \tau \wedge J > 0$)')
    if n > 1 and ci_risk_low is not None:
        ax_curve.fill_between(rounds, ci_risk_low, ci_risk_high, color=color_risk, alpha=0.2)

    if mean_sur is not None:
        ax_curve.plot(rounds, mean_sur, color=color_sur, linestyle='-', linewidth=2.0, label=r'Mean $\mathrm{SUR}$')
        if n > 1 and ci_sur_low is not None:
            ax_curve.fill_between(rounds, ci_sur_low, ci_sur_high, color=color_sur, alpha=0.15)

    if mean_j is not None:
        ax_curve.plot(rounds, mean_j, color=color_j, linestyle='-', linewidth=2.0, label=r'Mean $J$')
        if n > 1 and ci_j_low is not None:
            ax_curve.fill_between(rounds, ci_j_low, ci_j_high, color=color_j, alpha=0.15)

    ax_curve.set_xlabel('Federated Round')
    # Removed list assignment
    ax_curve.grid(True, alpha=0.3)
    # Use shared bounds across the three curves
    y_min = float(np.min(mean_risk))
    y_max = float(np.max(mean_risk))
    if ci_risk_low is not None and ci_risk_high is not None:
        y_min = min(y_min, float(np.min(ci_risk_low)))
        y_max = max(y_max, float(np.max(ci_risk_high)))
    if mean_sur is not None:
        y_min = min(y_min, float(np.min(mean_sur)))
        y_max = max(y_max, float(np.max(mean_sur)))
        if ci_sur_low is not None and ci_sur_high is not None:
            y_min = min(y_min, float(np.min(ci_sur_low)))
            y_max = max(y_max, float(np.max(ci_sur_high)))
    if mean_j is not None:
        y_min = min(y_min, float(np.min(mean_j)))
        y_max = max(y_max, float(np.max(mean_j)))
        if ci_j_low is not None and ci_j_high is not None:
            y_min = min(y_min, float(np.min(ci_j_low)))
            y_max = max(y_max, float(np.max(ci_j_high)))
    if y_max - y_min < 1e-6:
        y_min -= 0.1
        y_max += 0.1
    pad = 0.05 * (y_max - y_min)
    ax_curve.set_ylim(y_min - pad, y_max + pad)
    
    # Place legend to the right of the plot area
    ax_curve.legend(loc='right', fontsize=13, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=SAVE_DPI)
    plt.close()

def plot_stage_two_geometry(data):
    """Plots J vs Alignment (cos theta) for a small set of magnitudes."""
    if not data:
        print("No Stage II Geometry data found.")
        return

    print("Plotting Stage II Geometry...")
    
    df_list = []
    for d in data:
        df_list.append(d)
        
    magnitudes = sorted(list(set([d['magnitude'] for d in df_list])))

    # Select a small, representative set of magnitudes (log-spaced).
    n_display = min(4, len(magnitudes))
    if n_display == 0:
        print("No magnitudes available for Stage II geometry.")
        return
    
    # We want exactly 4 elements for a 2x2 matrix if possible
    mags = np.array(magnitudes)
    mags = mags[mags > 0]
    
    if len(mags) <= 4:
        display_mags = sorted(list(mags))
    else:
        targets = np.geomspace(mags.min(), mags.max(), n_display)
        display_mags = []
        for t in targets:
            idx = int(np.argmin(np.abs(mags - t)))
            display_mags.append(float(mags[idx]))
        display_mags = sorted(list(set(display_mags)))[:n_display]

    n_display = len(display_mags)
    if n_display == 4:
        cols, rows = 2, 2
    else:
        cols = min(n_display, 4)
        rows = int(np.ceil(n_display / cols))
        
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    colors = plt.cm.viridis(np.linspace(0, 1, len(display_mags)))
    for i, mag in enumerate(display_mags):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        subset = [d for d in df_list if abs(d['magnitude'] - mag) < 1e-9]
        subset.sort(key=lambda x: x['alignment'])
        aligns = np.array([d['alignment'] for d in subset])
        Js = np.array([d['J'] for d in subset])

        ax.plot(aligns, Js, label='Sim', color=colors[i], linewidth=2)
        if subset and 'linear_pred' in subset[0]:
            preds = np.array([d['linear_pred'] for d in subset])
            ref_phi = 1.0 / (1.0 + np.exp(-preds)) - 0.5
            ax.plot(aligns, ref_phi, 'k:', alpha=0.7, label='Linear', linewidth=1.5)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(fr'$\|u\| \approx$ {mag:.1e}', fontsize=16)
        ax.grid(True, alpha=0.3)
        if row == rows - 1:
            ax.set_xlabel(r'Alignment $\cos\theta$')
        if col == 0:
            ax.set_ylabel(r'Attribution $J(u)$')
        if i == 0:
            ax.legend(fontsize='small')

    # Hide any unused subplots
    for j in range(len(display_mags), rows * cols):
        axes[j // cols, j % cols].axis('off')

    plt.tight_layout()
    plt.savefig('figs/stage_two_geometry.png', dpi=SAVE_DPI)
    plt.close()

if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    
    try:
        with open('results/theory_simulation.json', 'r') as f:
            results = json.load(f)
            
        if 'stage_one' in results and 'error' not in results['stage_one']:
            if 'data_sample' in results['stage_one']:
                 plot_sur_geometry(results['stage_one']['data_sample'])
            else:
                 plot_sur_geometry(results['stage_one'])
        elif 'stage_one' in results:
            print(f"Skipping Stage I plot (task failed: {results['stage_one']['error']})")
            
        if 'stage_two' in results and 'error' not in results['stage_two']:
            s2 = results['stage_two']
            if isinstance(s2, dict) and 'points' in s2:
                # New format with cal/test splits
                plot_attribution_bound(s2)
                sm = s2.get('summary', {})
                ci = sm.get('violation_ci_95', sm.get('boot_ci_95', '?'))
                print(f"  Stage II: violation rate={sm.get('overall_violation_rate', '?')}, "
                      f"CI={ci}, L_med={sm.get('median_L_empirical', '?')}")
            else:
                # Legacy flat list format
                plot_attribution_bound({'points': s2, 'summary': {'overall_violation_rate': 0, 'violation_ci_95': [0,0], 'median_L_empirical': 0}})
        elif 'stage_two' in results:
            print(f"Skipping Stage II bound plot (task failed: {results['stage_two']['error']})")

        if 'stage_two_geometry' in results and 'error' not in results['stage_two_geometry']:
            plot_stage_two_geometry(results['stage_two_geometry'])
        elif 'stage_two_geometry' in results:
            print(f"Skipping Stage II geometry plot (task failed: {results['stage_two_geometry']['error']})")
            
        if 'trajectory' in results and 'error' not in results['trajectory']:
            plot_trajectories(results['trajectory'])
            plot_non_iid_image_and_risk(results['trajectory'])
        elif 'trajectory' in results:
            print(f"Skipping trajectory plot (task failed: {results['trajectory']['error']})")

        print("All plots generated in figs/")
            
    except FileNotFoundError:
        print("Results file not found. Run simulate_results.py first.")
