
import json
import numpy as np

def analyze():
    print("--- Analyzing Results ---")
    try:
        with open('results/theory_simulation.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Results file not found.")
        return

    if 'stage_two' not in data:
        print("Stage two data not found.")
        return

    s2 = data['stage_two']
    if 'summary' in s2:
        sm = s2['summary']
        print(f"Summary: {sm}")
    
    if 'points' in s2:
        points = s2['points']
        # Check small scale errors
        small_scale = [p for p in points if p['norm_u'] < 1e-4]
        if small_scale:
            max_err = max(p['error_quad'] for p in small_scale)
            print(f"Max error for ||u|| < 1e-4: {max_err:.2e}")
            
            print("\n--- Detailed Small Scale Errors ---")
            print(f"{'norm_u':<12} {'error_quad':<12} {'bound_quad':<12} {'ratio':<8}")
            # Sort by norm_u
            small_scale.sort(key=lambda x: x['norm_u'])
            for p in small_scale[:20]: # Show first 20
                 print(f"{p['norm_u']:<12.2e} {p['error_quad']:<12.2e} {p['bound_quad']:<12.2e} {p['error_quad']/p['bound_quad'] if p['bound_quad']>0 else 0:<8.2f}")
        else:
            print("No small scale points found.")
            
if __name__ == "__main__":
    analyze()
