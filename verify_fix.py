
import json
import numpy as np

with open('results/theory_simulation.json', 'r') as f:
    data = json.load(f)
    
if 'trajectory' in data and 'risks' in data['trajectory']:
    risks = data['trajectory']['risks']
    print("--- Risk Analysis ---")
    for scenario, values in risks.items():
        avg = np.mean(values)
        print(f"Scenario: {scenario:<10} | Avg Risk: {avg:.4f}")
        
    iid_risk = np.mean(risks.get('iid', [0]))
    non_iid_risk = np.mean(risks.get('non-iid', [0]))
    
    if non_iid_risk < iid_risk:
        print(f"\nSUCCESS: Non-IID Risk ({non_iid_risk:.4f}) < IID Risk ({iid_risk:.4f})")
    else:
        print(f"\nWARNING: Non-IID Risk ({non_iid_risk:.4f}) >= IID Risk ({iid_risk:.4f})")
else:
    print("Trajectory data not found in results.")
