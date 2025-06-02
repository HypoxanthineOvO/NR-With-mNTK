import numpy as np
import torch
import matplotlib.pyplot as plt
import os, sys, shutil
from NTK import GetEigenValuesData

def Read_PSNR(path: str) -> torch.Tensor:
    with open(path, 'r') as file:
        lines = file.readlines()
    psnr_values = []
    # Format: Step x, PSNR = y
    for line in lines:
        if "PSNR" in line:
            parts = line.split(',')
            psnr_value = float(parts[-1].split('=')[-1].strip())
            psnr_values.append(psnr_value)
    psnr_tensor = torch.tensor(psnr_values, dtype=torch.float32)
    return psnr_tensor

def Load_NTK(path: str) -> torch.Tensor:
    ntks = torch.load(path, map_location=torch.device('cpu'))

    assert isinstance(ntks, list), "NTK data should be a list of dictionaries."

    results = {
        "hash_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "rgb_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        }
    }

    for i, ntk in enumerate(ntks):
        #print(f"NTK {i+1}:")
        key: str
        value: torch.Tensor
        for key, value in ntk[0].items():
            assert isinstance(value, torch.Tensor), f"Value for key '{key}' should be a torch.Tensor."
            #print(f"  {key}:", end = " ")
            eigvals = GetEigenValuesData(value)
            #print(f"Max Eigenvalue: {eigvals['max']:.4e}, Condition Number: {eigvals['cond_num']:.4e}")
            if key == "hash_network":
                results["hash_network"]["Max Eigenvalue"].append(eigvals['max'])
                results["hash_network"]["Condition Number"].append(eigvals['cond_num'])
            elif key == "rgb_network":
                results["rgb_network"]["Max Eigenvalue"].append(eigvals['max'])
                results["rgb_network"]["Condition Number"].append(eigvals['cond_num'])
    
    results["hash_network"]["Max Eigenvalue"] = np.array(results["hash_network"]["Max Eigenvalue"]).mean()
    results["hash_network"]["Condition Number"] = np.array(results["hash_network"]["Condition Number"]).mean()
    results["rgb_network"]["Max Eigenvalue"] = np.array(results["rgb_network"]["Max Eigenvalue"]).mean()
    results["rgb_network"]["Condition Number"] = np.array(results["rgb_network"]["Condition Number"]).mean()
    print(f"Hash Network - Max Eigenvalue: {results['hash_network']['Max Eigenvalue']:.4e}, "
          f"Condition Number: {results['hash_network']['Condition Number']:.4e}")
    print(f"RGB Network - Max Eigenvalue: {results['rgb_network']['Max Eigenvalue']:.4e}, "
          f"Condition Number: {results['rgb_network']['Condition Number']:.4e}")
    return results

if __name__ == "__main__":
    result_to_plot = {
        "hash_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "rgb_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        }
    }
    psnr_val = Read_PSNR("./PSNR_RECS")
    max_steps = 25000
    psnr_to_plot = psnr_val[:max_steps // 1000].numpy().tolist()
    for step in range(0, max_steps, 1000):
        print(f"Loading NTK data for step {step}...")
        result = Load_NTK(f"./snapshots/lego_ntk_{step}.pt")
        result_to_plot["hash_network"]["Max Eigenvalue"].append(result["hash_network"]["Max Eigenvalue"])
        result_to_plot["hash_network"]["Condition Number"].append(result["hash_network"]["Condition Number"])
        result_to_plot["rgb_network"]["Max Eigenvalue"].append(result["rgb_network"]["Max Eigenvalue"])
        result_to_plot["rgb_network"]["Condition Number"].append(result["rgb_network"]["Condition Number"])

    plt.figure(figsize=(8, 6))
    hash_max_eigenvalues = result_to_plot["hash_network"]["Max Eigenvalue"]
    rgb_max_eigenvalues = result_to_plot["rgb_network"]["Max Eigenvalue"]

    max_val = max(max(hash_max_eigenvalues), max(rgb_max_eigenvalues))
    psnr_scaled = [val / 40 * max_val for val in psnr_to_plot]
    plt.plot(
        range(0, max_steps, 1000), 
        psnr_scaled, 
        label='PSNR (scaled)', marker='o'
    )

    plt.plot(
        range(0, max_steps, 1000), 
        result_to_plot["hash_network"]["Max Eigenvalue"], 
        label='Hash Network Max Eigenvalue', marker='o'
    )
    plt.plot(
        range(0, max_steps, 1000), 
        result_to_plot["rgb_network"]["Max Eigenvalue"],
        label="RGB Network Max Eigenvalue", marker='o'
    )
    plt.xlabel('Training Steps')
    plt.ylabel('Max Eigenvalue')
    plt.title('Max Eigenvalue of Networks Over Training Steps')
    plt.legend()
    plt.grid()
    plt.show()