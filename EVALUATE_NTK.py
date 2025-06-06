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

def parse_NTK(ntks: list) -> dict:

    assert isinstance(ntks, list), "NTK data should be a list of dictionaries."

    results = {
        "hash_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "rgb_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "hash_encoding": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "hash_grids": [
            [] for _ in range(16)  # Assuming 16 hash grids
        ],
        "psnr": -1
    }
    for i, ntk in enumerate(ntks):
        #print(f"NTK {i+1}:")
        key: str
        value: torch.Tensor
        for key, value in ntk[0].items():
            if key.startswith("hash_group_"):
                group_i = key.split("_")[-1]
                assert group_i.isdigit(), f"Key '{key}' does not end with a valid group index."
                group_id = int(group_i)
                eigval = GetEigenValuesData(value)
                results["hash_grids"][group_id].append(
                    eigval['max']
                )
                continue
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
            elif key == "hash_encoding":
                results["hash_encoding"]["Max Eigenvalue"].append(eigvals['max'])
                results["hash_encoding"]["Condition Number"].append(eigvals['cond_num'])
    
    results["hash_network"]["Max Eigenvalue"] = np.array(results["hash_network"]["Max Eigenvalue"]).mean()
    results["hash_network"]["Condition Number"] = np.array(results["hash_network"]["Condition Number"]).mean()
    results["rgb_network"]["Max Eigenvalue"] = np.array(results["rgb_network"]["Max Eigenvalue"]).mean()
    results["rgb_network"]["Condition Number"] = np.array(results["rgb_network"]["Condition Number"]).mean()
    results["hash_encoding"]["Max Eigenvalue"] = np.array(results["hash_encoding"]["Max Eigenvalue"]).mean()
    results["hash_encoding"]["Condition Number"] = np.array(results["hash_encoding"]["Condition Number"]).mean()
    #results["psnr"] = results["psnr"].item() if isinstance(results["psnr"], torch.Tensor) else results["psnr"]
    
    # for group_id, eigvals in enumerate(results["hash_grids"]):
    #     results["hash_grids"][group_id] = np.array(eigvals).mean()
    #     print(f"Hash Grid {group_id} - Max Eigenvalue: {results['hash_grids'][group_id]:.4e}")
    # print(f"PSNR: {results['psnr']:.4f}")
    return results

def Load_NTK(path: str) -> torch.Tensor:
    ntks = torch.load(path, map_location=torch.device('cpu'))

    results = parse_NTK(ntks)
    # print(f"Hash Network - Max Eigenvalue: {results['hash_network']['Max Eigenvalue']:.4e}, "
    #       f"Condition Number: {results['hash_network']['Condition Number']:.4e}")
    # print(f"RGB Network - Max Eigenvalue: {results['rgb_network']['Max Eigenvalue']:.4e}, "
    #       f"Condition Number: {results['rgb_network']['Condition Number']:.4e}")
    # print(f"Hash Encoding - Max Eigenvalue: {results['hash_encoding']['Max Eigenvalue']:.4e}, "
    #         f"Condition Number: {results['hash_encoding']['Condition Number']:.4e}")
    return results

if __name__ == "__main__":
    base_path: str = "./snapshots/Modify-OptScheduler-Baseline"
    result_to_plot = {
        "hash_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "rgb_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "hash_encoding": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "hash_grids": [[] for _ in range(16)]  # Assuming 16 hash grids
    }
    max_steps = 1000 * 3
    for step in range(0, max_steps, 1000):
        print(f"Loading NTK data for step {step}...")
        print("=" * 50)
        result = Load_NTK(os.path.join(base_path, f"lego_ntk_{step}.pt"))
        result_to_plot["hash_network"]["Max Eigenvalue"].append(result["hash_network"]["Max Eigenvalue"])
        result_to_plot["hash_network"]["Condition Number"].append(result["hash_network"]["Condition Number"])
        result_to_plot["rgb_network"]["Max Eigenvalue"].append(result["rgb_network"]["Max Eigenvalue"])
        result_to_plot["rgb_network"]["Condition Number"].append(result["rgb_network"]["Condition Number"])
        result_to_plot["hash_encoding"]["Max Eigenvalue"].append(result["hash_encoding"]["Max Eigenvalue"])
        result_to_plot["hash_encoding"]["Condition Number"].append(result["hash_encoding"]["Condition Number"])
        for group_id in range(16):
            result_to_plot["hash_grids"][group_id].append(result["hash_grids"][group_id])


    hash_max_eigenvalues = result_to_plot["hash_network"]["Max Eigenvalue"]
    rgb_max_eigenvalues = result_to_plot["rgb_network"]["Max Eigenvalue"]

    max_val = max(max(hash_max_eigenvalues), max(rgb_max_eigenvalues))

    plt.figure(figsize=(18, 6), dpi = 150)
    # Font: Arial
    plt.rcParams['font.family'] = 'serif'
    plt.subplot(1, 2, 1)
    if (os.path.exists("./PSNR_RECS")):
        psnr_val = Read_PSNR("./PSNR_RECS")
        psnr_to_plot = psnr_val[:max_steps // 1000].numpy().tolist()
        psnr_scaled = [val / 40 * max_val for val in psnr_to_plot]
        plt.plot(
            range(0, max_steps, 1000), 
            psnr_scaled, 
            label='PSNR (scaled)', marker='o'
        )

    plt.plot(
        range(0, max_steps, 1000), 
        result_to_plot["hash_network"]["Max Eigenvalue"], 
        color = "#EECA40",
        label='Hash Network Max Eigenvalue', marker='o'
    )
    plt.plot(
        range(0, max_steps, 1000), 
        result_to_plot["rgb_network"]["Max Eigenvalue"],
        color = "#FD763F",
        label="RGB Network Max Eigenvalue", marker='o'
    )
    plt.plot(
        range(0, max_steps, 1000), 
        result_to_plot["hash_encoding"]["Max Eigenvalue"],
        color = "#23BAC5",
        label="Hash Encoding Max Eigenvalue", marker='o'
    )
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Max Eigenvalue', fontsize=14)
    # Logarithmic scale for better visibility
    plt.yscale('log')
    plt.title('Max Eigenvalue of Networks Over Training Steps', fontsize=18)
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)

    # Use a qualitative colormap with good distinction (e.g., 'tab20' for 16 groups)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, 16)]

    for group_id in range(16):
        plt.plot(
            range(0, max_steps, 1000), 
            result_to_plot["hash_grids"][group_id], 
            color=colors[group_id],
            linestyle='-',
            linewidth=1.5,
            marker='o',
            markersize=4,
            label=f'Grid {group_id}'
        )

    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Max Eigenvalue', fontsize=14)
    plt.title('Max Eigenvalue of Hash Grids Over Training Steps', 
            fontsize=18)

    # Improve legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles, labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        framealpha=1,
        title='Grid ID',
        title_fontsize=10
    )

    # Grid and layout improvements
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    #plt.tight_layout()

    plt.savefig('ngp_eigenvalue_plot.png')