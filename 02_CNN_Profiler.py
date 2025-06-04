import os, sys, tqdm
import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt
import warnings
from deepspeed.profiling.flops_profiler import FlopsProfiler
warnings.filterwarnings("ignore", category=UserWarning, message=".*libpng warning: iCCP: known incorrect sRGB profile.*")

from utils import load_datasets, display_sample_images, dataset_to_tensor, sample_in_datasets

from Models import CNN
from NTK import GetEigenValuesData

if __name__ == "__main__":
    num_epochs = 50
    SAMPLE_TO_EVAL_NTK = 10  # Number of samples to evaluate NTK
    learning_rate = 0.001
    LOAD_PRETRAINED = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    max_eigval_records = []
    kappa_records = []
    first_20_eigval = []


    train_dataset, test_dataset, train_loader, test_loader = load_datasets(6000, 6000)
    print(f"Dataset Size: {len(train_dataset)} training samples, {len(test_dataset)} test samples")

    train_dataset_tensor = dataset_to_tensor(train_dataset)

    #display_sample_images(test_loader, num_images=5, plt_height=2)
    train_tensor = dataset_to_tensor(train_dataset)
    test_tensor = dataset_to_tensor(test_dataset)

    data_to_eval_ntk = sample_in_datasets(
        train_dataset, num_samples=SAMPLE_TO_EVAL_NTK
    )  # Sample a few images to evaluate NTK

    model: torch.nn.modules = CNN()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    model = model.to(device)
    prof = FlopsProfiler(model)


    total_step = len(train_loader)
    for epoch in tqdm.tqdm(range(num_epochs), desc='Epochs', position=0):
        model.train()
        with tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', position=1, leave=False) as t:
            for i, (images, labels) in enumerate(t):
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                t.set_postfix(loss=f'{loss.item():.4f}')
        if epoch % 5 == 0:
            prof.start_profile()
            images, labels = next(iter(train_loader))
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.to(device))
            prof.stop_profile()

            flops = prof.get_total_flops()
            macs = prof.get_total_macs()
            params = prof.get_total_params()
            prof.print_model_profile(profile_step = epoch)
        # Evaluate NTK and condition number
        ntks_max_eigval = {}
        ntks_kappa = {}
        first_20_eigval = {}
        module_names = model.get_module_eval_ntk_name()
        # Set keys for NTK dictionary
        for module_name in module_names:
            ntks_max_eigval[module_name] = []
            ntks_kappa[module_name] = []
            first_20_eigval[module_name] = []
        with torch.no_grad():
            images = data_to_eval_ntk.to(device)
            model.eval()
            ntk_on_one_sample = []
            for id in tqdm.trange(SAMPLE_TO_EVAL_NTK):
                image = images[id].unsqueeze(0)
                _, ntk_each_modules = model.forward_with_evaluate_ntk(image)
                for k, v in ntk_each_modules.items():
                    eig_val_metrics = GetEigenValuesData(v)
                    ntks_max_eigval[k].append(eig_val_metrics["max"])
                    ntks_kappa[k].append(eig_val_metrics["cond_num"])
                    first_20_eigval[k] = eig_val_metrics["max_20"]
                
        ntk_results = {k: np.mean(v) for k, v in ntks_max_eigval.items()}
        kappa_results = {k: np.mean(v) for k, v in ntks_kappa.items()}

        first_20_eigval_results = first_20_eigval
        max_eigval_records.append(ntk_results)
        kappa_records.append(kappa_results)
        
        
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            
        tqdm.tqdm.write(f"=============== Epoch {epoch+1}/{num_epochs} ===============")
        tqdm.tqdm.write(f"Accuracy: {100 * correct / total:.2f}%")
        tqdm.tqdm.write("NTK Eigen Values' Metrics:")

        for k, v1, v2 in zip(ntk_results.keys(), ntk_results.values(), kappa_results.values()):
            tqdm.tqdm.write(f"\tModule {k}: Max Eig Val: {v1:.4f}, Condition Number (Kappa): {v2:.4f}")
        # Draw the plot of NTK max eigen values
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
        #if epoch >= 0:
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            for module_name in module_names:
                values = [record[module_name] for record in max_eigval_records]
                x_ticks = np.arange(0, epoch + 1, 1, dtype = np.int32)
                plt.plot(x_ticks, values, linestyle='-', marker='o', label=module_name)
            plt.xlabel('Epoch')
            plt.xticks(np.arange(0, num_epochs + 1, 1, dtype = np.int32))
            plt.ylabel('NTK Max Eigen Value')
            plt.title('NTK Max Eigen Values Over Epochs')
            plt.legend()
            plt.grid()
            plt.subplot(1, 2, 2)
            for module_name in module_names:
                values = [record[module_name] for record in kappa_records]
                x_ticks = np.arange(0, epoch + 1, 1, dtype = np.int32)
                plt.plot(x_ticks, values, linestyle='-', marker='o', label=module_name)
            plt.xlabel('Epoch')
            plt.xticks(np.arange(0, num_epochs + 1, 1, dtype = np.int32))
            plt.ylabel('Condition Number (Kappa)')
            plt.title('Condition Number Over Epochs')
            plt.legend()
            plt.grid()

            plt.savefig(f"NTK_Eigval_Metrics_{epoch+1}.png")



    print("Finished Training")

