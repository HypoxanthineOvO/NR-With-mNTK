import os, sys, tqdm
import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*libpng warning: iCCP: known incorrect sRGB profile.*")

from utils import load_datasets, display_sample_images, dataset_to_tensor, sample_in_datasets

from Models import CNN

if __name__ == "__main__":
    num_epochs = 50
    SAMPLE_TO_EVAL_NTK = 10  # Number of samples to evaluate NTK
    learning_rate = 0.001
    LOAD_PRETRAINED = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ntk_records = []


    train_dataset, test_dataset, train_loader, test_loader = load_datasets(60000, 6000)
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

        # Evaluate NTK and condition number
        ntks = {}
        module_names = model.get_module_eval_ntk_name()
        # Set keys for NTK dictionary
        for module_name in module_names:
            ntks[module_name] = []
        with torch.no_grad():
            images = data_to_eval_ntk.to(device)
            model.eval()
            ntk_on_one_sample = []
            for id in range(SAMPLE_TO_EVAL_NTK):
                image = images[id].unsqueeze(0)
                _, ntk_each_modules = model.forward_with_evaluate_ntk(image)
                for k, v in ntk_each_modules.items():
                    assert k in ntks, f"Module {k} not found in NTK dictionary."
                    eig_val = torch.linalg.eigvalsh((v + v.T) / 2)  # Ensure symmetry
                    max_eig_val = eig_val.max().item()
                    ntks[k].append(max_eig_val)
                
        ntk_results = {k: np.mean(v) for k, v in ntks.items()}
        ntk_records.append(ntk_results)
        
        
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
        tqdm.tqdm.write("NTK Max Eigen Values:")
        for k, v in ntk_results.items():
            tqdm.tqdm.write(f"\tModule {k}: NTK Max Eigen Value: {v:.4f}")

        # Draw the plot of NTK max eigen values
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            plt.figure(figsize=(10, 6))
            for module_name in module_names:
                values = [record[module_name] for record in ntk_records]
                x_ticks = np.arange(0, epoch + 1, 1, dtype = np.int32)
                plt.plot(x_ticks, values, linestyle='-', marker='o', label=module_name)
            plt.xlabel('Epoch')
            plt.xticks(np.arange(0, num_epochs + 1, 1, dtype = np.int32))
            plt.ylabel('NTK Max Eigen Value')
            plt.title('NTK Max Eigen Values Over Epochs')
            plt.legend()
            plt.grid()
            plt.savefig(f"ntk_max_eigen_values_epoch_{epoch+1}.png")
            plt.close()



    print("Finished Training")

