import os
import random
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import logging
import shutil
import seaborn as sns


random.seed(55)
np.random.seed(55)
torch.manual_seed(55)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up a custom logger
logger = logging.getLogger("TrainingLogger")
logger.setLevel(logging.INFO)

# File Handler
file_handler = logging.FileHandler('training_log.log')
file_handler.setLevel(logging.INFO)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Adding handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Training started")

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("CUDA is not available. Using CPU.")

# Define Data Splitting Function
def split_data_by_individual(source_dir, train_dir, test_dir, yield_data_csv, train_ratio=0.85, test_ratio=0.15):
    yield_data = pd.read_csv(yield_data_csv)
    
    # Get a list of all individuals based on `_all_chromosomes.png` files
    all_individuals = [os.path.splitext(f)[0].replace("_all_chromosomes", "") for f in os.listdir(source_dir) if f.endswith("_all_chromosomes.png")]
    random.shuffle(all_individuals)

    # Split data into train and test
    train_split = int(train_ratio * len(all_individuals))
    train_individuals = all_individuals[:train_split]
    test_individuals = all_individuals[train_split:]

    def copy_and_record_individual(individual_list, dest_dir):
        records = []
        for individual in individual_list:
            # Generate image path
            img_file = f"{individual}_all_chromosomes.png"
            source_img_path = os.path.join(source_dir, img_file)
            dest_img_path = os.path.join(dest_dir, img_file)

            # Ensure the individual has a corresponding image and yield data
            if not os.path.exists(source_img_path):
                logger.warning(f"Image file not found for individual: {individual}")
                continue

            individual_yield_data = yield_data[yield_data['Name'] == individual]
            if individual_yield_data.empty:
                logger.warning(f"No yield data found for individual: {individual}")
                continue

            # Copy image to destination directory
            shutil.copy(source_img_path, dest_img_path)

            # Create records for the individual
            for _, row in individual_yield_data.iterrows():
                records.append({
                    'Name': individual,
                    'Image': dest_img_path,
                    'Location': row['Location'],
                    'Year': row['Year'],
                    'Reps': row['Reps'],
                    'Yield': row['Yield'],
                    'Pop': row['Pop'],
                    'SD': row['SD']
                })
        return records

    # Process and save splits
    pd.DataFrame(copy_and_record_individual(train_individuals, train_dir)).to_csv('output_vectors_Train.csv', index=False)
    pd.DataFrame(copy_and_record_individual(test_individuals, test_dir)).to_csv('output_vectors_Test.csv', index=False)
    logger.info(f"Split completed: {len(train_individuals)} train, {len(test_individuals)} test.")

# Data paths
source_folder = 'images_AF_combined/combined'  # Folder with `_all_chromosomes.png` images
train_folder = 'Train'
test_folder = 'Test'
yield_data_csv = 'WCC_Yield_no_SD.csv'  # Path to the yield data CSV file

# Ensure output directories exist
for folder in [train_folder,test_folder]:
    os.makedirs(folder, exist_ok=True)

# Split the data and create CSVs
split_data_by_individual(source_folder, train_folder,test_folder, yield_data_csv)

# Define the dataset
# class YieldDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None, img_size=(224, 224), normalization="minmax"):
#         self.data = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#         self.img_size = img_size
#         self.normalization = normalization
#         self.data['Location'] = self.data['Location'].astype('category').cat.codes
#         self.data['Year'] = pd.to_numeric(self.data['Year'], errors='coerce').fillna(0)
#         self.data['Reps'] = pd.to_numeric(self.data['Reps'], errors='coerce').fillna(0)
#         self.data['SD'] = pd.to_numeric(self.data['SD'], errors='coerce').fillna(0)
#         self.data['Pop'] = pd.to_numeric(self.data['Pop'], errors='coerce').fillna(0)
        
#         #remove rows with missing values in Yield column
#         self.data = self.data.dropna(subset=['Yield'])
#         self.data = self.data.reset_index(drop=True)
#         self.data['Yield'] = pd.to_numeric(self.data['Yield'], errors='coerce').fillna(0)
#         self.yield_min, self.yield_max = self.data['Yield'].min(), self.data['Yield'].max()


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         individual_name = self.data.loc[idx, 'Name']
#         img_file = os.path.join(self.root_dir, f"{individual_name}_all_chromosomes.png")
        
#         if not os.path.exists(img_file):
#             raise FileNotFoundError(f"Image not found: {img_file}")
        
#         # Load and transform the image
#         image = Image.open(img_file).convert('RGB').resize(self.img_size)
#         if self.transform:
#             image = self.transform(image)

#         # Metadata
#         metadata = torch.tensor([
#             self.data.loc[idx, 'Location'],
#             self.data.loc[idx, 'Year'],
#             self.data.loc[idx, 'Reps'],
#             self.data.loc[idx, 'Pop'],
#             self.data.loc[idx, 'SD']
#         ], dtype=torch.float32)

#         # Yield values
#         yield_ = (self.data.loc[idx, 'Yield'] - self.yield_min) / (self.yield_max - self.yield_min)
#         yield_value = torch.tensor([yield_], dtype=torch.float32)

#         return image, metadata, yield_value

class YieldDatasetWithMasks(Dataset):
    def __init__(self, csv_file, root_dir, mask_dir, transform=None, img_size=(224, 224), normalization="minmax"):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mask_dir = mask_dir  # Directory containing masks
        self.transform = transform
        self.img_size = img_size
        self.normalization = normalization
        self.data['Location'] = self.data['Location'].astype('category').cat.codes
        self.data['Year'] = pd.to_numeric(self.data['Year'], errors='coerce').fillna(0)
        self.data['Reps'] = pd.to_numeric(self.data['Reps'], errors='coerce').fillna(0)
        self.data['SD'] = pd.to_numeric(self.data['SD'], errors='coerce').fillna(0)
        self.data['Pop'] = pd.to_numeric(self.data['Pop'], errors='coerce').fillna(0)

        self.data = self.data.dropna(subset=['Yield'])
        self.data = self.data.reset_index(drop=True)
        self.data['Yield'] = pd.to_numeric(self.data['Yield'], errors='coerce').fillna(0)
        self.yield_min, self.yield_max = self.data['Yield'].min(), self.data['Yield'].max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        individual_name = self.data.loc[idx, 'Name']
        img_file = os.path.join(self.root_dir, f"{individual_name}_all_chromosomes.png")
        mask_file = os.path.join(self.mask_dir, f"{individual_name}_mask.png")
        
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image not found: {img_file}")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Mask not found: {mask_file}")
        
        # Load and transform the chromosome image
        image = Image.open(img_file).convert('RGB').resize(self.img_size)
        if self.transform:
            image = self.transform(image)
        
        # Load and transform the mask
        mask = Image.open(mask_file).convert('L').resize(self.img_size)  # Convert mask to grayscale
        mask = np.array(mask, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        mask = torch.tensor(mask).unsqueeze(0)  # Add channel dimension for Conv2D compatibility
        
        # Metadata
        metadata = torch.tensor([
            self.data.loc[idx, 'Location'],
            self.data.loc[idx, 'Year'],
            self.data.loc[idx, 'Reps'],
            self.data.loc[idx, 'Pop'],
            self.data.loc[idx, 'SD']
        ], dtype=torch.float32)

        # Yield values
        yield_ = (self.data.loc[idx, 'Yield'] - self.yield_min) / (self.yield_max - self.yield_min)
        yield_value = torch.tensor([yield_], dtype=torch.float32)

        return image, mask, metadata, yield_value


class LogCoshLoss(nn.Module):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(diff)))

class YieldNetWithMasks(nn.Module):
    def __init__(self, metadata_features=5):
        super(YieldNetWithMasks, self).__init__()
        # Load pretrained EfficientNet model
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Extract features from EfficientNet
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        # Add a separate convolutional branch for masks
        self.mask_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.mask_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.mask_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce spatial dimensions to 1x1

        # Fully connected layers for metadata
        self.bn_meta = nn.BatchNorm1d(metadata_features)
        self.meta_fc1 = nn.Linear(metadata_features, 64)
        self.meta_fc2 = nn.Linear(64, 32)
        self.meta_fc3 = nn.Linear(32, 16)

        # Compute the size of concatenated features dynamically
        self.fc1_input_size = num_ftrs + 32 + 16  # Combine features from image, mask, and metadata

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_size, 256)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, image, mask, metadata):
        # Process image through EfficientNet
        x_img = self.efficientnet(image)

        # Process mask through convolutional layers
        x_mask = F.relu(self.mask_conv1(mask))
        x_mask = F.relu(self.mask_conv2(x_mask))
        x_mask = self.mask_pool(x_mask).view(x_mask.size(0), -1)  # Flatten for concatenation

        # Process metadata
        metadata = self.bn_meta(metadata)
        metadata = F.relu(self.meta_fc1(metadata))
        metadata = F.relu(self.meta_fc2(metadata))
        metadata = F.relu(self.meta_fc3(metadata))

        # Concatenate features
        x = torch.cat((x_img, x_mask, metadata), dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class YieldNet(nn.Module):
    def __init__(self, metadata_features=5):
        super(YieldNet, self).__init__()
        # Load pretrained EfficientNet model
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Extract the number of features from the EfficientNet classifier
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()  # Replace the classifier with Identity to get features

        # Add a batch normalization layer for metadata
        self.bn_meta = nn.BatchNorm1d(metadata_features)
        self.meta_fc1 = nn.Linear(metadata_features, 64)
        self.meta_fc2 = nn.Linear(64, 32)
        self.meta_fc3 = nn.Linear(32, 16)  # Add another dense layer for metadata

        # Fully connected layers for combined features
        self.fc1 = nn.Linear(num_ftrs + 16, 256)  # Combine EfficientNet features and metadata
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 1)  # Output 2 values for regression

    def forward(self, image, metadata):
        # Process the image through EfficientNet
        x = self.efficientnet(image)

        # Process metadata
        metadata = self.bn_meta(metadata)
        metadata = F.relu(self.meta_fc1(metadata))
        metadata = F.relu(self.meta_fc2(metadata))
        metadata = F.relu(self.meta_fc3(metadata))

        # Concatenate image features and metadata
        x = torch.cat((x, metadata), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Saving fold metrics
metrics_file = 'cross_val_metrics.csv'
with open(metrics_file, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["Fold", "MAE", "R2"])  # Header for the CSV file

# Initialize the dataset
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    # transforms.Pad((2, 2), fill=0, padding_mode='constant'),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# train_dataset = YieldDataset(csv_file='output_vectors_Train.csv', root_dir='Train/', transform=train_transform, normalization="minmax")
# test_dataset = YieldDataset(csv_file='output_vectors_Test.csv', root_dir='Test/', transform=train_transform)



test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize dataset with masks
train_dataset = YieldDatasetWithMasks(
    csv_file='output_vectors_Train.csv',
    root_dir='Train/',
    mask_dir='images_AF_combined/masks/',
    transform=train_transform
)

test_dataset = YieldDatasetWithMasks(
    csv_file='output_vectors_Test.csv',
    root_dir='Test/',
    mask_dir='images_AF_combined/masks/',
    transform=test_transform
)


# Initialize Dataset and Dataloader
# train_dataset = YieldDataset(csv_file='output_vectors_Train.csv', root_dir='Train/', transform=train_transform, normalization="zscore")
# valid_dataset = YieldDataset(csv_file='output_vectors_Valid.csv', root_dir='Valid/', transform=test_transform)
# test_dataset = YieldDataset(csv_file='output_vectors_Test.csv', root_dir='Test/', transform=test_transform)


def train_and_validate():
    kf = KFold(n_splits=7, shuffle=True, random_state=42)
    fold_mae, fold_r2 = [], []
    best_fold = None
    best_valid_loss = float('inf')

    # Open the CSV file to append fold metrics
    with open(metrics_file, mode='a') as file:
        writer = csv.writer(file)

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            logger.info(f"Starting Fold {fold+1}")

            # Batch size and data loaders
            batch_size = 16
            trainloader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
            validloader = DataLoader(Subset(train_dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)

            # Use EnhancedYieldNet and Huber Loss for training
            net = YieldNetWithMasks().to(device)
            criterion = LogCoshLoss()
            optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

            best_loss = float("inf")
            patience = 50
            epochs_without_improvement = 0
            num_epochs = 250
            fold_train_losses, fold_valid_losses = [], []

            try:
                for epoch in range(num_epochs):
                    net.train()
                    running_train_loss = 0.0
                    train_mae, train_r2 = [], []  # Track metrics during training

                    for batch_idx, (inputs, masks, metadata, labels) in enumerate(trainloader):
                        # Move data to the device (e.g., GPU)
                        inputs, masks, metadata, labels = (
                            inputs.to(device),
                            masks.to(device),
                            metadata.to(device),
                            labels.to(device)
                        )
                        optimizer.zero_grad()

                        # Forward pass with masks
                        outputs = net(inputs, masks, metadata)
                        loss = criterion(outputs, labels) + 0.001 * torch.mean(outputs ** 2)  # Add L2 penalty
                        loss.backward()
                        optimizer.step()

                        running_train_loss += loss.item()
                        batch_mae = mean_absolute_error(labels.cpu().numpy(), outputs.cpu().detach().numpy())
                        batch_r2 = r2_score(labels.cpu().numpy(), outputs.cpu().detach().numpy())

                        train_mae.append(batch_mae)
                        train_r2.append(batch_r2)

                    # Log epoch-wise training metrics
                    train_loss = running_train_loss / len(trainloader)
                    fold_train_losses.append(train_loss)  # Append train loss
                    avg_train_mae = np.mean(train_mae)
                    avg_train_r2 = np.mean(train_r2)
                    logger.info(f"Epoch {epoch+1} - Training Loss: {train_loss:.4f}, MAE: {avg_train_mae:.4f}, R²: {avg_train_r2:.4f}")

                    # Validation phase
                    net.eval()
                    running_valid_loss = 0.0
                    valid_mae, valid_r2 = [], []
                    with torch.no_grad():
                        for inputs, masks, metadata, labels in validloader:
                            # Move data to the device
                            inputs, masks, metadata, labels = (
                                inputs.to(device),
                                masks.to(device),
                                metadata.to(device),
                                labels.to(device)
                            )

                            # Forward pass with masks
                            outputs = net(inputs, masks, metadata)

                            # Compute validation loss
                            loss = criterion(outputs, labels)
                            running_valid_loss += loss.item()

                            batch_mae = mean_absolute_error(labels.cpu().numpy(), outputs.cpu().numpy())
                            batch_r2 = r2_score(labels.cpu().numpy(), outputs.cpu().numpy())

                            valid_mae.append(batch_mae)
                            valid_r2.append(batch_r2)

                    valid_loss = running_valid_loss / len(validloader)
                    fold_valid_losses.append(valid_loss)  # Append validation loss
                    avg_valid_mae = np.mean(valid_mae)
                    avg_valid_r2 = np.mean(valid_r2)
                    logger.info(f"Epoch {epoch+1} - Validation Loss: {valid_loss:.4f}, MAE: {avg_valid_mae:.4f}, R²: {avg_valid_r2:.4f}")

                    # Pass validation loss to the scheduler
                    scheduler.step(valid_loss)

                    # Early stopping logic
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        epochs_without_improvement = 0
                        torch.save(net.state_dict(), f'best_model_fold{fold+1}.pth')
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= patience:
                        logger.info(f"Early stopping for fold {fold+1} at epoch {epoch+1}")
                        break

            except Exception as e:
                logger.error(f"Error during training in fold {fold+1}: {str(e)}")
                continue

            # Plot train and validation losses for each fold
            if fold_train_losses and fold_valid_losses:
                plt.figure()
                plt.plot(fold_train_losses, label="Training Loss")
                plt.plot(fold_valid_losses, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.title(f"Loss over Epochs for Fold {fold+1}")
                plt.savefig(f'loss_plot_fold_{fold+1}.png')
                logger.info(f"Saved loss plot for fold {fold+1} as loss_plot_fold_{fold+1}.png")
            else:
                logger.warning(f"No losses to plot for fold {fold+1}.")

            # Track the best fold
            if best_loss < best_valid_loss:
                best_valid_loss = best_loss
                best_fold = fold + 1  # Adjust for zero-based index

        if best_fold is None:
            logger.error("No successful folds. Check your data and model configurations.")
        else:
            logger.info(f"Best fold is Fold {best_fold} with validation loss {best_valid_loss:.4f}")
        return best_fold



def visualize_attention_weights(model, save_path="attention_weights.png"):
    """ Visualizes the attention weights saved during forward passes """
    if hasattr(model, 'attention_weights'):
        attention_weights = np.array(model.attention_weights).squeeze()  # Convert to numpy for easy plotting

        # Plot heatmap for the attention weights across chromosomes
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, annot=True, cmap="YlGnBu", cbar=True)
        plt.xlabel("Chromosome Index")
        plt.ylabel("Sample Index")
        plt.title("Attention Weights for Chromosomes")
        plt.savefig(save_path)
        plt.show()
        logging.info(f"Attention weights saved as {save_path}")
    else:
        logging.warning("Attention weights not found in the model.")

# Run training and validation
# best_fold = train_and_validate()

# logger.info(f"Best fold: {best_fold}")
# # Testing on the test dataset
# def test_model(model, test_loader, criterion):
#     model.eval()
#     test_loss, all_labels, all_preds = 0.0, [], []
    
#     with torch.no_grad():
#         for inputs, metadata, labels in test_loader:
#             inputs, metadata, labels = inputs.to(device), metadata.to(device), labels.to(device)
#             outputs = model(inputs, metadata)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item()
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(outputs.cpu().numpy())
    
#     test_loss /= len(test_loader)
#     test_mae = mean_absolute_error(all_labels, all_preds)
#     test_r2 = r2_score(all_labels, all_preds)

#     # Calculate accuracy
#     mean_actual = np.mean(all_labels)  # Mean of actual values
#     accuracy = 100 * (1 - (test_mae / mean_actual))

#     return test_loss, test_mae, test_r2, accuracy, all_labels, all_preds


# def run_test(best_fold):
#     net = YieldNet().to(device)
#     net.load_state_dict(torch.load(f'best_model_fold{best_fold}.pth'))

#     test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4, pin_memory=True)
#     criterion = nn.MSELoss()

#     test_loss, test_mae, test_r2, accuracy, test_labels, test_preds = test_model(net, test_loader, criterion)
#     logging.info(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}, Accuracy: {accuracy:.2f}%")
    
#     # Plot and save the predicted vs actual plot
#     plt.figure(figsize=(8, 6))
#     plt.scatter(test_labels, test_preds, alpha=0.7)
#     plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'r--')
#     plt.xlabel("Actual Values")
#     plt.ylabel("Predicted Values")
#     plt.title("Predicted vs Actual Yield (Test Set)")
#     plt.savefig('predicted_vs_actual_plot.png')
#     plt.show()

#     # Visualize attention weights after testing
#     visualize_attention_weights(net)


# # Run the testing function after train_and_validate
# run_test(best_fold)

# Testing on the test dataset
def test_model_with_masks(model, test_loader, criterion):
    model.eval()
    test_loss, all_labels, all_preds = 0.0, [], []

    with torch.no_grad():
        for inputs, masks, metadata, labels in test_loader:
            # Move data to the device
            inputs, masks, metadata, labels = (
                inputs.to(device),
                masks.to(device),
                metadata.to(device),
                labels.to(device)
            )

            # Forward pass with masks
            outputs = model(inputs, masks, metadata)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

    test_loss /= len(test_loader)
    test_mae = mean_absolute_error(all_labels, all_preds)
    test_r2 = r2_score(all_labels, all_preds)

    # Calculate accuracy
    mean_actual = np.mean(all_labels)  # Mean of actual values
    accuracy = 100 * (1 - (test_mae / mean_actual))

    return test_loss, test_mae, test_r2, accuracy, all_labels, all_preds


def run_test_with_masks(best_fold):
    # Initialize the model
    net = YieldNetWithMasks().to(device)
    net.load_state_dict(torch.load(f'best_model_fold{best_fold}.pth'))

    # Load test dataset and DataLoader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Define the criterion
    criterion = nn.MSELoss()

    # Run the test
    test_loss, test_mae, test_r2, accuracy, test_labels, test_preds = test_model_with_masks(net, test_loader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}, Accuracy: {accuracy:.2f}%")

    # Plot and save the predicted vs actual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(test_labels, test_preds, alpha=0.7)
    plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual Yield (Test Set)")
    plt.savefig('predicted_vs_actual_plot.png')
    plt.show()

    # Visualize attention weights after testing
    # visualize_attention_weights(net)


# Update the call to the test function
run_test_with_masks(3)

def extract_fc2_weights_with_metadata(model_class, dataset, best_fold_model_path, save_path="fc2_weights_with_metadata.csv"):
    """
    Extract outputs of the final fully connected layer (`fc2`) for all samples along with their metadata.

    Args:
    - model_class: The class of the model (YieldNetWithMasks).
    - dataset: The full dataset of 195 samples.
    - best_fold_model_path: Path to the saved model for the best fold.
    - save_path: Path to save the extracted `fc2` outputs and metadata.
    """
    # DataLoader for the entire dataset
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the model for the best fold
    model = model_class().to(device)
    model.load_state_dict(torch.load(best_fold_model_path))
    model.eval()

    fc2_outputs = []

    # Process the entire dataset
    with torch.no_grad():
        for batch_idx, (inputs, masks, metadata, labels) in enumerate(data_loader):
            inputs, masks, metadata, labels = (
                inputs.to(device),
                masks.to(device),
                metadata.to(device),
                labels.to(device),
            )

            # Forward pass
            activations = model.efficientnet(inputs)
            mask_features = model.mask_pool(F.relu(model.mask_conv2(F.relu(model.mask_conv1(masks)))))
            mask_features = mask_features.view(mask_features.size(0), -1)
            metadata_features = F.relu(model.meta_fc3(F.relu(model.meta_fc2(F.relu(model.meta_fc1(model.bn_meta(metadata)))))))
            concatenated_activations = torch.cat((activations, mask_features, metadata_features), dim=1)
            fc1_output = F.relu(model.fc1(concatenated_activations))
            fc2_output = model.fc2(fc1_output)  # Final layer output

            # Collect fc2 outputs and metadata for each sample
            for i in range(inputs.size(0)):
                metadata_row = metadata[i].cpu().numpy()  # Convert metadata to numpy for easier handling
                fc2_outputs.append({
                    "Sample_ID": dataset.data.loc[batch_idx * data_loader.batch_size + i, 'Name'],
                    "fc2_Output": fc2_output[i].item(),
                    "True_Label": labels[i].item(),
                    "Location": metadata_row[0],
                    "Year": metadata_row[1],
                    "Reps": metadata_row[2],
                    "Pop": metadata_row[3],
                    "SD": metadata_row[4],
                })

    # Convert to DataFrame
    df = pd.DataFrame(fc2_outputs)

    # Save to CSV
    df.to_csv(save_path, index=False)
    logger.info(f"Saved fc2 outputs with metadata for all 195 samples to {save_path}")



# # Call the function with the best fold's model
extract_fc2_weights_with_metadata(
    model_class=YieldNetWithMasks,
    dataset=train_dataset,  # Replace with your full dataset
    best_fold_model_path="best_model_fold3.pth",  # Path to the best fold's model
    save_path="best_fold_outputs_train.csv"
)

def extract_fc1_outputs_with_metadata(model_class, dataset, best_fold_model_path, save_path="fc1_outputs_with_metadata.csv"):
    """
    Extract outputs of the `fc1` layer for all samples along with their metadata.

    Args:
    - model_class: The class of the model (YieldNetWithMasks).
    - dataset: The full dataset of 195 samples.
    - best_fold_model_path: Path to the saved model for the best fold.
    - save_path: Path to save the extracted `fc1` outputs and metadata.
    """
    # DataLoader for the entire dataset
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the model for the best fold
    model = model_class().to(device)
    model.load_state_dict(torch.load(best_fold_model_path))
    model.eval()

    fc1_outputs = []

    # Process the entire dataset
    with torch.no_grad():
        for batch_idx, (inputs, masks, metadata, labels) in enumerate(data_loader):
            inputs, masks, metadata, labels = (
                inputs.to(device),
                masks.to(device),
                metadata.to(device),
                labels.to(device),
            )

            # Forward pass
            activations = model.efficientnet(inputs)
            mask_features = model.mask_pool(F.relu(model.mask_conv2(F.relu(model.mask_conv1(masks)))))
            mask_features = mask_features.view(mask_features.size(0), -1)
            metadata_features = F.relu(model.meta_fc3(F.relu(model.meta_fc2(F.relu(model.meta_fc1(model.bn_meta(metadata)))))))
            concatenated_activations = torch.cat((activations, mask_features, metadata_features), dim=1)
            fc1_output = F.relu(model.fc1(concatenated_activations))  # Extract fc1 output

            # Collect fc1 outputs and metadata for each sample
            for i in range(inputs.size(0)):
                metadata_row = metadata[i].cpu().numpy()  # Convert metadata to numpy for easier handling
                fc1_outputs.append({
                    "Sample_ID": dataset.data.loc[batch_idx * data_loader.batch_size + i, 'Name'],
                    **{f"fc1_Output_{j}": fc1_output[i, j].item() for j in range(fc1_output.size(1))},  # Save all fc1 output features
                    "True_Label": labels[i].item(),
                    "Location": metadata_row[0],
                    "Year": metadata_row[1],
                    "Reps": metadata_row[2],
                    "Pop": metadata_row[3],
                    "SD": metadata_row[4],
                })

    # Convert to DataFrame
    df = pd.DataFrame(fc1_outputs)

    # Save to CSV
    df.to_csv(save_path, index=False)
    logger.info(f"Saved fc1 outputs with metadata for all 195 samples to {save_path}")


# # Call the function with the best fold's model
extract_fc1_outputs_with_metadata(
    model_class=YieldNetWithMasks,
    dataset=train_dataset,  # Replace with your full dataset
    best_fold_model_path="best_model_fold3.pth",  # Path to the best fold's model
    save_path="fc1_outputs_train_fc1.csv"
)


# aggregate_outputs_across_folds(
#     model_class=YieldNetWithMasks,
#     dataset=train_dataset,  # Your dataset
#     model_paths=[f"best_model_fold{i+1}.pth" for i in range(7)],
#     n_splits=7,
#     save_path="final_outputs_train.csv"
# )

def extract_fc2_weights(model_class, best_fold_model_path, save_path="fc2_layer_weights.csv"):
    """
    Extract weights of the final fully connected layer (`fc2`).

    Args:
    - model_class: The class of the model (YieldNetWithMasks).
    - best_fold_model_path: Path to the saved model for the best fold.
    - save_path: Path to save the extracted `fc2` weights.
    """
    # Load the model for the best fold
    model = model_class().to(device)
    model.load_state_dict(torch.load(best_fold_model_path))
    model.eval()

    # Extract `fc2` weights and bias
    fc2_weights = model.fc2.weight.data.cpu().numpy()
    fc2_bias = model.fc2.bias.data.cpu().numpy()

    # Save weights and bias to a CSV file
    weight_df = pd.DataFrame(fc2_weights, columns=[f"Weight_{i}" for i in range(fc2_weights.shape[1])])
    weight_df["Bias"] = fc2_bias
    weight_df.to_csv(save_path, index=False)

    logger.info(f"Saved fc2 weights and bias to {save_path}")

extract_fc2_weights(
    model_class=YieldNetWithMasks,
    best_fold_model_path="best_model_fold3.pth",
    save_path="fc2_layer_weights.csv"
)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader
import shap

# Logger setup (assuming logger is defined elsewhere in your script)
# logger.info for logging progress or error reporting

def adjust_layer_weights(layer_name, current_weights, target_shape):
    """
    Adjust weights dynamically to match target shape.
    Args:
        layer_name: Name of the layer being adjusted.
        current_weights: Existing weights of the layer.
        target_shape: Expected shape for the layer weights.
    Returns:
        Adjusted weights.
    """
    new_weights = torch.zeros(target_shape, device=current_weights.device)
    if len(current_weights.size()) == 1:  # For 1D tensors
        min_len = min(current_weights.size(0), target_shape[0])
        new_weights[:min_len] = current_weights[:min_len]
    elif len(current_weights.size()) == 2:  # For 2D tensors
        min_rows = min(current_weights.size(0), target_shape[0])
        min_cols = min(current_weights.size(1), target_shape[1])
        new_weights[:min_rows, :min_cols] = current_weights[:min_rows, :min_cols]
    print(f"Adjusted {layer_name} to {new_weights.shape}")
    return new_weights


class MetadataModelWrapper:
    def __init__(self, model, device):
        """
        Initialize the MetadataModelWrapper.

        Args:
            model: The trained PyTorch model.
            device: The device (CPU or GPU) to run the model on.
        """
        self.model = model
        self.device = device

        # Extract metadata size from the model's metadata pathway
        metadata_size = self.model.meta_fc3.out_features

        # Define a new fully connected layer for metadata-specific predictions
        self.metadata_fc2 = nn.Linear(metadata_size, 1).to(self.device)
        self.metadata_fc2.weight = nn.Parameter(self.model.fc2.weight[:, :metadata_size])
        self.metadata_fc2.bias = self.model.fc2.bias

    def __call__(self, metadata):
        """
        Process metadata through the model's metadata pathway.

        Args:
            metadata: NumPy array of metadata features.

        Returns:
            NumPy array of predictions based on metadata alone.
        """
        metadata = torch.tensor(metadata, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            metadata_processed = F.relu(
                self.model.meta_fc3(
                    F.relu(
                        self.model.meta_fc2(
                            F.relu(self.model.meta_fc1(self.model.bn_meta(metadata)))
                        )
                    )
                )
            )
            final_output = self.metadata_fc2(metadata_processed)
            return final_output.cpu().numpy()


def preprocess_metadata_for_shap(metadata, feature_names):
    """
    Preprocess metadata for SHAP explanations, including scaling and one-hot encoding.
    
    Args:
        metadata: PyTorch tensor or NumPy array of metadata.
        feature_names: List of feature names corresponding to metadata columns.

    Returns:
        Transformed metadata and the transformer object.
    """
    pop_index = feature_names.index("Pop")
    metadata_numpy = metadata.cpu().numpy()

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [i for i in range(len(feature_names)) if i != pop_index]),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [pop_index]),
        ]
    )
    metadata_transformed = transformer.fit_transform(metadata_numpy)

    # Debugging: Print statistics
    print("Feature means after scaling:", transformer.named_transformers_["num"].mean_)
    print("Feature variances after scaling:", transformer.named_transformers_["num"].var_)

    return metadata_transformed, transformer


def explain_model_predictions(model, dataset, feature_names, output_path="shap_explanations.png"):
    """
    Generate SHAP explanations for model predictions.

    Args:
        model: Trained PyTorch model.
        dataset: Dataset containing inputs and metadata.
        feature_names: List of feature names.
        output_path: Path to save SHAP summary plot.
    """
    model_wrapper = MetadataModelWrapper(model, device)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Aggregate all metadata
    all_metadata = []
    for _, _, metadata, _ in loader:
        all_metadata.append(metadata.cpu().numpy())
    all_metadata = np.vstack(all_metadata)

    # Initialize SHAP explainer and compute SHAP values
    explainer = shap.Explainer(model_wrapper, all_metadata)
    shap_values = explainer(all_metadata)

    # **Enlarge the SHAP plot by adjusting the figure size**
    plt.figure(figsize=(12, 8))  # Increase the figure size (e.g., 12x8 inches)
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.savefig(output_path, dpi=300)  # Save the plot with high DPI for better quality
    print(f"Saved enlarged SHAP explanations to {output_path}")



def test_model_with_logging(model, test_loader, criterion):
    """
    Test the model with detailed logging.
    
    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test dataset.
        criterion: Loss function.

    Returns:
        Test metrics and predictions.
    """
    try:
        loss, mae, r2, accuracy, labels, predictions = test_model_with_masks(model, test_loader, criterion)
        print(f"Test Loss: {loss:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Accuracy: {accuracy:.2f}%")
        return labels, predictions
    except Exception as e:
        print(f"Error during testing: {e}")
        raise

from torch.utils.data import DataLoader

# Define feature names
metadata_feature_names = ["Location", "Year", "Reps", "Pop", "SD"]

# Initialize the dataset
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = YieldDatasetWithMasks(
    csv_file='output_vectors_Test.csv',
    root_dir='Test/',
    mask_dir='images_AF_combined/masks/',
    transform=test_transform
)

# Create a DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the best model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YieldNetWithMasks(metadata_features=5).to(device)
model.load_state_dict(torch.load("best_model_fold3.pth"))
model.eval()

explain_model_predictions(
    model=model,
    dataset=test_dataset,
    feature_names=metadata_feature_names,
    output_path="shap_explanations.png"
)

criterion = nn.MSELoss()
labels, predictions = test_model_with_logging(model, test_loader, criterion)

