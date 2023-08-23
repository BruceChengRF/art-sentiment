import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
from PIL import Image
from torchvision.models import ResNet50_Weights

Image.MAX_IMAGE_PIXELS = 150000000
# Configurations
data_dir = 'wikiart'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
num_epochs = 25
batch_size = 32
lr = 0.0001
patience = 5  # For early stopping

# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),  # Resizing step
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),  # Resizing step
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# Load the entire Wikiart dataset
full_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])

# Split the dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create dataloaders for the training and validation sets
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}

models_to_train = ['deit', 'densenet', 'resnet']

for model_name in models_to_train:
    print(f"\nTraining {model_name.upper()}...")
    if model_name == 'densenet':
        model = models.densenet121(weights=models.densenet.DenseNet121_Weights.IMAGENET1K_V1)  # CHANGE: Use weights argument for DenseNet
    elif model_name == 'resnet':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # CHANGE: Use weights argument for ResNet
    elif model_name == 'deit':
        model = timm.create_model('deit_small_patch16_224', pretrained=True)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Modification 4: Switch to AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Modification 5: Add OneCycleLR scheduler
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=25, steps_per_epoch=len(dataloaders['train']))

    def train_model(model, criterion, optimizer, num_epochs=25, patience=5):
        best_loss = float('inf')
        no_improve_epochs = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                    running_loss += loss.item() * inputs.size(0)
                    corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = corrects.double() / len(dataloaders[phase].dataset)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Early stopping and model checkpointing
                if phase == 'val':
                    if epoch_loss < best_loss:
                        print(f"Improved validation loss from {best_loss:.4f} to {epoch_loss:.4f}. Saving model...")
                        best_loss = epoch_loss
                        torch.save(model.state_dict(), f'{model_name}_wikiart_best.pth')
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                        if no_improve_epochs >= patience:
                            print(f"No improvement for {patience} epochs. Stopping training...")
                            return model

            print()

        return model

    train_model(model, criterion, optimizer, num_epochs, patience)

print("\nTraining completed!")
