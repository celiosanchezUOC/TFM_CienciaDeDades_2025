import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

if __name__ == '__main__':
    # Configuration parameters
    IMG_SIZE = 256  # Swin Transformer can handle various sizes
    BATCH_SIZE = 16  # Reduced for offline training with limited resources
    EPOCHS = 30
    INITIAL_LR = 1e-4
    FINE_TUNE_LR = 1e-5
    NUM_CLASSES = 2
    CLASS_NAMES = ['benign', 'malign']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths to data directories
    base_dir = 'DatasetTotal'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Path to pretrained weights (must be downloaded in advance and placed in this location)
    PRETRAINED_WEIGHTS_PATH = 'swinv2_tiny_patch4_window8_256.pth'

    # Custom Swin Transformer V2 implementation (simplified for offline use)
    class SwinTransformerV2(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            
            # Basic configuration (tiny version)
            self.patch_embed = nn.Conv2d(3, 96, kernel_size=4, stride=4)
            self.num_layers = 4
            self.embed_dim = 96
            self.depths = [2, 2, 6, 2]
            self.num_heads = [3, 6, 12, 24]
            
            # Create transformer blocks
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = nn.ModuleList([
                    SwinTransformerBlock(
                        dim=self.embed_dim * (2 ** i_layer),
                        num_heads=self.num_heads[i_layer],
                        window_size=8,
                        shift_size=0 if (i % 2 == 0) else 8 // 2
                    ) for i in range(self.depths[i_layer])
                ])
                self.layers.append(layer)
                
                if i_layer < self.num_layers - 1:
                    self.layers.append(PatchMerging(self.embed_dim * (2 ** i_layer)))
            
            # Classification head
            self.norm = nn.LayerNorm(self.embed_dim * (2 ** (self.num_layers - 1)))
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim * (2 ** (self.num_layers - 1)), 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            
            for layer in self.layers:
                if isinstance(layer, nn.ModuleList):
                    for blk in layer:
                        x = blk(x)
                else:
                    x = layer(x)
            
            x = self.norm(x)
            x = self.avgpool(x.transpose(1, 2))
            x = torch.flatten(x, 1)
            x = self.head(x)
            return x

    # Simplified Swin Transformer Block (for offline implementation)
    class SwinTransformerBlock(nn.Module):
        def __init__(self, dim, num_heads, window_size=8, shift_size=0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = WindowAttention(
                dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size
            )
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        
        def forward(self, x):
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + x
            
            shortcut = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = shortcut + x
            return x

    # Simplified Window Attention (for offline implementation)
    class WindowAttention(nn.Module):
        def __init__(self, dim, num_heads, window_size, shift_size=0):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.window_size = window_size
            self.shift_size = shift_size
            
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            
            # Relative position bias (simplified)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            
            # Initialize relative position bias
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
        
        def forward(self, x):
            B, N, C = x.shape
            window_size = self.window_size
            assert int(N ** 0.5) ** 2 == N, "Input sequence length must be a perfect square"
            H = W = int(N ** 0.5)
            assert H % window_size == 0 and W % window_size == 0, "H and W must be divisible by window_size"

            # Reshape x to windows
            x = x.view(B, H, W, C)
            x = x.unfold(1, window_size, window_size).unfold(2, window_size, window_size)
            x = x.contiguous().view(-1, window_size * window_size, C)  # (num_windows*B, window_size*window_size, C)

            qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            attn = attn / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32, device=x.device))

            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                window_size * window_size, window_size * window_size, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

            attn = F.softmax(attn, dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], C)
            x = self.proj(x)

            # Merge windows back
            x = x.view(B, H // window_size, W // window_size, window_size, window_size, C)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
            x = x.view(B, N, C)
            return x

    # Patch Merging (for offline implementation)
    class PatchMerging(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = nn.LayerNorm(4 * dim)
        
        def forward(self, x):
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            
            x = x.view(B, H, W, C)
            
            x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            
            x = x.view(B, -1, 4 * C)
            x = self.norm(x)
            x = self.reduction(x)
            return x

    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize model
    model = SwinTransformerV2(num_classes=NUM_CLASSES)

    # Load pretrained weights if available
    if os.path.exists(PRETRAINED_WEIGHTS_PATH):
        try:
            state_dict = torch.load(PRETRAINED_WEIGHTS_PATH, map_location='cpu')
            # Adapt state dict to our simplified model (this may need adjustment)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained weights successfully")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
    else:
        print("No pretrained weights found, training from scratch")

    model = model.to(DEVICE)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training and validation functions (same as before)
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # Training loop (same as before)
    def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
        best_val_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_swin_v2_model_offline.pth')
                print('Saved best model')
        
        return history

    # Initial training (only head)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    print("Training head only...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS//2, DEVICE)

    # Fine-tuning (unfreeze all layers)
    for param in model.parameters():
        param.requires_grad = True

    # Use lower learning rate for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS//2)

    print("Fine-tuning all layers...")
    history_fine = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS//2, DEVICE)

    # Combine histories
    full_history = {
        'train_loss': history['train_loss'] + history_fine['train_loss'],
        'train_acc': history['train_acc'] + history_fine['train_acc'],
        'val_loss': history['val_loss'] + history_fine['val_loss'],
        'val_acc': history['val_acc'] + history_fine['val_acc']
    }

    # Plot training history (same as before)
    def plot_training_history(history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    plot_training_history(full_history)

    # Evaluation on test set (same as before)
    model.load_state_dict(torch.load('best_swin_v2_model_offline.pth'))
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            prob = torch.softmax(outputs, dim=1)[:, 1]
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    # Confusion matrix
    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    plot_confusion_matrix(y_true, y_pred)

    # Classification report
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ROC curve
    def plot_roc_curve(y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    plot_roc_curve(y_true, y_prob)

    # Precision-Recall curve
    def plot_precision_recall_curve(y_true, y_prob):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        average_precision = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                 label=f'Precision-Recall curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()

    plot_precision_recall_curve(y_true, y_prob)

    # Save final model
    torch.save(model.state_dict(), 'swin_v2_final_model_offline.pth')