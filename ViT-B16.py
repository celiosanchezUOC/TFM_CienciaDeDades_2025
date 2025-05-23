import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device, calc_probs=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            probs = nn.functional.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            if calc_probs:
                all_probs.extend(probs)
    avg_loss = running_loss / total
    avg_acc = correct / total
    if calc_probs:
        return avg_loss, avg_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)
    else:
        return avg_loss, avg_acc

if __name__ == '__main__':
    # === CONFIGURACIÓN GENERAL ===
    DATASET_ROOT = "Dataset"
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 2  # benign, malign
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === TRANSFORMACIONES DE IMAGEN ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # === DATASETS Y DATALOADERS ===
    train_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "val"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # === INTEGRACIÓN OFFLINE: CARGA MANUAL DE PESOS PREENTRENADOS ===
    VIT_B16_WEIGHTS_PATH = "vit_b_16-c867db91.pth"  # ruta del archivo

    model = models.vit_b_16(weights=None)  # No descarga nada online
    # Carga los pesos preentrenados de imagenet
    state_dict = torch.load(VIT_B16_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    # Reemplaza la cabeza para 2 clases
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # === PÉRDIDA Y OPTIMIZADOR ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # === ENTRENAMIENTO ===
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)[:2]
        print(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f} - Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        # Guarda el mejor modelo según validación
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "best_vit_b16.pth")
            best_val_acc = val_acc

    # Guarda el modelo final entrenado
    torch.save(model.state_dict(), "final_vit_b16.pth")

    # === EVALUACIÓN FINAL EN TEST CON MÉTRICAS COMPLETAS ===
    model.load_state_dict(torch.load("best_vit_b16.pth", map_location=DEVICE))
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, DEVICE, calc_probs=True)
    print(f"Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    # === MÉTRICAS ===
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=test_dataset.classes)
    roc_auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", conf_mat)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # === GUARDADO DE MÉTRICAS ===
    with open("vit_b16_metrics.txt", "w") as f:
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Test acc: {test_acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(conf_mat))

    # === VISUALIZACIONES ===
    plt.figure(figsize=(5,5))
    plt.imshow(conf_mat, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], test_dataset.classes)
    plt.yticks([0, 1], test_dataset.classes)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, conf_mat[i, j], ha="center", va="center", color="red")
    plt.tight_layout()
    plt.savefig("vit_b16_confusion_matrix.png")
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("vit_b16_roc_curve.png")
    plt.show()