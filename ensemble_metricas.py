import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer

# --- SWIN TRANSFORMER V2 & sub-classes ---
class SwinTransformerV2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.num_layers = 4
        self.embed_dim = 96
        self.depths = [2, 2, 6, 2]
        self.num_heads = [3, 6, 12, 24]
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

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
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
        x = x.view(B, H, W, C)
        x = x.unfold(1, window_size, window_size).unfold(2, window_size, window_size)
        x = x.contiguous().view(-1, window_size * window_size, C)
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        attn = attn / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32, device=x.device))
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            window_size * window_size, window_size * window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], C)
        x = self.proj(x)
        x = x.view(B, H // window_size, W // window_size, window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        x = x.view(B, N, C)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x
    
class CustomScaleLayer(Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        # Verifica si la entrada es una lista
        if isinstance(inputs, list):
            # Combina los tensores de la lista (por ejemplo, sumándolos)
            inputs = tf.add_n(inputs)
        # Aplica la escala
        return inputs * self.scale

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale': self.scale})
        return config

# --- Carga modelos Keras ---
densenet = load_model("Classif_3_DenseNet201.h5")
xception = load_model("Classif_3_Xception.h5")
inception = load_model("Classif_3_inceptionresnetv2.h5", custom_objects={'CustomScaleLayer': CustomScaleLayer})
#inception = load_model("Classif_1_inceptionresnetv2_complete.h5", custom_objects={})

# --- Carga modelos PyTorch ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swin = SwinTransformerV2(num_classes=2)
swin.load_state_dict(torch.load("Classif_3_SwinV2.pth", map_location=device))
swin.eval().to(device)

vit = models.vit_b_16(weights=None)
vit.heads.head = nn.Linear(vit.heads.head.in_features, 2)
vit.load_state_dict(torch.load("Classif_3_ViT-B16.pth", map_location=device))
vit.eval().to(device)

# --- Preprocesado por modelo ---
def preprocess_keras(img_path, img_size):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img = tf.keras.utils.img_to_array(img)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_torch(img_path, img_size):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

# --- Predicción individual ---
def pred_keras(model, img):
    prob = model.predict(img)[0]
    if prob.shape == ():  # escalar (sigmoid)
        prob = np.array([1-prob, prob])
    elif len(prob.shape) == 1 and prob.shape[0] == 1:  # sigmoid
        prob = np.array([1-prob[0], prob[0]])
    elif len(prob.shape) == 1 and prob.shape[0] == 2:  # softmax
        pass
    return prob

def pred_torch(model, img, device):
    with torch.no_grad():
        img = img.to(device)
        logits = model(img)
        if logits.shape[-1] == 1:  # Sigmoid
            prob = torch.sigmoid(logits).cpu().numpy()[0]
            prob = np.array([1-prob, prob])
        else:  # Softmax
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return prob

# --- Ensemble predict ---
def ensemble_predict(img_path):
    img_dn    = preprocess_keras(img_path, (224,224))
    img_xcept = preprocess_keras(img_path, (299,299))
    img_incept = preprocess_keras(img_path, (299,299))
    img_swin  = preprocess_torch(img_path, (256,256))
    img_vit   = preprocess_torch(img_path, (224,224))
    p_dn    = pred_keras(densenet, img_dn)
    p_xcept = pred_keras(xception, img_xcept)
    p_incept= pred_keras(inception, img_incept)
    p_swin  = pred_torch(swin, img_swin, device)
    p_vit   = pred_torch(vit, img_vit, device)
    probs = np.stack([p_dn, p_xcept, p_incept, p_swin, p_vit])
    avg_probs = probs.mean(axis=0)
    pred_class = np.argmax(avg_probs)
    return avg_probs, pred_class

# --- Métricas sobre todo el test ---
def get_test_images_and_labels(test_root):
    img_paths = []
    y_true = []
    for label_name, label_idx in [("benign", 0), ("malignant", 1)]:
        folder = os.path.join(test_root, label_name)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                img_paths.append(os.path.join(folder, fname))
                y_true.append(label_idx)
    return img_paths, y_true

if __name__ == "__main__":
    # Cambia esto a la ruta de tu conjunto de test
    test_root = "DatasetTotal/test"
    img_paths, y_true = get_test_images_and_labels(test_root)

    y_pred = []
    y_prob = []
    for img_path in img_paths:
        probs, pred = ensemble_predict(img_path)
        y_pred.append(pred)
        y_prob.append(probs[1])  # probabilidad de 'malignant'

    # Calcula métricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['benign','malignant'])
    rocauc = roc_auc_score(y_true, y_prob)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("ROC AUC:", rocauc)
    print("Confusion Matrix:\n", conf_mat)
    print("Classification Report:\n", report)

    # Guarda métricas en un archivo
    with open("ensemble_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {rocauc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_mat))
        f.write("\nClassification Report:\n")
        f.write(report)

    # Matriz de confusión (dibujar y guardar)
    plt.figure(figsize=(5,5))
    plt.imshow(conf_mat, cmap='Blues')
    plt.title("Matriu de confusió(ensemble)")
    plt.xlabel("Predita")
    plt.ylabel("Real")
    plt.xticks([0, 1], ['benign', 'malignant'])
    plt.yticks([0, 1], ['benign', 'malignant'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, conf_mat[i, j], ha="center", va="center", color="red", fontsize=16)
    plt.tight_layout()
    plt.savefig("ensemble_confusion_matrix.png")
    plt.show()

    # (Opcional) Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {rocauc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC (Ensemble)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ensemble_roc_curve.png")
    plt.show()

