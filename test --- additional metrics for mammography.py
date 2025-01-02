import os
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
from transformers import ViTModel
import matplotlib.pyplot as plt
import seaborn as sns

"""
Poniżej znajduje się kod dostosowany do określania dodatkowych metryk testu, których nie ujęto w kodzie modelu 3
Korzysta się tutaj z najlepszych wag uzyskanych podczas trenowania danego modelu
Kod trzeba dostosować w zależności od parametrów modelu, na którym trenowano
"""


# ---------------------------------
# Funkcje pomocnicze
# ---------------------------------

def compute_per_class_accuracy(conf_matrix):
    """
    Oblicza accuracy dla każdej klasy na podstawie macierzy pomyłek.

    Args:
        conf_matrix (np.ndarray): Macierz pomyłek (confusion matrix).

    Returns:
        dict: Słownik z accuracy dla każdej klasy.
    """
    per_class_acc = {}
    total_samples = conf_matrix.sum()
    num_classes = conf_matrix.shape[0]

    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = total_samples - (TP + FP + FN)

        accuracy = (TP + TN) / total_samples
        per_class_acc[f"Grade {i}"] = accuracy

    return per_class_acc


# ---------------------------------
# Klasa Dataset dla PyTorch
# ---------------------------------

class MammographyDataset(Dataset):
    def __init__(self, images_folder, labels_csv, augment=False, mean=None, std=None):
        """
        Initializes the dataset by reading image paths and labels from a CSV file.
        """
        self.images_folder = images_folder
        self.labels_df = pd.read_csv(labels_csv)
        self.labels = self.labels_df['pathology'].astype(int).values
        self.filenames = self.labels_df['filename'].values
        self.augment = augment
        self.mean = mean
        self.std = std
        self.augment_counters = defaultdict(int)

        if augment:
            self.transforms = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.4),
                A.MotionBlur(blur_limit=5, p=0.4),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.4),
                A.Rotate(limit=10, p=0.1),
                A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.0), rotate=(-45, 45), shear=(-20, 20), p=0.5),
                A.Normalize(mean=mean.tolist(), std=std.tolist()),
                ToTensorV2()
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=mean.tolist(), std=std.tolist()),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        image = np.dstack([image] * 3).astype(np.float32) / 255.0

        if self.augment:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        else:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


# ---------------------------------
# Klasa Modelu Vision Transformer
# ---------------------------------

class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformerModel, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        vit_hidden_size = self.vit.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(vit_hidden_size, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        outputs = self.vit(pixel_values=images)
        features = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(features)
        return logits


# ---------------------------------
# Klasa Focal Loss
# ---------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else torch.tensor(1.0)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# ---------------------------------
# Funkcja Walidacji (na potrzeby testowania)
# ---------------------------------

def validate_network(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    report = classification_report(all_labels, all_preds,
                                   target_names=["Grade 0", "Grade 1"],
                                   zero_division=0,
                                   output_dict=True)
    print(
        classification_report(all_labels, all_preds,
                              target_names=["Grade 0", "Grade 1"],
                              zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    print("===== CONFUSION MATRIX =====")
    print(cm)

    cm_df = pd.DataFrame(cm,
                         index=["Grade 0", "Grade 1"],
                         columns=["Predicted Grade 0", "Predicted Grade 1"])
    print("\n===== CONFUSION MATRIX (DataFrame) =====")
    print(cm_df)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    per_class_acc = compute_per_class_accuracy(cm)
    print("\n===== PER-CLASS ACCURACY =====")
    for cls, acc in per_class_acc.items():
        print(f"{cls}: {acc * 100:.2f}%")

    return avg_loss, accuracy, report, cm, per_class_acc


# ---------------------------------
# Główna Funkcja Testująca
# ---------------------------------

def main():
    test_folder = r"C:\Users\a\Desktop\mammografia\manifest-1734734641099\MAMMOGRAFIAPNG\TEST"
    labels_test_csv_path = os.path.join(test_folder, "labels.csv")
    best_model_path = r"C:\Users\a\PycharmProjects\Praca inżynierska\best_model_mam3.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    mean_3chan = np.array([0.0021, 0.0021, 0.0021])
    std_3chan = np.array([0.0007, 0.0007, 0.0007])

    test_dataset = MammographyDataset(
        images_folder=test_folder,
        labels_csv=labels_test_csv_path,
        augment=False,
        mean=mean_3chan,
        std=std_3chan
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    num_classes = 2
    model = VisionTransformerModel(num_classes=num_classes).to(device)

    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Model załadowany pomyślnie.")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    criterion = FocalLoss(alpha=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device), gamma=3, reduction='mean')

    test_loss, test_acc, test_report, test_cm, test_per_class_acc = validate_network(model, test_loader, device,
                                                                                     criterion)

    print("\n=== WYNIKI NA ZBIORZE TESTOWYM ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print("Precision/Recall per klasa:")
    for grade_name, metrics in test_report.items():
        if grade_name.startswith("Grade"):
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            f1 = metrics.get('f1-score', 0.0)
            support = metrics.get('support', 0)
            print(
                f" - {grade_name}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1-Score = {f1:.4f}, Support = {support}")

    print("\n=== PER-CLASS ACCURACY ===")
    for cls, acc in test_per_class_acc.items():
        print(f"{cls}: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
