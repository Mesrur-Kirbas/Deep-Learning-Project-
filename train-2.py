import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import PlantDiseaseCNN

# AYARLAR
DATA_DIR = 'data'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
IMG_SIZE = 128

# Veri Ön İşleme
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

if __name__ == '__main__':
    # Veri Yükleme
    print("Veriler yükleniyor...")
    try:
        full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    except FileNotFoundError:
        print("HATA: 'data' klasörü bulunamadı. Lütfen zip dosyasını çıkarın.")
        exit()

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model Kurulumu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan Cihaz: {device}")

    model = PlantDiseaseCNN(num_classes=len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Eğitim
    print("Eğitim Başlıyor...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Kayıp: {running_loss/len(train_loader):.4f}")

    # Kaydetme
    torch.save(model.state_dict(), "plant_model.pth")
    print("Model başarıyla eğitildi ve kaydedildi!")
