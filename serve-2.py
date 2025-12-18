import gradio as gr
import torch
from torchvision import transforms
from model import PlantDiseaseCNN
from PIL import Image
import os

# --- AYARLAR ---
MODEL_PATH = "plant_model.pth"

# SINIF LİSTESİ (Alfabetik Sıraya Göre: Early -> Healthy -> Late)
CLASSES = [
    'Erken Yanıklık (Early Blight)', 
    'Sağlıklı (Healthy)', 
    'Geç Yanıklık (Late Blight)'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modelimiz artık 3 sınıfı tanıyor
model = PlantDiseaseCNN(num_classes=3).to(device)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(" Model başarıyla yüklendi (3 Sınıf modunda)!")
    except RuntimeError as e:
        print(f" Model yükleme hatası: {e}")
        print("Model boyutu ile kod uyuşmuyor. Lütfen train.py'yi çalıştırdığınızdan emin olun.")
else:
    print(" Model dosyası yok! Önce eğitimi çalıştırın.")

def predict_plant(image):
    if image is None: return None
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
    return {CLASSES[i]: float(probs[i])/100 for i in range(len(CLASSES))}

interface = gr.Interface(
    fn=predict_plant,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Patates Hastalık Tespiti (Tam Kapsamlı)",
    description="Yaprak resmini yükleyin. Model: Erken Yanıklık, Geç Yanıklık veya Sağlıklı teşhisi koysun."
)

if __name__ == "__main__":
    interface.launch(share=True)
