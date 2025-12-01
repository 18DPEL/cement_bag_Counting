import torch
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO

# Download pretrained DnCNN model from torch hub
model = torch.hub.load('cszn/DnCNN', 'dncnn', pretrained=True)
model.eval()

# Load image
img = Image.open(r"C:\Users\ayuba\OneDrive\Desktop\unnamed.jpg").convert("RGB")

# Preprocess
transform = T.Compose([
    T.ToTensor(),
])
input_tensor = transform(img).unsqueeze(0)

# Run model
with torch.no_grad():
    output = model(input_tensor)

# Postprocess
output_img = T.ToPILImage()(output.squeeze().clamp(0,1))

# Save
output_img.save("denoised_dncnn.jpg")
output_img.show()



