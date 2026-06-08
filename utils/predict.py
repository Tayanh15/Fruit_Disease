import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
import json
from datetime import datetime

# ========================
# Tạo thư mục nếu chưa có
# ========================
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ========================
# Danh sách model theo trái cây
# ========================
fruit_models = {
    "apple": "model/apple.pth",
    "mango": "model/mango.pth",
    "Luu": "model/Luu.pth",
    "Tonghop": "model/Tonghop.pth"
}

fruit_classes = {
    "apple": ["Anthracnose","Black_Pox","Black_Rot","Codling_Moth","Healthy","Powdery_Mildew"],
    "mango": ["Anthracnose","Bacterial Canker","Healthy","Scab","Stem End Rot"],
    "Luu": ["Alternaria","Anthracnose","Bacterial_Blight","Cercospora","Healthy"],
    "Tonghop": ["Apple__Healthy","Apple__Rotten","Banana__Healthy","Banana__Rotten","Bellpepper__Healthy","Bellpepper__Rotten","Carrot__Healthy","Carrot__Rotten","Cucumber__Healthy","Cucumber__Rotten","Grape__Healthy","Grape__Rotten","Guava__Healthy","Guava__Rotten","Jujube__Healthy","Jujube__Rotten","Mango__Healthy","Mango__Rotten","Orange__Healthy","Orange__Rotten","Pomegranate__Healthy","Pomegranate__Rotten","Potato__Healthy","Potato__Rotten","Strawberry__Healthy","Strawberry__Rotten","Tomato__Healthy","Tomato__Rotten"]
}
# ========================
# Transform ảnh
# ========================
stats = (
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225)
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])


# ========================
# Load model theo fruit_type
# ========================
def load_model(model_path, num_classes):
    model = models.resnet50(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    model.eval()

    return model


# ========================
# Grad-CAM
# ========================
def generate_gradcam(model, img_tensor, class_idx):
    gradients = []
    activations = []

    target_layer = model.layer4[-1]

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)

    model.zero_grad()

    loss = output[0, class_idx]
    loss.backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)

    cam = cv2.resize(cam, (224, 224))
    cam = cam / (cam.max() + 1e-8)

    handle_fw.remove()
    handle_bw.remove()

    return cam


# ========================
# Ghép heatmap vào ảnh
# ========================
def overlay_heatmap(image, cam):
    image = np.array(image)

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay


# ========================
# Lưu lịch sử
# ========================
def save_history(original_path, result, confidence, heatmap_path):
    history_file = "data/history.json"

    if os.path.exists(history_file):
        try:
            with open(
                    history_file,
                    "r",
                    encoding="utf-8"
            ) as f:
                history = json.load(f)
        except:
            history = []
    else:
        history = []

    history.append({
        "image": original_path,
        "result": result,
        "confidence": round(confidence, 4),
        "boxed": heatmap_path,
        "time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    })

    with open(
            history_file,
            "w",
            encoding="utf-8"
    ) as f:
        json.dump(
            history,
            f,
            indent=4,
            ensure_ascii=False
        )


# ========================
# Predict chính
# ========================
def predict_image(img_path, fruit_type):
    # kiểm tra fruit_type
    if fruit_type not in fruit_models:
        raise ValueError(
            f"Không hỗ trợ loại trái cây: {fruit_type}"
        )

    model_path = fruit_models[fruit_type]
    classes = fruit_classes[fruit_type]

    # load model tương ứng
    model = load_model(
        model_path,
        len(classes)
    )

    # đọc ảnh
    image = Image.open(img_path).convert("RGB")
    image_resized = image.resize((224, 224))

    img_tensor = transform(image).unsqueeze(0)

    # predict
    outputs = model(img_tensor)

    probs = torch.softmax(outputs, dim=1)

    predicted = torch.argmax(probs, dim=1)

    class_idx = predicted.item()

    confidence = round(
        probs[0][class_idx].item() * 100,3
    )

    result = classes[class_idx]

    # Grad-CAM
    cam = generate_gradcam(
        model,
        img_tensor,
        class_idx
    )

    heatmap_img = overlay_heatmap(
        image_resized,
        cam
    )

    # lưu heatmap
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    heatmap_path = os.path.join(
        "static/uploads",
        f"cam_{timestamp}.jpg"
    )

    cv2.imwrite(
        heatmap_path,
        heatmap_img
    )

    # lưu history
    save_history(
        img_path,
        result,
        confidence,
        heatmap_path
    )

    return result, confidence, heatmap_path