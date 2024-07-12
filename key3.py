import os
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from fastai.vision.all import *
from fastai.vision.widgets import *
import pytesseract
from torchvision.ops import MultiScaleRoIAlign

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class KeyHeadDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用預訓練的ResNet50作為骨幹網絡
        self.backbone = models.resnet50(pretrained=True)
        # 移除最後的全連接層
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # 添加注意力模塊
        self.attention = AttentionModule(2048)
        # 添加ROI Align層用於特徵提取
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        # 添加分類頭
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)  # 二元分類：是否為鑰匙頭
        )

    def forward(self, x, boxes):
        features = self.backbone(x)
        attended_features = self.attention(features)
        # 假設每個圖像只有一個ROI
        boxes_list = [boxes]
        roi_features = self.roi_align(
            {"0": attended_features},
            boxes_list,
            [attended_features.shape[2:]]
        )
        roi_features = roi_features.view(roi_features.size(0), -1)
        return self.classifier(roi_features)

class KeyDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_full_image=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_full_image = use_full_image
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.use_full_image:
            # 使用整張圖片作為感興趣區域
            box = [0, 0, image.width, image.height]
            label = 1  # 假設所有圖片都包含鑰匙頭
        else:
            # 使用標註文件
            annotation_file = img_name.rsplit('.', 1)[0] + '.txt'
            with open(annotation_file, 'r') as f:
                anno = f.read().strip().split()
            box = list(map(float, anno[:4]))
            label = int(anno[4])  # 假設 1 表示鑰匙頭，0 表示非鑰匙頭

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(box), torch.tensor(label, dtype=torch.float32)

def train_model(dataloader, model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        for images, boxes, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images, boxes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"紀元 {epoch+1}/{num_epochs}, 損失: {loss.item()}")

def detect_key_head(model, image_path):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    # 使用整張圖片作為感興趣區域
    boxes = torch.tensor([[0, 0, 223, 223]], dtype=torch.float32)
    
    with torch.no_grad():
        output = model(image_tensor, boxes)
    
    prob = torch.sigmoid(output).item()
    return prob > 0.5, prob

def recognize_text(image_path, lang='eng'):
    # 使用 Tesseract OCR 來識別英文和數字
    image = Image.open(image_path)
    # 設定 Tesseract 只識別英文字母和數字
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
    return text

def main():
    # 設定訓練數據路徑
    train_data_dir = "path/to/your/training/images"
    
    # 創建數據集和數據加載器
    train_dataset = KeyDataset(train_data_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]), use_full_image=True)  # 設置為 True 來使用整張圖片作為輸入
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 初始化模型
    model = KeyHeadDetector()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 訓練模型
    num_epochs = 10
    train_model(train_loader, model, criterion, optimizer, num_epochs)

    # 保存模型
    torch.save(model.state_dict(), "key_head_detector.pth")

    # 使用模型進行推理
    test_image_path = "path/to/your/test/image.jpg"
    is_key_head, confidence = detect_key_head(model, test_image_path)
    if is_key_head:
        print(f"偵測到鑰匙頭，信心度：{confidence:.2f}")
        text = recognize_text(test_image_path)
        print(f"識別到的英文和數字：{text}")
    else:
        print(f"未偵測到鑰匙頭，信心度：{confidence:.2f}")

if __name__ == "__main__":
    main()
    
    
