import os
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytesseract
from torchvision.ops import MultiScaleRoIAlign
import random
from torchvision.models import ResNet50_Weights
import json
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 定義注意力模組
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 生成注意力圖並應用到輸入特徵上
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

# 定義鑰匙頭檢測器模型
class KeyHeadDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用預訓練的ResNet50作為骨幹網絡
        self.backbone = models.resnet50(pretrained=True)
        # self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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
            nn.Linear(1024, 128)  # 128維的特徵向量用於自監督學習
        )

    def forward(self, x, boxes):
        features = self.backbone(x)
        attended_features = self.attention(features)
        # 假設每個圖像只有一個ROI
        boxes_list = [boxes.float()]
        roi_features = self.roi_align(
            {"0": attended_features},
            boxes_list,
            [attended_features.shape[2:]]
        )
        roi_features = roi_features.view(roi_features.size(0), -1)
        return self.classifier(roi_features)

# 定義數據集類
class KeyDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_labels=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_labels = use_labels
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.use_labels:
            # 使用標註文件
            label_file = os.path.join(self.root_dir, 'labels', self.image_files[idx].rsplit('.', 1)[0] + '.txt')
            with open(label_file, 'r') as f:
                anno = f.read().strip().split()
            class_id, x_center, y_center, width, height = map(float, anno)
            # 轉換成 (x1, y1, x2, y2) 格式
            x1 = (x_center - width/2) * image.width
            y1 = (y_center - height/2) * image.height
            x2 = (x_center + width/2) * image.width
            y2 = (y_center + height/2) * image.height
            box = [x1, y1, x2, y2]
        else:
            # 使用整張圖片作為感興趣區域
            box = [0, 0, image.width, image.height]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(box)

# 定義對比損失函數
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        # 計算特徵之間的相似度矩陣
        similarity_matrix = torch.matmul(features, features.T)
        # 應用溫度縮放
        similarity_matrix = similarity_matrix / self.temperature
        # 創建標籤矩陣（對角線為正例，其他為負例）
        labels = torch.eye(features.shape[0], device=features.device)
        # 計算交叉熵損失
        loss = nn.functional.cross_entropy(similarity_matrix, labels)
        return loss

# 訓練模型的函數
def train_model(dataloader, model, criterion, optimizer, num_epochs=10, start_epoch=0, checkpoint_dir='checkpoints'):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    previous_loss = None  # 用於存儲前一個紀元的平均損失
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for images, boxes in dataloader:
            optimizer.zero_grad()
            features = model(images, boxes)
            loss = criterion(features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        
        # 計算減少的平均損失
        if previous_loss is not None:
            loss_reduction = previous_loss - avg_loss
        else:
            loss_reduction = 0.0
            
        print(f"紀元 {epoch+1}/{num_epochs}, 平均損失: {avg_loss:.6f}, 減少了 {loss_reduction:.6f}")
        
        # 更新前一個紀元的平均損失
        previous_loss = avg_loss
        
        # 保存檢查點
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, f'{checkpoint_dir}/best_model.pth')
        
        # 保存訓練歷史
        history = {
            'epoch': epoch + 1,
            'loss': avg_loss,
        }
        with open(f'{checkpoint_dir}/history.json', 'a') as f:
            json.dump(history, f)
            f.write('\n')
      
# 使用模型進行鑰匙頭檢測的函數
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
        features = model(image_tensor, boxes)
    
    # 這裡我們假設特徵向量的範數（magnitude）代表鑰匙頭的可能性
    confidence = torch.norm(features).item()
    return confidence > 0.5, confidence

# 使用OCR識別文字的函數
def recognize_text(image_path, lang='eng'):
    image = Image.open(image_path)
    # 設定Tesseract只識別英文字母和數字
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
    return text

# 訓練模型的主函數
def train(resume=False):
    # 設定訓練數據路徑
    train_data_dir = "data/train/H646"
    checkpoint_dir = 'checkpoints'
    
    # 創建數據集和數據加載器
    train_dataset = KeyDataset(train_data_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]), use_labels=True)  # 設置為False來使用自監督學習
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 初始化模型
    model = KeyHeadDetector()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_epoch = 0
    num_epochs = 10
    if resume:
        # 加載最新的檢查點
        ## lambda 判定位數，使得排序9<10
        checkpoints = sorted(Path(checkpoint_dir).glob('checkpoint_epoch_*.pth'),key=lambda x: int(x.stem.split('_')[-1]))
        
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            num_epochs = start_epoch + 10
            print(f"繼續從紀元 {start_epoch} 開始訓練")
        else:
            print("沒有找到檢查點，從頭開始訓練")
    
    # 訓練模型
    # num_epochs = 10
    train_model(train_loader, model, criterion, optimizer, num_epochs, start_epoch, checkpoint_dir)

# 使用模型進行測試的主函數
def test(use_best_model=True):
    checkpoint_dir = 'checkpoints'
    model = KeyHeadDetector()
    
    if use_best_model:
        checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
    else:
        # 使用最新的檢查點
        checkpoints = sorted(Path(checkpoint_dir).glob('checkpoint_epoch_*.pth'))
        checkpoint_path = checkpoints[-1] if checkpoints else None
    
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加載模型從: {checkpoint_path}")
    else:
        print("沒有找到模型檢查點，請先訓練模型")
        return

    model.eval()

    # 使用模型進行推理
    test_image_path = "data/test/S__24117254_0.jpg"
    is_key_head, confidence = detect_key_head(model, test_image_path)
    if is_key_head:
        print(f"偵測到鑰匙頭，信心度：{confidence:.2f}")
        text = recognize_text(test_image_path)
        print(f"識別到的英文和數字：{text}")
    else:
        print(f"未偵測到鑰匙頭，信心度：{confidence:.2f}")
        text = recognize_text(test_image_path)
        print(f"識別到的英文和數字：{text}")

if __name__ == "__main__":
    # 可以在這裡選擇執行訓練或測試
    # train(resume=True)  # 執行訓練，並從上次的檢查點繼續
    test(use_best_model=True)  # 執行測試，使用最佳模型
    pass



# 更新了 train_model 函數：

# 現在它接受 start_epoch 參數，允許從特定的紀元開始訓練。
# 每個紀元結束時，它會保存一個檢查點（checkpoint），包含模型狀態、優化器狀態和當前紀元。
# 它還會保存最佳模型（根據損失）。
# 訓練歷史被保存在一個 JSON 文件中，方便後續分析。


# 更新了 train 函數：

# 添加了 resume 參數，用於控制是否從之前的檢查點繼續訓練。
# 如果 resume=True，它會尋找最新的檢查點並加載。
# 如果沒有找到檢查點，它會從頭開始訓練。


# 更新了 test 函數：

# 添加了 use_best_model 參數，用於選擇使用最佳模型還是最新的檢查點。
# 如果沒有找到檢查點，它會提示用戶先訓練模型。



# 使用方法：

# 開始新的訓練：
# train(resume=False)

# 從上次的檢查點繼續訓練：
# train(resume=True)

# 使用最佳模型進行測試：
# test(use_best_model=True)

# 使用最新檢查點進行測試：
# test(use_best_model=False)


# 注意事項：

# 檢查點和模型會被保存在 'checkpoints' 目錄下。確保這個目錄存在或有權限創建。
# 訓練歷史被保存在 'checkpoints/history.json' 文件中，您可以使用這個文件來繪製損失曲線或進行其他分析。
# 每次訓練都會保存多個檢查點，可能會佔用較大的磁盤空間。您可能需要定期清理舊的檢查點。

#  1. `AttentionModule` 和 `KeyHeadDetector` 類：
#     - 這些類定義了模型的結構，包括注意力機制和特徵提取。
#     - `KeyHeadDetector` 使用預訓練的ResNet50作為骨幹網絡，並添加了注意力模塊和ROI Align層。
# 2. `KeyDataset` 類：
#     - 這個類處理數據加載。
#     - 通過 `use_labels` 參數，可以選擇是否使用標註數據。
#     - 如果使用標註，它會讀取相應的txt文件並將座標轉換為模型所需的格式。
# 3. `ContrastiveLoss` 類：
#     - 實現了對比損失函數，用於自監督學習。
#     - 它計算特徵之間的相似度，並使用交叉熵損失來優化模型。
# 4. `train_model` 函數：
#     - 實現了模型的訓練循環。
#     - 它遍歷數據集，計算損失，並更新模型參數。
# 5. `detect_key_head` 和 `recognize_text` 函數：
#     - 這些函數用於模型的推理階段。
#     - `detect_key_head` 使用訓練好的模型來檢測鑰匙頭。
#     - `recognize_text` 使用OCR來識別圖像中的文字。
# 6. `train` 和 `test` 函數：
#     - 這些是主函數，分別用於訓練和測試模型。
#     - 它們被分開定義，以便您可以單獨執行訓練或測試。

# 使用說明：

# 1. 要開始訓練，請設置正確的 `train_data_dir` 路徑，然後調用 `train()` 函數。
# 2. 訓練完成後，模型將被保存為 "key_head_detector.pth"。
# 3. 要進行測試，請設置正確的 `test_image_path`，然後調用 `test()` 函數。

# 注意事項：

# - 這個模型使用自監督學習，不需要明確的標籤。但如果您有標籤數據，可以將 `KeyDataset` 中的 `use_labels` 設為 True。
# - 自監督學習的效果可能需要更多的訓練數據和更長的訓練時間來達到最佳效果。
# - 您可能需要根據實際數據調整模型結構、損失函數和超參數。   
    
