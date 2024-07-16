from ultralytics import YOLO
import cv2
import os
import argparse
import yaml
import torch


def train_model(resume=False, imgsz=1280, batch=None):
    """ 訓練鑰匙檢測模型，支持斷點續訓和靈活的參數設置 """
    # 初始化 YOLO 模型
    if resume:
        model = YOLO('runs/detect/key_detection/weights/last.pt')
        print("正在從上次的檢查點繼續訓練...")
    else:
        model = YOLO('yolov8l.pt')
        print("正在開始新的訓練...")

    # 讀取並打印數據集信息
    try:
        with open('data.yaml', 'r', encoding='utf-8') as file:
            data_info = yaml.safe_load(file)
        print(f"數據集信息: {data_info}")
    except UnicodeDecodeError:
        print("無法以UTF-8編碼讀取data.yaml檔案。嘗試使用其他編碼...")
        encodings = ['utf-8-sig', 'gb18030', 'big5', 'gbk']
        for enc in encodings:
            try:
                with open('data.yaml', 'r', encoding=enc) as file:
                    data_info = yaml.safe_load(file)
                print(f"成功使用 {enc} 編碼讀取data.yaml檔案")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("無法讀取data.yaml檔案。請檢查檔案編碼。")

    # 驗證數據集路徑並打印詳細信息
    base_path = data_info.get('path', '')
    for split in ['train', 'val', 'test']:
        images_path = os.path.join(base_path, data_info.get(split, ''))
        labels_path = images_path.replace('images', 'labels')

        if os.path.exists(images_path):
            image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f"{split} 圖像數量: {len(image_files)}")
        else:
            print(f"警告: {split} 圖像路徑不存在: {images_path}")

        if os.path.exists(labels_path):
            label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
            print(f"{split} 標籤數量: {len(label_files)}")
        else:
            print(f"警告: {split} 標籤路徑不存在: {labels_path}")

        # 檢查圖像和標籤數量是否匹配
        if os.path.exists(images_path) and os.path.exists(labels_path):
            if len(image_files) != len(label_files):
                print(f"警告: {split} 集的圖像數量 ({len(image_files)}) 與標籤數量 ({len(label_files)}) 不匹配")
            
            # 檢查每個圖像是否有對應的標籤文件
            for img_file in image_files:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                if label_file not in label_files:
                    print(f"警告: {split} 集中的圖像 {img_file} 沒有對應的標籤文件")

    # 檢查 CUDA 是否可用
    device='cpu'
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")

    # 準備訓練參數
    train_args = {
        'data': 'data.yaml',
        'epochs': 5,
        'name': 'key_detection',
        'plots': True,
        'resume': resume,
        'save_period': 10,
        'imgsz': imgsz,  # 使用較大的圖像尺寸
        'patience': 50,  # 提前停止的耐心值
        'device': device,  # 根據可用性使用 GPU 或 CPU
        'rect': True,  # 使用矩形訓練
        'mosaic': 0.0,  # 禁用 mosaic 增強
        'close_mosaic': 0,  # 禁用 mosaic 增強
        'augment': True,  # 使用默認增強
        'copy_paste': 0.0,  # 禁用 copy_paste 增強
        'degrees': 0.0,  # 禁用旋轉增強
    }
    if batch is not None:
        train_args['batch'] = batch

    # 開始訓練
    try:
        results = model.train(**train_args)
        print("訓練完成！")
    except Exception as e:
        print(f"訓練過程中出現錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

    # 在訓練結束後顯示一些標註結果
    for i in range(5):
        img_path = f'runs/detect/key_detection/train_batch{i}_labels.jpg'
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            cv2.imshow(f'Training Image {i}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            


def evaluate_model(model_path):
    model = YOLO(model_path)
    results = model.val()
    print("\n最終模型評估結果:")
    try:
        print(f"mAP50: {results.maps[50]:.4f}")
        print(f"mAP50-95: {results.maps[0]:.4f}")
        
    except IndexError:
        print("mAP50: 数据不可用")
        print("mAP50-95: 数据不可用")
        
        
    
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='訓練鑰匙檢測模型')
    parser.add_argument('--resume', action='store_true', help='從上次的檢查點繼續訓練')
    parser.add_argument('--imgsz', type=int, default=640, help='輸入圖像大小')
    parser.add_argument('--batch', type=int, help='批次大小')
    args = parser.parse_args()

    train_model(resume=args.resume, imgsz=args.imgsz, batch=args.batch)
    evaluate_model('runs/detect/key_detection/weights/best.pt')


# 1. train.py
# 使用方法：
#     - 主要功能：訓練鑰匙檢測模型。
#     - 關鍵特性：
#     a) 支持斷點續訓（使用 `-resume` 參數）。
#     b) 靈活的輸入圖像大小設置（使用 `-imgsz` 參數）。
#     c) 可調整的批次大小（使用 `-batch` 參數）。
#     - 訓練過程：
#     a) 初始化 YOLO 模型（新訓練或從檢查點恢復）。
#     b) 設置訓練參數，包括數據路徑、訓練輪數等。
#     c) 執行訓練。
#     d) 訓練結束後顯示部分標註結果。
    
#     ```
#     Copy
#     python train.py [--resume] [--imgsz SIZE] [--batch BATCH_SIZE]
    
#     ```
    
# 2. test.py：

# 使用方法：

# （注意：請先在代碼中設置正確的 `model_path` 和 `test_image_folder`）
#     - 主要功能：使用訓練好的模型進行測試。
#     - 關鍵步驟：
#     a) 載入訓練好的模型。
#     b) 對指定資料夾中的所有圖片進行預測。
#     c) 顯示每張圖片的預測結果，包括邊界框和標籤。
    
#     ```
#     Copy
#     python test.py
    
#     ```
    
# 3. data.yaml：
#     - 功能：配置數據集信息。
#     - 包含內容：
#     a) 數據集路徑。
#     b) 訓練集和驗證集的相對路徑。
#     c) 類別數量和名稱。

# 使用步驟：

# 1. 準備數據集：
#     - 使用 labelimg 工具標註您的圖片。
#     - 確保標註文件（.txt）位於相應的 `labels` 資料夾中。
#     - 按照範例創建並配置 `data.yaml` 文件。
# 2. 訓練模型：
#     - 運行 `train.py`，可以選擇是否使用 `-resume`、`-imgsz` 和 `-batch` 參數。
#     - 訓練過程和結果將保存在 `runs/detect/key_detection/` 目錄下。
# 3. 測試模型：
#     - 在 `test.py` 中設置正確的模型路徑和測試圖片資料夾。
#     - 運行 `test.py` 進行測試。

# 注意事項：

# - 確保所有路徑設置正確，包括 `data.yaml` 中的路徑和程式碼中的路徑。
# - 第一次訓練後，檢查點會自動保存。之後可以使用 `-resume` 參數從檢查點繼續訓練。
# - 可以根據您的硬體資源和具體需求調整 `imgsz` 和 `batch` 參數。


