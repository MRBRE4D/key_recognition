from ultralytics import YOLO
import cv2
import os
import argparse
import yaml
import torch
import pytesseract

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
    # device='cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
    
    # 使用更穩定的方法獲取評估指標
    metrics = results.results_dict
    
    # 打印所有可用的指標
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
            print(f"{metric}: {value[0]:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # 特別關注 mAP 指標
    if 'metrics/mAP50(B)' in metrics:
        print(f"mAP50: {metrics['metrics/mAP50(B)']:.4f}")
    if 'metrics/mAP50-95(B)' in metrics:
        print(f"mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")

def detect_and_ocr(model_path, image_path):
    # 載入訓練好的模型
    model = YOLO(model_path)

    # 讀取圖像
    img = cv2.imread(image_path)
    
    # 執行檢測
    results = model(img)

    # 處理檢測結果
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 獲取邊界框座標
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # 獲取類別和置信度
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 在圖像上繪製邊界框和標籤
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'Class: {cls}, Conf: {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 如果是文字區域（標籤為2），執行OCR
            if cls == 2:
                # 裁剪文字區域
                text_region = img[int(y1):int(y2), int(x1):int(x2)]
                
                # 執行OCR
                text = pytesseract.image_to_string(text_region, config='--psm 7')
                
                # 在圖像上顯示辨識出的文字
                cv2.putText(img, f'OCR: {text.strip()}', (int(x1), int(y2) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 顯示結果
    cv2.imshow('Detection and OCR Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='訓練鑰匙檢測模型')
    parser.add_argument('--resume', action='store_true', help='從上次的檢查點繼續訓練')
    parser.add_argument('--imgsz', type=int, default=640, help='輸入圖像大小')
    parser.add_argument('--batch', type=int, help='批次大小')
    parser.add_argument('--test_image', type=str, help='測試圖像的路徑')
    parser.add_argument('--evaluate', action='store_true', help='評估模型')
    args = parser.parse_args()

    if args.test_image:
        detect_and_ocr('runs/detect/key_detection/weights/best.pt', args.test_image)
    elif args.evaluate:
        evaluate_model('runs/detect/key_detection/weights/best.pt')
    else:
        train_model(resume=args.resume, imgsz=args.imgsz, batch=args.batch)
        evaluate_model('runs/detect/key_detection/weights/best.pt')