import torch
import cv2
import os
import argparse
import yaml
import pytesseract
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

def train_model(data_yaml, weights='yolov5s.pt', epochs=100, batch_size=16, img_size=640, resume=False):
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    
    # 讀取並打印數據集信息
    try:
        with open(data_yaml, 'r', encoding='utf-8') as file:
            data_info = yaml.safe_load(file)
        print(f"數據集信息: {data_info}")
    except UnicodeDecodeError:
        print("無法以UTF-8編碼讀取data.yaml檔案。嘗試使用其他編碼...")
        encodings = ['utf-8-sig', 'gb18030', 'big5', 'gbk']
        for enc in encodings:
            try:
                with open(data_yaml, 'r', encoding=enc) as file:
                    data_info = yaml.safe_load(file)
                print(f"成功使用 {enc} 編碼讀取data.yaml檔案")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("無法讀取data.yaml檔案。請檢查檔案編碼。")

    # 開始訓練
    cmd = f"python train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data {data_yaml} --weights {weights} --device {device}"
    if resume:
        cmd += " --resume"
    
    os.system(cmd)
    print("訓練完成！")

def evaluate_model(weights, data_yaml, img_size=640):
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    
    # 載入模型
    model = attempt_load(weights, device=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)

    # 設置數據加載器
    dataset = LoadImages(data_yaml, img_size=imgsz, stride=stride)

    # 運行推理
    model.eval()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # 處理檢測結果
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    print(f"檢測到物體：類別 {int(cls)}, 置信度 {conf:.2f}")

    print("評估完成！")

def detect_and_ocr(weights, image_path, img_size=640, conf_thres=0.25, iou_thres=0.45):
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    
    # 載入模型
    model = attempt_load(weights, device=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)

    # 載入圖像
    img0 = cv2.imread(image_path)
    img = LoadImages(image_path, img_size=imgsz, stride=stride).extract_fn(img0)

    # 執行推理
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

    # 處理檢測結果
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=(0, 255, 0), line_thickness=3)
                
                # 如果是文字區域（標籤為2），執行OCR
                if int(cls) == 2:
                    x1, y1, x2, y2 = map(int, xyxy)
                    text_region = img0[y1:y2, x1:x2]
                    text = pytesseract.image_to_string(text_region, config='--psm 7')
                    cv2.putText(img0, f'OCR: {text.strip()}', (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 顯示結果
    cv2.imshow('Detection and OCR Result', img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # 實現畫框和標籤的函數
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv5 鑰匙檢測和OCR')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='初始權重路徑')
    parser.add_argument('--data', type=str, default='data.yaml', help='數據配置文件')
    parser.add_argument('--img-size', type=int, default=640, help='推理大小 (像素)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='物體置信度閾值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU 閾值')
    parser.add_argument('--resume', action='store_true', help='從上次的檢查點繼續訓練')
    parser.add_argument('--epochs', type=int, default=100, help='訓練輪數')
    parser.add_argument('--batch-size', type=int, default=16, help='總批次大小')
    parser.add_argument('--test-image', type=str, help='測試圖像的路徑')
    parser.add_argument('--evaluate', action='store_true', help='評估模型')
    args = parser.parse_args()

    if args.test_image:
        detect_and_ocr(args.weights, args.test_image, args.img_size, args.conf_thres, args.iou_thres)
    elif args.evaluate:
        evaluate_model(args.weights, args.data, args.img_size)
    else:
        train_model(args.data, args.weights, args.epochs, args.batch_size, args.img_size, args.resume)
        evaluate_model(args.weights, args.data, args.img_size)