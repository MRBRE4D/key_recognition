import torch
import cv2
import os
import yaml
import pytesseract
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device


# 設置Tesseract OCR的路徑
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 設置參數
WEIGHTS = 'yolov5s.pt'  # 初始權重路徑
DATA_YAML = 'data.yaml'  # 數據配置文件
IMG_SIZE = 640  # 推理大小 (像素)
CONF_THRES = 0.25  # 物體置信度閾值
IOU_THRES = 0.45  # NMS IOU 閾值
RESUME = False  # 是否從上次的檢查點繼續訓練
EPOCHS = 10  # 訓練輪數
BATCH_SIZE = 16  # 總批次大小
TEST_IMAGE = None  # 測試圖像的路徑，如果要測試單張圖片，請在此設置路徑
DEVICE = ''  # 設備選擇，留空為自動選擇

def train_model(data_yaml=DATA_YAML, weights=WEIGHTS, epochs=EPOCHS, batch_size=BATCH_SIZE, img_size=IMG_SIZE, resume=RESUME):
    device = select_device(DEVICE)
    print(f"使用設備: {device}")
    
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

    
    # 訓練結束後顯示標註結果
    show_training_results()

def show_training_results(num_images=5):
    print("顯示訓練結果...")
    for i in range(num_images):
        img_path = f'runs/train/exp/train_batch{i}_labels.jpg'
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            cv2.imshow(f'Training Image {i}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"找不到圖片: {img_path}")
    print("訓練結果顯示完成。")
    
    
def evaluate_model(weights=WEIGHTS, data_yaml=DATA_YAML, img_size=IMG_SIZE):
    device = select_device(DEVICE)
    print(f"使用設備: {device}")
    
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

        with torch.no_grad():
            pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=None, agnostic=False)

        # 處理檢測結果
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    print(f"檢測到物體：類別 {int(cls)}, 置信度 {conf:.2f}")

    print("評估完成！")

def detect_and_ocr(weights=WEIGHTS, image_path=TEST_IMAGE, img_size=IMG_SIZE):
    if image_path is None:
        print("請設置 TEST_IMAGE 變數為要測試的圖片路徑。")
        return

    device = select_device(DEVICE)
    print(f"使用設備: {device}")
    
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

    with torch.no_grad():
        pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=None, agnostic=False)

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
    # 根據需要取消註釋並運行相應的功能
    train_model()
    # evaluate_model()
    # detect_and_ocr()
    pass