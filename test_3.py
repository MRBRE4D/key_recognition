from ultralytics import YOLO
import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
import json

def plot_with_labels(image, results):
    """
    使用 OpenCV 在给定影像上繪製詳細的檢測結果和標籤。
    """
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, b)  # 将坐标转换为整数
            
            # 繪製邊界框
            color = (0, 255, 0)  # 綠色
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 準備標籤文字
            class_name = r.names[int(c)]
            label = f"{class_name}: {conf:.2f}"
            
            # 獲取文字大小
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # 繪製標籤背景
            cv2.rectangle(image, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
            
            # 繪製標籤文字
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # 添加更多詳細信息
            info = f"Class: {class_name}\nConf: {conf:.2f}\nBox: ({x1},{y1}),({x2},{y2})"
            y_offset = y2 + 20
            for i, line in enumerate(info.split('\n')):
                cv2.putText(image, line, (x1, y_offset + i*20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

def process_results(results, class_names):
    """
    處理檢測結果並返回詳細資訊。
    """
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].tolist()  # get box coordinates in (top, left, bottom, right) format
            c = int(box.cls)
            conf = float(box.conf[0])
            detections.append({
                'class': class_names[c],
                'confidence': conf,
                'bbox': b
            })
    return detections

def test_model(model_path, test_image_folder, output_dir, conf_threshold=0.25):
    """
    測試鑰匙檢測模型並儲存帶有詳細標籤的結果。
    """
    # 載入訓練好的模型
    try:
        model = YOLO(model_path)
        print(f"成功載入模型: {model_path}")
        print(f"模型類別: {model.names}")
    except Exception as e:
        print(f"載入模型時發生錯誤: {str(e)}")
        return

    # 取得測試影像
    image_files = [f for f in os.listdir(test_image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print(f"在 {test_image_folder} 中沒有找到圖片文件")
        return
    print(f"找到 {len(image_files)} 張圖片")

    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)

    # 準備結果摘要
    results_summary = {}

    # 處理測試影像
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(test_image_folder, image_file)
        
        # 進行預測
        try:
            results = model(image_path, conf=conf_threshold)
            print(f"\n處理圖片: {image_file}")
            print(f"原始結果: {results}")
        except Exception as e:
            print(f"處理 {image_file} 時發生錯誤: {str(e)}")
            continue

        # 讀取原始影像
        image = cv2.imread(image_path)
        if image is None:
            print(f"無法讀取圖片: {image_path}")
            continue

        # 繪製結果
        image_with_labels = plot_with_labels(image.copy(), results)

        # 儲存結果
        output_path = os.path.join(output_dir, f'result_{image_file}')
        cv2.imwrite(output_path, image_with_labels)

        # 處理檢測結果
        detections = process_results(results, model.names)
        print(f"處理後的檢測結果: {detections}")
        
        # 儲存結果摘要
        results_summary[image_file] = {
            'detections': detections,
            'total_objects': len(detections)
        }

    # 將結果摘要保存為 JSON 文件
    with open(os.path.join(output_dir, 'results_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=4)

    print(f"所有結果已保存到 {output_dir}")
    print(f"結果摘要已保存到 {os.path.join(output_dir, 'results_summary.json')}")

if __name__ == '__main__':
    # 預設參數
    DEFAULT_MODEL_PATH = 'runs/detect/key_detection/weights/best.pt'
    DEFAULT_TEST_FOLDER = 'dataset/keys/test/images'
    DEFAULT_OUTPUT_DIR = 'test_results'
    DEFAULT_CONF_THRESHOLD = 0.1

    parser = argparse.ArgumentParser(description='測試鑰匙檢測模型')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='訓練好的模型路徑')
    parser.add_argument('--test_folder', type=str, default=DEFAULT_TEST_FOLDER, help='測試影像資料夾路徑')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR, help='輸出結果的資料夾路徑')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF_THRESHOLD, help='置信度閾值')

    args = parser.parse_args()

    print(f"使用以下參數運行測試：")
    print(f"模型路徑: {args.model}")
    print(f"測試資料夾: {args.test_folder}")
    print(f"輸出資料夾: {args.output}")
    print(f"置信度閾值: {args.conf}")

    test_model(args.model, args.test_folder, args.output, args.conf)