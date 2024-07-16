import torch
print(torch.cuda.is_available())  # 應該返回 True
print(torch.cuda.device_count())  # 顯示可用的 GPU 數量
print(torch.cuda.get_device_name(0))  # 顯示 GPU 型號