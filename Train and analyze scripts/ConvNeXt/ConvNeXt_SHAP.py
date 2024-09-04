import torch
import shap
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import convnext_tiny as create_model  # 确保从您的模型定义文件导入ConvNeXt模型

# 1. 加载模型
def load_model(pth_file, num_classes, device):
    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load(pth_file, map_location=device))
    model.eval().to(device)
    return model

# 2. 处理输入图片
def process_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

# 3. 可视化SHAP值
def show_shap_values(shap_values, image_tensor):
    # 提取第一个类别的SHAP值
    shap_values = shap_values[0][..., 0]  # 选择第一个类别的SHAP值
    shap_numpy = np.transpose(shap_values, (1, 2, 0))  # 转置为 (H, W, C)
    
    image_numpy = np.transpose(image_tensor.squeeze().cpu().numpy(), (1, 2, 0))  # 转置为 (H, W, C)

    # 转换为NumPy数组并传递给shap.image_plot
    shap.image_plot(np.array([shap_numpy]), np.array([image_numpy]))

# 4. 主函数
if __name__ == "__main__":
    # 配置路径和设备
    pth_file = 'C://Users//15097//Desktop//result_version//05_results//weights//best_model.pth'  # 模型文件路径
    image_path = 'C://Users//15097//Desktop//Spectograms_77_24_Xethrue//activity_spectogram_77GHz//05_Walking_towards_radar//04000000_1574696114_Raw_0.png'  # 输入图片路径
    num_classes = 11  # 模型类别数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载ConvNeXt模型
    model = load_model(pth_file, num_classes, device)

    # 处理输入图像
    image_tensor = process_image(image_path).to(device)

    # 选择背景数据，用于SHAP值计算 (可以选取多张代表性图片)
    background = torch.cat([process_image(image_path).to(device) for _ in range(10)], dim=0)

    # 使用SHAP DeepExplainer解释模型
    explainer = shap.DeepExplainer(model, background)

    # 计算SHAP值
    shap_values = explainer.shap_values(image_tensor)

    # 可视化SHAP值
    show_shap_values(shap_values, image_tensor)
