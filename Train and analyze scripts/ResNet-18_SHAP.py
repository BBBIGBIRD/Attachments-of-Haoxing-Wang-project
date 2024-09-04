import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import numpy as np
import matplotlib.pyplot as plt
import shap

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # 数据转换
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(img_size),  # 仅调整大小到224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(root=args.data_path, transform=data_transform)

    # 获取每个类别的第一张图片及其标签
    class_indices = {}
    all_images_path = []
    all_images_label = []

    for img_path, label in dataset.samples:
        if label not in class_indices:
            class_indices[label] = img_path
            all_images_path.append(img_path)
            all_images_label.append(label)

    # 加载这些图片
    images = [data_transform(dataset.loader(img_path)).unsqueeze(0) for img_path in all_images_path]
    images = torch.cat(images).to(device)
    labels = torch.tensor(all_images_label).to(device)

    # 加载ResNet-18模型
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)  # 修改最后的全连接层
    model = model.to(device)

    # 加载训练好的权重
    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.eval()  # 设置模型为评估模式

    # 使用 SHAP 的 GradientExplainer 解释模型
    print("Calculating SHAP values for the first class...")
    background = images  # 直接使用加载的图片作为背景数据
    explainer = shap.GradientExplainer(model, background)

    # 创建保存SHAP可视化结果的目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 计算SHAP值并进行可视化
    shap_values = explainer.shap_values(images)
    print(f"shap_values shape: {np.array(shap_values).shape}")  # 输出SHAP值的形状以调试

    for j in range(images.shape[0]):
        # 转换图像数据的形状以适应 image_plot 的输入要求
        test_image = images[j].permute(1, 2, 0).cpu().numpy()

        # 使用该类别的 SHAP 值
        shap_value_for_class = shap_values[labels[j]][j]

        # 保存可视化结果
        plt.figure()
        shap.image_plot(shap_values=[shap_value_for_class], pixel_values=test_image, show=False)
        plt.savefig(os.path.join(save_dir, f"shap_image_class{labels[j]}.png"), dpi=300)  # 提高分辨率
        plt.close()

    print(f"SHAP visualizations for the first image of each class saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--data-path', type=str, default="C://Users//15097//Desktop//Spectograms_77_24_Xethrue//activity_spectogram_77GHz")
    parser.add_argument('--weights', type=str, default='C://Users//15097//Desktop//训练过程结果//resnet18_classification.pth', help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--save_dir', type=str, default='C://Users//15097//Desktop//SHAP_result', help='Directory to save SHAP visualizations')
    
    opt = parser.parse_args()
    main(opt)
