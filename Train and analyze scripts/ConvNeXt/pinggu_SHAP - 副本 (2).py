import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import shap  # 导入 SHAP

from my_dataset import MyDataSet
from model import convnext_tiny as create_model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # 获取每个类别的第一张图片及其标签
    all_images_path, all_images_label = read_first_image_per_class(args.data_path)

    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=all_images_path,
                            images_class=all_images_label,
                            transform=data_transform)

    batch_size = len(all_images_path)  # 处理的图片数量等于类别数
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=0,  # 降低复杂性，因为数据量很小
                            collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    # 加载训练好的权重
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))

    model.eval()  # 设置模型为评估模式

    # SHAP解释部分
    print("Calculating SHAP values for the first class...")
    images, labels = next(iter(val_loader))  # 获取数据加载器中的所有图片
    images = images.to(device)
    background = images  # 直接使用加载的图片作为背景数据
    explainer = shap.DeepExplainer(model, background)

    # 创建保存SHAP可视化结果的目录
    save_dir = "C://Users//15097//Desktop//SHAP_result"  # 修改为你想保存的路径
    os.makedirs(save_dir, exist_ok=True)

    # 计算SHAP值并进行可视化
    shap_values = explainer.shap_values(images, check_additivity=False)
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

def read_first_image_per_class(data_path):
    classes = [cls for cls in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cls))]
    all_images_path = []
    all_images_label = []

    for cls in classes:
        cls_folder = os.path.join(data_path, cls)
        img_name = os.listdir(cls_folder)[0]  # 仅获取第一个图片
        img_path = os.path.join(cls_folder, img_name)
        all_images_path.append(img_path)
        all_images_label.append(classes.index(cls))

    return all_images_path, all_images_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=11)
    
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="C://Users//15097//Desktop//Spectograms_77_24_Xethrue//activity_spectogram_77GHz")
    
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='C://Users//15097//Desktop//result_version//05_results//weights//best_model.pth', help='initial weights path')

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
