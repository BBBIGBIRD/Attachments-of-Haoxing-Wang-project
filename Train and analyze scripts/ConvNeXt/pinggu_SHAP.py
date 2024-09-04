import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap  # 导入 SHAP

from my_dataset import MyDataSet
from model import convnext_tiny as create_model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # 获取所有数据路径和标签
    all_images_path, all_images_label = read_all_data(args.data_path)

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

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=nw,
                            collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    # 加载训练好的权重
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))

    model.eval()  # 设置模型为评估模式

    # 评估模型
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算整体准确率
    overall_accuracy = accuracy_score(all_labels, all_preds)
    print(f'Overall Validation Accuracy: {overall_accuracy * 100:.2f}%')

    # 计算每个类别的准确率
    class_accuracies = {}
    for class_idx in range(args.num_classes):
        class_mask = all_labels == class_idx
        class_accuracy = accuracy_score(all_labels[class_mask], all_preds[class_mask])
        class_accuracies[class_idx] = class_accuracy
        print(f'Accuracy of class {class_idx}: {class_accuracy * 100:.2f}%')

    # 计算并输出混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(args.num_classes), yticklabels=range(args.num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # SHAP解释部分
    print("Calculating SHAP values for the first class...")
    background = next(iter(val_loader))[0][:100].to(device)  # 选择一小部分数据作为背景数据
    explainer = shap.DeepExplainer(model, background)

    # 创建保存SHAP可视化结果的目录
    save_dir = "C://Users//15097//Desktop//SHAP_result"  # 修改为你想保存的路径
    os.makedirs(save_dir, exist_ok=True)

    # 遍历所有验证数据进行SHAP值计算和可视化（仅针对第一个类别）
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        shap_values = explainer.shap_values(images, check_additivity=False)

        print(f"shap_values shape: {np.array(shap_values).shape}")  # 输出SHAP值的形状以调试

        for j in range(images.shape[0]):
            if labels[j] == 0:  # 只处理属于第一个类别的样本
                if len(shap_values) > 0 and len(shap_values[0]) > j:
                    # 转换图像数据的形状以适应 image_plot 的输入要求
                    test_image = images[j].permute(1, 2, 0).cpu().numpy()

                    # 使用第一个类别的 SHAP 值
                    shap_value_for_first_class = shap_values[0][j]

                    # 保存可视化结果
                    plt.figure()
                    shap.image_plot(shap_values=[shap_value_for_first_class], pixel_values=test_image, show=False)
                    plt.savefig(os.path.join(save_dir, f"shap_image_class0_{i*batch_size + j}.png"))
                    plt.close()
                else:
                    print(f"Skipping sample {i*batch_size + j} due to insufficient SHAP values.")

        print(f"Processed batch {i+1}/{len(val_loader)}")

    print(f"SHAP visualizations for the first class saved to {save_dir}")

def read_all_data(data_path):
    classes = [cls for cls in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cls))]
    all_images_path = []
    all_images_label = []

    for cls in classes:
        cls_folder = os.path.join(data_path, cls)
        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            all_images_path.append(img_path)
            all_images_label.append(classes.index(cls))

    return all_images_path, all_images_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--batch-size', type=int, default=16)
    
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="C://Users//15097//Desktop//Spectograms_77_24_Xethrue//activity_spectogram_77GHz")
    
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='C://Users//15097//Desktop//result_version//05_results//weights//best_model.pth', help='initial weights path')

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
