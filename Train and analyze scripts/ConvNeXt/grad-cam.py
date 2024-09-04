import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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
        transforms.CenterCrop(img_size),
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
    
    # 创建Grad-CAM对象
    target_layers = [model.features[-1]]  # 假设最后一个卷积层是features中的最后一层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    
    output_heatmap_dir = args.heatmap_output_dir
    if not os.path.exists(output_heatmap_dir):
        os.makedirs(output_heatmap_dir)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 生成Grad-CAM热力图并保存
            grayscale_cam = cam(input_tensor=images, targets=[ClassifierOutputTarget(pred.item()) for pred in preds])
            for j in range(images.size(0)):
                img = images[j].cpu().numpy().transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())
                heatmap = show_cam_on_image(img, grayscale_cam[j], use_rgb=True)
                plt.imsave(os.path.join(output_heatmap_dir, f"heatmap_{i*batch_size + j}.png"), heatmap)
    
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
    parser.add_argument('--weights', type=str, default='E://BaiduNetdiskDownload//torch_convnext//16batch_150epochs_0.001lr_0.4valid.pth', help='initial weights path')
    
    # 保存热力图的目录
    parser.add_argument('--heatmap-output-dir', type=str, default='C://Users//15097//Desktop//reli', help='output directory for heatmaps')

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
