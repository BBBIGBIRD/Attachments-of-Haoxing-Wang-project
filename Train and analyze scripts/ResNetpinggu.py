import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_model(model_path, num_classes=11):
    # 加载预训练的 ResNet-18 模型
    model = models.resnet18(pretrained=False)
    
    # 修改最后的全连接层以适应具体的类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(11), yticklabels=range(11))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # 定义数据预处理
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 数据集路径
    data_dir = 'C://Users//15097//Desktop//Spectograms_77_24_Xethrue//activity_spectogram_77GHz'  # 数据集路径
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 加载模型
    model_path = 'C://Users//15097//Desktop//训练过程结果//resnet18_classification.pth'  # 替换为您的.pth文件路径
    num_classes = 11  # 根据实际情况调整类别数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = load_model(model_path, num_classes)
    model = model.to(device)
    
    # 评估模型并输出混淆矩阵
    labels, preds = evaluate_model(model, data_loader, device)
    plot_confusion_matrix(labels, preds)

if __name__ == '__main__':
    main()
