import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm

def main():
    # 数据增强和预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 加载数据集
    data_dir = 'C://Users//15097//Desktop//Spectograms_77_24_Xethrue//activity_spectogram_77GHz'  # 替换为你的数据集路径
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The provided dataset directory {data_dir} does not exist.")

    dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])

    # 按 80/20 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 应用不同的 transform
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 使用预训练的 AlexNet 模型
    model = models.alexnet(pretrained=True)

    # 修改最后的全连接层以适应11个类别
    num_classes = 11  # 如果类别数不同，请修改此处
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 训练和验证循环
    num_epochs = 150  # 设置训练的 epoch 数

    for epoch in range(num_epochs):
        # 输出当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Current Learning Rate: {current_lr:.6f}")

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)

        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # 使用调度器调整学习率
        scheduler.step(val_epoch_loss)

    # 保存模型
    torch.save(model.state_dict(), 'alexnet_classification.pth')


if __name__ == '__main__':
    main()
