import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义一个钩子函数来获取中间层的特征图和梯度
class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None
        self.gradients = None

    def hook_fn(self, module, input, output):
        self.features = output
        output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def close(self):
        self.hook.remove()

# 加载VGG-16模型
def load_vgg16_model(pth_file, num_classes, device):
    model = models.vgg16()
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)  # 修改输出类别
    model.load_state_dict(torch.load(pth_file, map_location=device))
    model.eval().to(device)
    return model

# 生成Grad-CAM的热力图
def generate_heatmap(model, img_tensor, target_layer):
    # 获取目标层的特征图和梯度
    activation = SaveFeatures(target_layer)

    output = model(img_tensor)
    pred = torch.argmax(output, dim=1).item()
    pred_class = output[:, pred]

    model.zero_grad()
    pred_class.backward()

    # 获取梯度和激活特征图
    gradients = activation.gradients[0].cpu().data.numpy()
    activations = activation.features[0].cpu().data.numpy()

    # 权重计算
    weights = np.mean(gradients, axis=(1, 2))

    # 加权和生成热力图
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_tensor.shape[2], img_tensor.shape[3]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

# 加载和处理输入图片
def process_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

# 绘制并保存热力图
def show_and_save_heatmap(heatmap, img_path, save_path='heatmap.jpg'):
    img = cv2.imread(img_path)
    
    # 调整热力图大小以匹配原始图片的大小
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    cam_img = heatmap + np.float32(img) / 255
    cam_img = cam_img / np.max(cam_img)
    
    cv2.imwrite(save_path, np.uint8(255 * cam_img))
    plt.imshow(cam_img)
    plt.show()

if __name__ == "__main__":
    pth_file = 'C://Users//15097//Desktop//训练过程结果//vgg16_classification.pth'  # 替换为你的VGG-16模型文件路径
    image_path = 'C://Users//15097//Desktop//Spectograms_77_24_Xethrue//activity_spectogram_77GHz//10_kneeling//04100010_1573838526_Raw_0.png'  # 输入图片路径
    save_path = 'vgg16_heatmap.jpg'  # 热力图保存路径
    num_classes = 11  # 修改为模型的输出类别数量
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载VGG-16模型和图片
    model = load_vgg16_model(pth_file, num_classes, device)
    img_tensor = process_image(image_path).to(device)

    # 选择VGG-16的最后一层卷积层
    target_layer = model.features[0]

    # 生成热力图
    heatmap = generate_heatmap(model, img_tensor, target_layer)

    # 显示并保存热力图
    show_and_save_heatmap(heatmap, image_path, save_path)
