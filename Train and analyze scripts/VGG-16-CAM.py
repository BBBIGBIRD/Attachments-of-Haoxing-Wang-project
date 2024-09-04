import torch
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定义Grad-CAM类
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax().item()

        self.model.zero_grad()
        output[0, target_class].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        cam = torch.mean(self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))

        return cam

# 加载模型
def load_model(model_path):
    model = models.vgg16(pretrained=False)
    # 修改分类器以匹配你训练的模型
    model.classifier[6] = torch.nn.Linear(4096, 11)  # 11 是你训练时的分类数
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 预处理输入图像
def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

# 显示热力图
def show_cam_on_image(img_path, cam):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + np.float32(img) / 255
    cam_img = cam_img / np.max(cam_img)
    cam_img = np.uint8(255 * cam_img)
    cv2.imshow('Grad-CAM', cam_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主函数
def main():
    model_path = 'C://Users//15097//Desktop//训练过程结果//vgg16_classification.pth'  # 这里替换为你的模型文件路径
    img_path = 'C://Users//15097//Desktop//Spectograms_77_24_Xethrue//activity_spectogram_77GHz//05_Walking_towards_radar//04050030_1574193843_Raw_0.png'  # 这里替换为你要输入的图片路径

    model = load_model(model_path)
    target_layer = model.features[-1]  # 选择VGG-16的最后一层卷积层
    grad_cam = GradCAM(model, target_layer)

    img_tensor = preprocess_image(img_path)
    cam = grad_cam.generate_cam(img_tensor)

    show_cam_on_image(img_path, cam)

if __name__ == '__main__':
    main()
