import os
import argparse

import torch
from torchvision import transforms
from my_dataset import MyDataSet
from model import convnext_tiny as create_model
from utils import read_split_data, evaluate


def test_model(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    _, _, test_images_path, test_images_label = read_split_data(args.data_path, val_rate=1.0)
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    model.eval()
    class_correct = [0] * args.num_classes
    class_total = [0] * args.num_classes
    test_times = 0
    with torch.no_grad():
        for images, labels in test_loader:
            test_times += 1
            images, labels = images.to(device), labels.to(device)
            outputs = torch.squeeze(model(images))
            _, predicted = torch.max(outputs, 1)
            # predict class
            # output = torch.squeeze(model(images.to(device))).cpu()
            # predict = torch.softmax(output, dim=0)
            # predicted = torch.argmax(predict).numpy()
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(args.num_classes):
        print(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.5f}%')
    print(f'Accuracy of the network on the {test_times * batch_size} test images: ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-path', type=str, default="C://Users//15097//Desktop//Spectograms_77_24_Xethrue//activity_spectogram_77GHz")
    parser.add_argument('--weights', type=str, default='C://Users//15097//Desktop//result_version//05_results//weights//best_model.pth', help='path to weights file')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    test_model(opt)
