import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


# 训练集 Dataset：从 train.csv 中加载图片 id 和标签
class AptosDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): 包含 id_code 与 diagnosis 两列的 CSV 文件（训练标签）。
            img_dir (str): 训练图像所在的文件夹，aptos2019-blindness-detection/train_images"
            transform (callable, optional): 图片预处理和数据增强方法。
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_id = row['id_code']
        label = int(row['diagnosis'])
        img_path = os.path.join(self.img_dir, img_id + ".png")
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class AptosTestDataset(Dataset):
    def __init__(self, img_dir, transform=None, csv_file=None):
        self.img_dir = img_dir
        self.transform = transform
        if csv_file is not None:
            data = pd.read_csv(csv_file)
            self.img_ids = data['id_code'].tolist()
        else:
            # 直接读取文件夹中所有文件（不带扩展名）
            self.img_ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if
                            os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_path = os.path.join(self.img_dir, img_id + ".png")
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_id


# 定义基础残差块（BasicBlock）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        return F.relu(out)


# 定义 ResNet-18 模型（输出类别数设置为 5，对应 0~4 级别）
class ResNet_18(nn.Module):
    def __init__(self, block, num_classes=5):
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        # 保持全连接层输入为512个特征
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 使用自适应平均池化，将输出空间尺寸调整到 1×1
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.fc(out)


if __name__ == '__main__':
    # --------------- 训练阶段 ---------------
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 训练集 CSV 文件和图像文件夹路径
    train_csv = 'aptos2019-blindness-detection/train.csv'
    train_img_dir = 'aptos2019-blindness-detection/train_images'
    train_dataset = AptosDataset(train_csv, train_img_dir, transform=train_transforms)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 创建模型、损失函数和优化器
    model = ResNet_18(BasicBlock, num_classes=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0
    print("训练结束！")

    # --------------- 测试阶段 ---------------
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    sample_csv = 'aptos2019-blindness-detection/sample_submission.csv'
    test_img_dir = 'aptos2019-blindness-detection/test_images'
    test_dataset = AptosTestDataset(test_img_dir, transform=test_transforms, csv_file=sample_csv)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    all_predictions = []
    all_img_ids = []
    with torch.no_grad():
        for imgs, img_ids in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            all_img_ids.extend(img_ids)

    submission = pd.DataFrame({'id_code': all_img_ids, 'diagnosis': all_predictions})
    submission.to_csv('submission.csv', index=False)
    print("测试预测完成，提交文件已保存为 submission.csv")
