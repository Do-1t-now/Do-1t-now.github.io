---
title: "基于深度卷积神经网络——结直肠癌——病理图像分类"
excerpt: "本项目基于PyTorch框架，使用ResNet50迁移学习模型对NCT-CRC-HE-100K结直肠癌病理图像数据集进行9类组织分类，通过混淆矩阵、ROC曲线和Grad-CAM热力图实现模型性能评估与可解释性分析。"
collection: portfolio
---
---
title： 
collection：portfolio
type："Computer Vision & Medical Imaging"
permalink： /portfolio/pathology-tissue-classification
date：2026-01-17 
excerpt： "本项目基于PyTorch框架，使用ResNet50迁移学习模型对NCT-CRC-HE-100K结直肠癌病理图像数据集进行9类组织分类，通过混淆矩阵、ROC曲线和Grad-CAM热力图实现模型性能评估与可解释性分析。"
tags：
-ResNet50
-NCT-CRC-HE-100K
-Deep Learning 
-病理图像分类 
-Grad-CAM
tech_stack：
-name： Python
-name： PyTorch
-name： ResNet50
-name： Numpy
-name： Pandas
-name： Sklearn
-name：matplotlib
-name：Seaborn
---
## 项目概述
本项目基于PyTorch框架，采用**ResNet50**迁移学习模型对**NCT-CRC-HE-100K**结直肠癌病理图像数据集进行9类组织分类，通过混淆矩阵、ROC曲线和Grad-CAM热力图实现模型性能评估与可解释性分析，为病理诊断提供客观量化的辅助支持。

## 数据集详情
### 数据集介绍
本研究使用的**NCT-CRC-HE-100K**数据集包含100,000张512×512像素的苏木精-伊红（H&E）染色病理切片图像，涵盖9类结直肠癌相关组织类型：
| 组织类型 | 中文名称       | 样本数量 |
|----------|----------------|----------|
| ADI      | 脂肪组织       | 10,000   |
| BACK     | 背景组织       | 10,000   |
| DEB      | 退变坏死组织   | 10,000   |
| LYM      | 淋巴细胞浸润   | 10,000   |
| MUC      | 粘液分泌组织   | 10,000   |
| NORM     | 正常黏膜组织   | 10,000   |
| STRO     | 结缔组织       | 10,000   |
| TUM      | 肿瘤组织       | 20,000   |
| UNK      | 未分类组织     | 10,000   |

### 数据增强策略
为提升模型泛化能力，采用以下数据增强策略：
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪并 resize 到224×224
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色扰动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 模型与训练
### 模型架构
采用**ResNet50**预训练模型进行迁移学习，冻结底层卷积层以保留通用图像特征，替换顶层全连接层为9类输出：
```python
import torch
import torch.nn as nn
from torchvision import models

# 加载预训练ResNet50模型
model = models.resnet50(pretrained=True)

# 冻结卷积层参数
for param in model.parameters():
    param.requires_grad = False

# 替换全连接层为9类输出
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 9)
```

### 训练参数
- **优化器**：Adam（仅训练全连接层参数）
- **学习率**：1e-4
- **批量大小**：128
- **训练轮数**：3
- **损失函数**：交叉熵损失（CrossEntropyLoss）
- **设备**：GPU加速（CUDA）

### 训练代码
```python
import torch.optim as optim
from torch.utils.data import DataLoader

# 数据加载（Dataset定义省略）
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(50):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{50}, Loss: {epoch_loss:.4f}')
    
    # 验证阶段（代码省略）
```

## 实验结果
### 混淆矩阵
模型在验证集上的混淆矩阵如下，总体准确率达到99.7%：
![混淆矩阵](/assets/images/portfolio/confusion_matrix.png)

### ROC曲线与AUC值
各类别的ROC曲线下面积（AUC）均超过0.99，平均AUC为0.99，表明模型具有良好的分类性能：
![ROC曲线](/assets/images/portfolio/roc_curve.png)

### Grad-CAM可解释性分析
通过Grad-CAM热力图可视化模型关注的病理区域，红色高亮区域表示模型在分类时最关注的区域。如果红色区域集中在细胞核密集处（对于LYM/TUM）或特定的纹理结构（对于MUS/STR），说明模型学到了正确的病理形态学特征，而非学习了背景噪声。
![Grad-CAM热力图](/assets/images/portfolio/grad_cam.png)


