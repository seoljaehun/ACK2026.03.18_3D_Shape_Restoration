########################################
#  Model Encoder                       #
# :ResNet18 (Feature Extraction)       #
########################################

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

#==========================================
# Pretrained ResNet18 Encoder
#==========================================

# 1. ResNet18 Encoder 정의
class Resnet18Encoder(nn.Module):
    def __init__(self, c_dim=256, pretrained=True):
        """
            사전 훈련된 ResNet18 모델 기반 Feature Encoder
        Args:
            c_dim: 출력 차원 크기
            pretrained (bool, optional): 사전 훈련된 가중치 사용 여부. Defaults to True.
        """
        super().__init__()
        
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
            
        # 사전 훈련된 가중치 로드 설정
        resnet = models.resnet18(weights=weights)
        
        self.features = nn.Module()
        # Layer 분리
        self.features.conv1 = resnet.conv1       # 224x224x3 -> 112x112x64
        self.features.bn1 = resnet.bn1
        self.features.relu = resnet.relu
        self.features.maxpool = resnet.maxpool   # 112x112x64 -> 56x56x64

        self.features.layer1 = resnet.layer1     # 56x56x64 -> 56x56x64
        self.features.layer2 = resnet.layer2     # 56x56x64 -> 28x28x128
        self.features.layer3 = resnet.layer3     # 28x28x128 -> 14x14x256
        self.features.layer4 = resnet.layer4     # 14x14x256 -> 7x7x512

        # 전역 평균 풀링 레이어
        self.avgpool = resnet.avgpool   # 7x7x512 -> 1x1x512
        
        # 차원 압축 
        self.fc = nn.Linear(512, c_dim) # 1x1x512 -> 1x1xc_dim
        
    def forward(self, x):
        """
        Args:
            x(입력): (B, 3, 224, 224)

        Returns:
            global_feat: (B, c_dim)
        """
        x = self.features.conv1(x)        # (B, 3, 224, 224) -> (B, 64, 112, 112)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)      # (B, 64, 112, 112) -> (B, 64, 56, 56)
        
        x1 = self.features.layer1(x)      # (B, 64, 56, 56) -> (B, 64, 56, 56) 
        x2 = self.features.layer2(x1)     # (B, 64, 56, 56) -> (B, 128, 28, 28)
        x3 = self.features.layer3(x2)     # (B, 128, 28, 28) -> (B, 256, 14, 14)
        x4 = self.features.layer4(x3)     # (B, 256, 14, 14) -> (B, 512, 7, 7)
        
        out = self.avgpool(x4)             # (B, 512, 7, 7) -> (B, 512, 1, 1)
        out = out.view(out.size(0), -1)    # (B, 512, 1, 1) -> (B, 512)
        global_feat = self.fc(out)         # (B, 512) -> (B, c_dim)
        
        return global_feat