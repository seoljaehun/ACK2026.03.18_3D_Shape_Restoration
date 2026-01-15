########################################
#  Model Encoder                       #
# :ResNet34 Layer3 + Skip Connections  #
########################################

import torch.nn as nn
from torchvision import models

#==========================================
# Pretrained ResNet34 Encoder
#==========================================

# 1. ResNet34 Encoder 정의
class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained=True):
        """
            사전 훈련된 ResNet34 모델 및 가중치 불러오기
        Args:
            pretrained (bool, optional): 사전 훈련된 가중치 사용 여부. Defaults to True.
        """
        super(ResNet34Encoder, self).__init__()
        
        # ResNet34 로드
        original_model = models.resnet34(pretrained=pretrained)
        
        # Layer별 분리 (Feature Extraction)
        self.layer0 = nn.Sequential(
            original_model.conv1,   # conv1: 224x224x3 -> 112x112x64
            original_model.bn1,     # bn1, relu: 정규화 및 활성화 함수
            original_model.relu,
            original_model.maxpool  # maxpool: 112x112x64 -> 56x56x64
        )
        self.layer1 = original_model.layer1 # layer1: 56x56x64 -> 56x56x64
        self.layer2 = original_model.layer2 # layer2: 56x56x64 -> 28x28x128
        self.layer3 = original_model.layer3 # layer3: 28x28x128 -> 14x14x256
        
        # Layer4와 FC는 사용하지 않음
        # 특징 크기가 7x7로 되면 너무 작아져서 공간 정보가 손실됨

    def forward(self, x):
        """
        Args:
            x(입력): (B, 3, 224, 224)

        Returns:
            x3(출력), x2(Skip), x1(Skip): (B, 256, 14, 14), (B, 128, 28, 28), (B, 64, 56, 56)
        """
        x0 = self.layer0(x)  # (B, 3, 224, 224) -> (B, 64, 56, 56)
        
        x1 = self.layer1(x0) # (B, 64, 56, 56) -> (B, 64, 56, 56) -> Skip1
        x2 = self.layer2(x1) # (B, 64, 56, 56) -> (B, 128, 28, 28) -> Skip2
        x3 = self.layer3(x2) # (B, 128, 28, 28) -> (B, 256, 14, 14) -> Main Input
        
        # x1, x2는 Skip Connection 용으로 반환, x3는 Transformer 입력으로 반환
        return x3, x2, x1