########################################
#  Model Encoder                       #
# :ResNet18 (Feature Extraction)       #
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

#==========================================
# Frequency Feature Extraction
#==========================================

# 1. Discrete Wavelet Transform 정의
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        """
            2D 이미지로부터 Haar wavelet 변환을 수행하여 가로, 세로, 대각선 방향 고주파수 추출
        """
        self.requires_grad = False  # 학습되지 않는 고정 필터
        
        # Haar Wavelet 필터 정의 (LL, LH, HL, HH)
        # LL: Low-Low (평균)
        self.register_buffer('ll_filter', torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        # LH: Low-High (수직)
        self.register_buffer('lh_filter', torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        # HL: High-Low (수평)
        self.register_buffer('hl_filter', torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        # HH: High-High (대각선)
        self.register_buffer('hh_filter', torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x(입력): 2D 객체 이미지 (B, 3, H, W)

        Returns:
            ll: 저주파수
            lh: 수직 방향 고주파수
            hl: 수평 방향 고주파수
            hh: 대각선 방향 고주파수
        """
        b, c, h, w = x.shape
        
        # RGB 채널을 따로따로 나누기 (B, 3, H, W) -> (Bx3, 1, H, W)
        x_reshaped = x.contiguous().view(b * c, 1, h, w)
        
        # DWT 수행 (Stride 2 -> 해상도 1/2 감소)
        ll = F.conv2d(x_reshaped, self.ll_filter, stride=2, padding=0)
        lh = F.conv2d(x_reshaped, self.lh_filter, stride=2, padding=0)
        hl = F.conv2d(x_reshaped, self.hl_filter, stride=2, padding=0)
        hh = F.conv2d(x_reshaped, self.hh_filter, stride=2, padding=0)
        
        # 차원 복구 (B*3, 1, H/2, W/2) -> (B, 3, H/2, W/2)
        ll = ll.view(b, c, ll.shape[2], ll.shape[3])
        lh = lh.view(b, c, lh.shape[2], lh.shape[3])
        hl = hl.view(b, c, hl.shape[2], hl.shape[3])
        hh = hh.view(b, c, hh.shape[2], hh.shape[3])
        
        # 각 주파수 이미지 반환
        return ll, lh, hl, hh

# 2. Residual Frequency Block 정의
class RFB(nn.Module):
    def __init__(self, channels):
        super(RFB, self).__init__()
        """
            입력된 고주파 특징의 디테일을 보존하면서 정제
            : Conv -> BN -> ReLU -> Conv -> BN -> Add Residual -> ReLU
        args:
            channels: 입력 채널
        """
        # 1번 Conv: 특징 추출 + BN, ReLU
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 2번 Conv: 특징 추출 + BN, ReLU
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Args:
            x(입력): (B, C, H, W)

        Returns:
            out(출력): (B, C, H, W)
        """
        # 입력 특징 복사
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity       # 잔차 연결
        
        out = self.relu(out)
        
        return out

# 3. Frequency Encoder 정의
class FrequencyEncoder(nn.Module):
    def __init__(self):
        super(FrequencyEncoder, self).__init__()
        """
            이미지의 주파수 정보를 다중 스케일로 추출하는 encoder
            DWT -> 분기(L1, L2) -> CNN -> Fusion -> CNN x 2 -> RFB
        """
        # DWT 객체 생성
        self.dwt = DWT()
        
        # L1 Branch (Input: 112x112, 9채널)
        self.l1_stage1 = nn.Sequential(
            # 1. CNN: 확장 112x112x9 -> 112x112x18
            nn.Conv2d(9, 18, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(18),
            nn.ReLU(inplace=True),
        )
        
        # L2 Branch (Input: 56x56, 9채널)
        self.l2_stage1 = nn.Sequential(
            # 1. CNN: 확장 56x56x9 -> 56x56x18
            nn.Conv2d(9, 18, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(18),
            nn.ReLU(inplace=True),
        )
        
        # Fusion 후 특징 추출
        self.fusion_expand = nn.Sequential(
            # 112x112x36 -> 112x112x72
            nn.Conv2d(36, 72, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            # 112x112x72 -> 112x112x144
            nn.Conv2d(72, 144, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(144),
            nn.ReLU(inplace=True)
        )    
        
        self.final_rfb = RFB(144)
     
    def forward(self, x):
        """
        Args:
            x(입력): 원본 2D 이미지 (B, 3, 224, 224)

        Returns:
            hf_features(출력): 고주파수 특징 지도 (B, 144, 112, 112)
        """
        # [DWT Level 1] -> 112x112x9
        ll1, lh1, hl1, hh1 = self.dwt(x)
        high1 = torch.cat([lh1, hl1, hh1], dim=1) # (B, 9, 112, 112)
        
        # [L1 Branch]
        l1_feat = self.l1_stage1(high1)   # (B, 9, 112, 112) -> (B, 18, 112, 112)
        
        # [DWT Level 2] -> 56x56x9 (from LL1)
        ll2, lh2, hl2, hh2 = self.dwt(ll1)
        high2 = torch.cat([lh2, hl2, hh2], dim=1) # (B, 9, 56, 56)
        
        # [L2 Branch]
        l2_feat = self.l2_stage1(high2)   # (B, 9, 56, 56) -> (B, 18, 56, 56)
        l2_feat_up = F.interpolate(l2_feat, size=(112, 112), mode='bilinear', align_corners=False)
        
        # [Fusion]
        concat = torch.cat([l1_feat, l2_feat_up], dim=1)    # (B, 36, 112, 112)
        expanded = self.fusion_expand(concat)               # (B, 36, 112, 112) -> (B, 144, 112, 112)
        hf_features = self.final_rfb(expanded)              # (B, 144, 112, 112) -> (B, 144, 112, 112)
        
        return hf_features
    
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
        
        # 고주파수 마스크 생성 모듈
        self.hf_feat_generator = FrequencyEncoder()
        
    def forward(self, x):
        """
        Args:
            x(입력): (B, 3, 224, 224)

        Returns:
            global_feat: 글로벌 특징 (B, c_dim)
            local_feat: 각 레이어 로컬 특징 [x1, x2, x3, x4]
            hf_features: 고주파수 특징 지도 (B, 144, 112, 112)
        """
        hf_features = self.hf_feat_generator(x)
        
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
        
        local_feat = [x1, x2, x3, x4]
        
        return global_feat, local_feat, hf_features