#################################################################
#  Model Decoder                                                #
# :Transformer Bottleneck + Feature Decoder + Implicit Decoder  #
#################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#==========================================
# Transformer Bottleneck
#==========================================

# 1. Transformer Bottleneck 정의
class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels=256, embed_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        """
            Transformer Bottleneck 모듈 초기화
        Args:
            in_channels: 입력 채널 수 (ResNet34 Layer3 출력 채널 수)
            embed_dim: Transformer 임베딩 차원(벡터 크기)
            num_layers: Transformer 레이어 수
            num_heads: 멀티헤드 어텐션 헤드 수
        """
        # Learnable Absolute Positional Embedding
        # : 데이터의 순서를 학습시키기 위해 파라미터 부여
        # 14x14 = 196 patches를 만들고 각 패치에 무작위 위치 임베딩 부여(학습 가능)
        self.pos_embedding = nn.Parameter(torch.randn(1, (14*14), embed_dim) * 0.02)
        
        # Transformer Encoder
        # transformer 기본 블록 (Self-Attention + Feed Forward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        # encoder_layer을 num_layers개 쌓아서 TransformerEncoder 생성
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 입력 채널 수가 임베딩 차원과 다르면 1x1 Conv로 차원 맞춤
        self.project_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1) if in_channels != embed_dim else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x(입력): (B, 256, 14, 14)

        Returns:
            feat(출력): (B, 256, 14, 14)
        """
        x = self.project_in(x)
        B, C, H, W = x.shape
        
        # Flatten: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2) # (B, 256, 14, 14) -> (B, 196, 256)
        
        # Add Positional Embedding
        x_flat = x_flat + self.pos_embedding
        
        # Transformer Process
        feat = self.transformer(x_flat) # (B, 196, 256)
        
        # Reshape back: (B, 196, 256) -> (B, 256, 14, 14)
        feat = feat.transpose(1, 2).view(B, C, H, W)
        
        return feat

#==========================================
# 2D Feature Decoder (TransUNet Style)
#==========================================

# 1. TransUNet Decoder 정의
class FeatureDecoder2D(nn.Module):
    def __init__(self, main_ch=256, skip2_ch=128, skip1_ch=64, out_ch=128):
        super().__init__()
        """
            TransUNet 스타일 2D 특징 디코더 초기화
        Args:
            main_ch: Transformer 출력 채널 수
            skip2_ch: ResNet34 Layer2 출력 채널 수 (x2: Skip Connection)
            skip1_ch: ResNet34 Layer1 출력 채널 수 (x1: Skip Connection)
            out_ch: 최종 출력 채널 수 (Implicit Decoder 입력용)
        """
        # Upsample: 이미지 크기 강제로 2배 키우기 (14x14 -> 28x28)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # conv1: 메인 피처와 스킵2 피처 합치기
        self.conv1 = nn.Sequential( # 메인(256) + 스킵2(128) = 384
            nn.Conv2d(main_ch + skip2_ch, 256, kernel_size=3, padding=1),   # 384 -> 256 압축
            nn.BatchNorm2d(256),    # 정규화 및 활성화 함수
            nn.ReLU(inplace=True)
        )
        
        # Upsample: 이미지 크기 강제로 2배 키우기 (28x28 -> 56x56)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # conv2: 메인 피처와 스킵1 피처 합치기
        self.conv2 = nn.Sequential( # 메인(256) + 스킵1(64) = 320
            nn.Conv2d(256 + skip1_ch, out_ch, kernel_size=3, padding=1),    # 320 -> 128 압축
            nn.BatchNorm2d(out_ch), # 정규화 및 활성화 함수
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1) # Refine, 추가 공정
        )

    def forward(self, x, skip2, skip1):
        """
        Args:
            x(입력): (B, 256, 14, 14)
            skip2: (B, 128, 28, 28)
            skip1: (B, 64, 56, 56)

        Returns:
            x(출력): 특징 지도(B, 128, 56, 56)
        """
        # Upsample to 28x28
        x = self.up1(x) # (B, 256, 14, 14) -> (B, 256, 28, 28)
        x = torch.cat([x, skip2], dim=1) # Concat (256+128) -> (B, 384, 28, 28)
        x = self.conv1(x) # (B, 384, 28, 28) -> (B, 256, 28, 28)
        
        # Upsample to 56x56
        x = self.up2(x) # (B, 256, 28, 28) -> (B, 256, 56, 56)
        x = torch.cat([x, skip1], dim=1) # Concat (256+64) -> (B, 320, 56, 56)
        x = self.conv2(x) # (B, 320, 56, 56) -> (B, 128, 56, 56)
        
        return x

# 2. 3D Implicit Decoder (Point Query) 정의
class ImplicitDecoder(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=64, mapping_size=64, scale=10):
        super().__init__()
        """
        Args:
            feature_dim: 2D 특징 지도 채널 수
            hidden_dim: MLP 은닉층 크기
            mapping_size: 퓨리에 매핑 차원
            scale: 퓨리에 매핑 스케일 (클수록 고주파)
        """
        # 퓨리에 매핑 설정
        # mapping_size: 퓨리에 특징의 차원 수 (클수록 고주파 성분 많아짐)
        # scale: 가우시안 분포의 표준편차 (클수록 더 높은 주파수까지 커버)
        self.mapping_size = mapping_size
        self.scale = scale
        
        # 가우시안 랜덤 행렬 B 생성 (mapping_size x 3)
        # register_buffer를 써야 모델 저장/로드 시에 값이 유지됨 (학습이 안되는 고정 값)
        self.register_buffer('B_gauss', torch.randn((mapping_size, 3)) * scale)
        
        # MLP 입력 차원 계산
        # 입력: Feature(128) + 퓨리에 특징(sin, cos 각각 mapping_size개)
        # -> 128 + (64 * 2) = 256
        input_dim = feature_dim + (mapping_size * 2)
        
        # MLP 정의
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output: Occupancy (확신 Score)
        )

    # 3D 좌표(x,y,z)를 고차원 퓨리에 특징으로 변환 (퓨리에 매핑)
    def fourier_mapping(self, coords):
        """
        Args:
            coords: 입력 좌표 (B, N, 3)

        Returns:
            퓨리에 3D 좌표: 퓨리에 특징 (B, N, mapping_size * 2)
        """
        # 행렬 곱셈: (B, N, 3) @ (3, mapping_size) -> (B, N, mapping_size)
        x_proj = (2.0 * np.pi * coords) @ self.B_gauss.t()
        
        # [sin, cos] 연결 -> (B, N, mapping_size * 2)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    # 3D 점을 2D 지도에 매핑해서 정보를 추출 (투영 + 보간법)
    def query_feature(self, grid, points, camera_k, camera_rt):
        """
        Args:
            grid: 2D 특징 지도 (B, C, H, W)
            points: 일반 3D 점 좌표 (B, N, 3)
            camera_k: 카메라 내부 파라미터 (B, 3, 3)
            camera_rt: 카메라 외부 파라미터 (B, 3, 4)
            
        Returns:
            features: 2D 특징 지도에서 추출된 3D 점의 특징 (B, N, C)
        """
        B, N, _ = points.shape
        
        # world -> image 좌표 변환
        # 동차 좌표 만들기 (x, y, z) -> (x, y, z, 1) : 행렬 곱셈을 하기 위함
        points_homo = torch.cat([points, torch.ones(B, N, 1, device=points.device)], dim=2)
        # 외부 파라미터 적용
        points_cam = torch.bmm(camera_rt, points_homo.permute(0, 2, 1)).permute(0, 2, 1) 
        
        # camera -> image 좌표 변환
        # 내부 파라미터 적용
        points_img = torch.bmm(camera_k, points_cam.permute(0, 2, 1)).permute(0, 2, 1) 
        
        # 원근 나눗셈
        z = points_img[:, :, 2:3] + 1e-8 
        uv = points_img[:, :, :2] / z 
        valid_mask = (points_img[:, :, 2:3] > 0).float()
        
        # 좌표 정규화
        # align_corners=False 기준 정규화 (0~224 -> -1~1)
        H, W = grid.shape[-2:]
        
        u = uv[..., 0]
        v = uv[..., 1]
        
        u_norm = (2.0 * u + 1.0) / W - 1.0
        v_norm = (2.0 * v + 1.0) / H - 1.0
        
        uv_norm = torch.stack([u_norm, v_norm], dim=-1)
        
        # 그리드 샘플링
        uv_norm = uv_norm.unsqueeze(1) # (B, 1, N, 2)
        features = F.grid_sample(grid, uv_norm, align_corners=False, mode='bilinear', padding_mode='zeros')
        features = features.squeeze(2).transpose(1, 2) 
        
        # 카메라 뒤에 있는 점들은 특징을 강제로 0으로 만듦(무시)
        features = features * valid_mask
        
        return features

    def forward(self, features, points, k, rt):
        """
        Args:
            features: 2D 특징 지도 (B, 128, 56, 56)
            points: 일반 3D 점 좌표 (B, N, 3)
            k: 카메라 내부 파라미터 (B, 3, 3)
            rt: 카메라 외부 파라미터 (B, 3, 4)

        Returns:
            out: 3D 점별 Occupancy 예측값 (B, N, 1)
        """

        # Interpolate features from 2D map (원본 좌표 사용)
        img_feats = self.query_feature(features, points, k, rt)
        
        # Fourier Mapping
        coord_feats = self.fourier_mapping(points) # (B, N, mapping_size*2)
        
        # Concat (이미지 특징 + 퓨리에 좌표)
        inp = torch.cat([img_feats, coord_feats], dim=2)
        
        # MLP -> predict occupancy
        out = self.fc(inp) 
        
        return out