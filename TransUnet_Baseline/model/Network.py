######################################################
#  Model Network                                     #
# : Full TransUNet + Occupancy Decoder Architecture  #
######################################################

import torch.nn as nn
from model.Encoder import ResNet34Encoder
from model.Decoder import TransformerBottleneck, FeatureDecoder2D, ImplicitDecoder

# 최종 모델 정의
class BaselineTransUNet(nn.Module):
    def __init__(self):
        super(BaselineTransUNet, self).__init__()
        
        #==========================================
        # Encoder (Backbone: ResNet34)
        #==========================================
        # :2D RGB 이미지에서 공간 특징 추출
        self.encoder = ResNet34Encoder(pretrained=True)
        # Output: x3(Main), x2(Skip), x1(Skip) 반환
        
        #==========================================
        # Bottleneck (Transformer)
        #==========================================
        # : 14x14 해상도에서 이미지 전체의 문맥(Global Context) 학습
        self.bottleneck = TransformerBottleneck(
            in_channels=256,  # ResNet Layer3 출력 채널
            embed_dim=256,    # Transformer 내부 차원
            num_layers=4,     # 레이어 깊이
            num_heads=8       # 멀티헤드 개수
        )
        
        #==========================================
        # 2D Feature Decoder (TransUNet)
        #==========================================
        # : Transformer 출력값에 Skip Connection을 합쳐 56x56까지 복원
        self.feature_decoder = FeatureDecoder2D(
            main_ch=256,      # Transformer 출력
            skip2_ch=128,     # ResNet Layer2 (skip)
            skip1_ch=64,      # ResNet Layer1 (skip)
            out_ch=128        # 최종 2D 특징 맵 채널 수
        )
        # output: 2D 특징 지도 (B, 128, 56, 56)
        
        #==========================================
        # 3D Implicit Decoder
        #==========================================
        # : 3D 점을 투영하여 특징을 추출하고 Occupancy(내부/외부) 예측
        self.implicit_decoder = ImplicitDecoder(
            feature_dim=128,  # FeatureDecoder 출력 채널
            hidden_dim=64,    # MLP 은닉층 크기
            mapping_size=64,  # 퓨리에 매핑 차원
            scale=10          # 퓨리에 매핑 스케일 (클수록 고주파)
        )
        # output: 점별 Occupancy 여부 (B, N, 1)

    def forward(self, images, points, cameras_k, cameras_rt):
        """
        args:
            images: (B, 3, 224, 224) -> 입력 이미지
            points: (B, N, 3)        -> 물어볼 3D 좌표들 (Query Points)
            cameras_k: (B, 3, 3)     -> 카메라 내부 파라미터
            cameras_rt: (B, 3, 4)    -> 카메라 외부 파라미터
        returns:
            logits: (B, N, 1)        -> Occupancy 예측값
        """
        
        # 1. Feature Extraction (Encoder)
        # x3: (B, 256, 14, 14) -> 가장 깊은 특징
        # x2: (B, 128, 28, 28) -> 중간 특징 (Skip)
        # x1: (B, 64, 56, 56)  -> 얕은 특징 (Skip)
        x3, x2, x1 = self.encoder(images)
        
        # 2. Global Context Modeling (Transformer)
        # x3_trans: (B, 256, 14, 14) -> 전역 정보가 강화된 특징
        x3_trans = self.bottleneck(x3)
        
        # 3. Feature Fusion & Upsampling (2D Decoder)
        # feature_grid: (B, 128, 56, 56) -> Transformer 출력 + x2 + x1을 모두 융합한 특징 지도
        feature_grid = self.feature_decoder(x3_trans, x2, x1)
        
        # 4. Occupancy Prediction (3D Decoder)
        # logits: (B, N, 1) -> 3D 점을 2D 지도에 투영하고 퓨리에 매핑을 적용해 예측
        logits = self.implicit_decoder(feature_grid, points, cameras_k, cameras_rt)
        
        return logits