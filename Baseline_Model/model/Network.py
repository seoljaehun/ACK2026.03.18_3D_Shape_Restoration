######################################################
#   Model Network                                    #
# : ResNet18 Encoder + CBN Implicit Decoder          #
######################################################

import torch.nn as nn
from model.Encoder import Resnet18Encoder
from model.Decoder import ImplicitDecoder

# 최종 모델 정의
class OccupancyNetwork(nn.Module):
    def __init__(self, c_dim=256):
        """
            Occupancy Network 모델 정의
        Args:
            c_dim: 이미지 특징 벡터 차원
        """
        super(OccupancyNetwork, self).__init__()
        
        #==========================================
        # 1. Encoder (Backbone: ResNet18)
        #==========================================
        # :2D RGB 이미지에서 공간 특징 추출
        self.encoder = Resnet18Encoder(c_dim=c_dim, pretrained=True)
        # Output: global_feat(main) 반환
        
        #==========================================
        # 2. 3D Implicit Decoder
        #==========================================
        # : 3D 점을 투영하여 특징을 추출하고 Occupancy(내부/외부) 예측
        self.decoder = ImplicitDecoder(
            dim=3,            # 입력 점의 차원 (x, y, z)
            c_dim=c_dim,      # 이미지 특징 차원
            hidden_size=256,  # MLP 내부 은닉층 크기
        )
        # output: 점별 Occupancy 여부 (B, N, 1)

    def forward(self, images, points):
        """
        args:
            images: (B, 3, 224, 224) -> 입력 이미지
            points: (B, N, 3)        -> 물어볼 3D 좌표들 (Query Points)
            
        returns:
            logits: (B, N)           -> Occupancy 예측값
        """
        
        # 1. Feature Extraction (Encoder)
        global_feat = self.encoder(images)  # (B, 256)

        # 2. Occupancy Prediction (3D Decoder)
        logits = self.decoder(points, global_feat)
        
        return logits