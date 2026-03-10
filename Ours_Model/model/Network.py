######################################################
#   Model Network                                    #
# : Global (OccNet) + Heavy Local Two-Track Network  #
######################################################

import torch.nn as nn
from model.Encoder import Resnet18Encoder
from model.Decoder import GlobalDecoder, LocalDecoder

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
        # Output: global_feat(main), Local_feat 반환
        
        #==========================================
        # 2. Global Decoder 
        #==========================================
        # : 글로벌 특징을 기반으로 대략적인 형상(Topology) 예측
        self.decoder = GlobalDecoder(dim=3, c_dim=c_dim, hidden_size=256)
        # Output: global 3D Shape
        
        #==========================================
        # 3. Local Decoder
        #==========================================
        # : 3D 점을 투영하여 특징을 추출하고 잔차(Residual) 예측
        self.local_decoder = LocalDecoder(dim=3, c_dim=c_dim, hidden_size=512, num_blocks=4)
        # output: Local Residual

    def freeze_global_model(self):
        """
            Encoder와 Global Decoder의 가중치 동결
            오직 Local Decoder만 역전파(Backpropagation)를 통해 학습
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        for param in self.encoder.hf_feat_generator.parameters():
            param.requires_grad = True
            
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def forward(self, images, points, scale, loc, camera_k, camera_rt):
        """
        args:
            images: (B, 3, 224, 224) -> 입력 이미지
            points: (B, N, 3)        -> 정규화된 3D 좌표들 (Query Points)
            scale: (B, 1)            -> 역정규화를 위한 스케일
            loc: (B, 3)              -> 역정규화를 위한 중심 좌표
            camera_k: (B, 3, 3)      -> 카메라 내부 파라미터 행렬
            camera_rt: (B, 3, 4)     -> 카메라 외부 파라미터 행렬
            
        returns:
            final_logits: (B, N)           -> Occupancy 예측값
        """        
        # 1. Feature Extraction (Encoder)
        global_feat, local_feat, hf_features = self.encoder(images)  # (B, 256)
        
        # 2. Global Decoder
        global_logits = self.decoder(points, global_feat)

        # 3. Local Residual Prediction
        local_logits = self.local_decoder(
            points=points, 
            local_features=local_feat, 
            global_feat=global_feat, 
            hf_feat=hf_features,
            scale=scale, 
            loc=loc, 
            camera_k=camera_k, 
            camera_rt=camera_rt
        )
        
        # 최종 3D 형상 복원
        final_logits = global_logits + local_logits
        
        return final_logits
