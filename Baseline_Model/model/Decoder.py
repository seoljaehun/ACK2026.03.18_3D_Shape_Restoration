#################################################################
#  Model Decoder                                                #
# :Conditional Batch Norm Decoder (OccNet)                      #
#################################################################

import torch.nn as nn

#################################
# CBN 기반 ResNet Block         #
#################################

# 1. CBN(Conditional Batch Normalization) 정의
class CBatchNorm1d(nn.Module):
    def __init__(self, c_dim, f_dim):
        """
            3D 좌표 특징을 Normalize한 뒤, 이미지 특징에 맞춰서 모양을 변형(Scale & Shift)
        Args:
            c_dim: 이미지 특징 벡터 차원
            f_dim: 3D 좌표 특징 차원 (Hidden Dim)
        """
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        # 1D Conv로 이미지 특징에서 Scale과 Shift 예측
        # conv_gamma: 특징 지도를 보고, 3D 특징을 얼마나 증폭시킬지 결정하는 값
        # conv_beta: 특징 지도를 보고, 3D 특징을 얼마나 이동시킬지 결정하는 값
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        
        # 3D 좌표 특징에 대한 BatchNorm (학습 x)
        self.bn = nn.BatchNorm1d(f_dim, affine=False)

        # 가중치 초기화 (gamma=1, beta=0)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
            CBN 레이어 가중치 초기화
        """
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)
        
    def forward(self, x, c):
        """
        Args:
            x: 3D 좌표 특징 (B, f_dim, N)
            c: 이미지 특징 벡터 (B, c_dim)
            
        Returns:
            out: 정규화된 3D 좌표 특징 (B, f_dim, N)
        """
        if c.dim() == 2:
            c = c.unsqueeze(2)     # (B, c_dim) -> (B, c_dim, 1)
            
        gamma = self.conv_gamma(c) # 이미지 특징으로 Scale 예측, (B, f_dim, 1)
        beta = self.conv_beta(c)   # 이미지 특징으로 Shift 예측, (B, f_dim, 1)
        
        # 조건부 정규화 적용
        out = self.bn(x)
        out = gamma * out + beta
        
        return out

# 2. ResNet Block with CBN
class CResnetBlock(nn.Module):
    def __init__(self, c_dim, f_dim):
        """
            CBN이 적용된 ResNet 블록 정의
        Args:
            c_dim: 이미지 특징 벡터 차원
            f_dim: 3D 좌표 특징 차원 (Hidden Dim)
        """
        super().__init__()
        
        # 첫 번째 레이어
        self.bn_0 = CBatchNorm1d(c_dim, f_dim)  # CBN 적용
        self.fc_0 = nn.Conv1d(f_dim, f_dim, 1)  # conv1 (채널, 해상도 유지)    
        
        # 두 번째 레이어
        self.bn_1 = CBatchNorm1d(c_dim, f_dim)  # CBN 적용
        self.fc_1 = nn.Conv1d(f_dim, f_dim, 1) # conv2 (채널, 해상도 유지)
        
        self.actvn = nn.ReLU()   # 활성화 함수

    def forward(self, x, c):
        """
        Args:
            x: 3D 좌표 특징 (B, f_dim, N)
            c: 이미지 특징 벡터 (B, c_dim)

        Returns:
            out: CBN resnet을 거친 3D 좌표 특징 (B, f_dim, N)
        """
        # Skip Connection 저장
        res = x
        
        # First Block
        out = self.bn_0(x, c)
        out = self.actvn(out)
        out = self.fc_0(out)
        
        # Second Block
        out = self.bn_1(out, c)
        out = self.actvn(out)
        out = self.fc_1(out)
        
        # Add Residual
        return out + res
    
############################################
# 3D Implicit Decoder (Main Class)         #
############################################

# 1. Implicit Decoder (Point Query) 정의
class ImplicitDecoder(nn.Module):
    def __init__(self, dim=3, c_dim=256, hidden_size=256):
        """
            2D 특징 지도를 기반으로 3D 점의 Occupancy를 예측하는 Decoder
        Args:
            dim: 입력 점의 차원 (x, y, z)
            c_dim: 이미지 특징 벡터 차원
            hidden_size: MLP 은닉층 크기
            num_blocks: ResNet Block 개수
        """
        super().__init__()
        self.c_dim = c_dim
        
        # 좌표 임베딩 (3 -> hidden_size)
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        
        # CBN ResNet Block 생성
        self.block0 = CResnetBlock(c_dim, hidden_size)
        self.block1 = CResnetBlock(c_dim, hidden_size)
        self.block2 = CResnetBlock(c_dim, hidden_size)
        self.block3 = CResnetBlock(c_dim, hidden_size)
        self.block4 = CResnetBlock(c_dim, hidden_size)
        
        # 마지막 정규화 및 출력 레이어
        self.bn = CBatchNorm1d(c_dim, hidden_size)
        self.fc_out = nn.Conv1d(hidden_size, 1, 1) # 1 = Occupancy Logit
        self.actvn = nn.ReLU()
    
    def forward(self, points, global_feat):
        """
        Args:
            points: 일반 3D 점 좌표 (B, N, 3)
            global_feat: 글로벌 특징 벡터 (B, 256)

        Returns:
            out: 3D 점별 Occupancy 예측값 (B, N, 1)
        """ 
        
        # 차원 변환    
        p = points.transpose(1, 2) # (B, N, 3) -> (B, 3, N)

        # 3D 좌표 임베딩
        net = self.fc_p(p)  # (B, 3, N) -> (B, hidden_size, N)

        # CBN ResNet Blocks
        for block in [self.block0, self.block1, self.block2, self.block3, self.block4]:
            net = block(net, global_feat)
            
        # 최종 CBN, 활성화 함수 적용
        out = self.bn(net, global_feat)
        out = self.actvn(out)
        out = self.fc_out(out) # (B, hidden_size, N) -> (B, 1, N)

        # 차원 복구
        out = out.squeeze(1)    # (B, 1, N) -> (B, N)
        
        return out