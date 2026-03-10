#################################################################
#  Model Decoder                                                #
# : Global (OccNet) + Local Two-Track Decoder                   #
#################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# 1. Global Decoder 정의
class GlobalDecoder(nn.Module):
    def __init__(self, dim=3, c_dim=256, hidden_size=256):
        """
            글로벌 특징을 기반으로 3D 점의 Occupancy를 예측하는 Decoder
        Args:
            dim: 입력 점의 차원 (x, y, z)
            c_dim: 이미지 특징 벡터 차원
            hidden_size: MLP 은닉층 크기
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

# 2. Local Decoder 정의
class LocalDecoder(nn.Module):
    def __init__(self, dim=3, c_dim=256, hidden_size=512, num_blocks=4):
        """
            투영된 2D 로컬 특징들을 모아서 잔차(Residual)를 학습하는 Decoder
        Args:
            dim: 입력 점의 차원 (x, y, z)
            c_dim: 글로벌 특징 벡터 차원
            hidden_size: MLP 은닉층 크기
            num_blocks: 잔차 블록 개수
        """
        super().__init__()
        
        # 3D 좌표 임베딩 
        self.point_mlp = nn.Sequential(
            nn.Linear(dim, 64), # (B, N, 3) -> (B, N, 64)
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.z_mlp = nn.Sequential(
            nn.Linear(1, 16),   # (B, N, 1) -> (B, N, 16)
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Attention 기반 고주파수 특징 게이트
        self.depth_gate = nn.Sequential(
            nn.Conv1d(416, 64, 1), 
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # 입력 채널 수
        in_channels = 960 + 64 + c_dim
        
        # 레이어
        self.fc_in = nn.Conv1d(in_channels, hidden_size, 1) # (B, 1280, N) -> (B, hidden_size, N)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, 1)
            ) for _ in range(num_blocks)
        ])
        
        self.relu = nn.ReLU()
        self.fc_out = nn.Conv1d(hidden_size, 1, 1)  # (B, hidden_size, N) -> (B, 1, N)
        
    # 3D 점을 2D 지도에 매핑해서 정보를 추출 (투영 + 보간법)
    def query_feature(self, grid, points_norm, scale, loc, camera_k, camera_rt):
        """
        Args:
            grid: 2D 특징 지도 (B, C, H, W)
            points: 정규화된 일반 3D 점 좌표 (B, N, 3)
            scale: 정규화 스케일 (B, 1)
            loc: 정규화 위치 (B, 3)
            camera_k: 카메라 내부 파라미터 (B, 3, 3)
            camera_rt: 카메라 외부 파라미터 (B, 3, 4)
            
        Returns:
            features: 2D 특징 지도에서 추출된 3D 점의 특징 (B, N, C)
            z_cam: 카메라 기준 3D 점의 Z축 깊이 값 (B, N, 1)
        """
        B, N, _ = points_norm.shape
        
        # 정규화 해제
        points = (points_norm * scale.view(B, 1, 1)) + loc.view(B, 1, 3)
        
        # world -> image 좌표 변환
        # 동차 좌표 만들기 (x, y, z) -> (x, y, z, 1) : 행렬 곱셈을 하기 위함
        points_homo = torch.cat([points, torch.ones(B, N, 1, device=points.device)], dim=2)
        # 외부 파라미터 적용
        points_cam = torch.bmm(camera_rt, points_homo.permute(0, 2, 1)).permute(0, 2, 1) 
        points_cam = points_cam[:, :, :3]
        
        z_cam = points_cam[:, :, 2:3]
        
        # camera -> image 좌표 변환
        # 내부 파라미터 적용
        points_img = torch.bmm(camera_k, points_cam.permute(0, 2, 1)).permute(0, 2, 1) 
        
        # 원근 나눗셈
        z = points_img[:, :, 2:3] + 1e-8 
        uv = points_img[:, :, :2] / z 
        valid_mask = (points_img[:, :, 2:3] > 0).float()
        
        # 좌표 정규화
        # align_corners=False 기준 정규화 (0~224 -> -1~1)
        img_H, img_W = 224.0, 224.0
        
        u = uv[..., 0]
        v = uv[..., 1]
        
        u_norm = (2.0 * u + 1.0) / img_W - 1.0
        v_norm = (2.0 * v + 1.0) / img_H - 1.0
        
        uv_norm = torch.stack([u_norm, v_norm], dim=-1)
        
        # 그리드 샘플링
        uv_norm = uv_norm.unsqueeze(1) # (B, 1, N, 2)
        features = F.grid_sample(grid, uv_norm, align_corners=False, mode='bilinear', padding_mode='zeros')
        features = features.squeeze(2).transpose(1, 2) 
        
        # 카메라 뒤에 있는 점들은 특징을 강제로 0으로 만듦(무시)
        features = features * valid_mask
        
        return features, z_cam
    
    def forward(self, points, local_features, global_feat, hf_feat, scale, loc, camera_k, camera_rt):
        """
        Args:
            points: 일반 3D 점 좌표 (B, N, 3)
            global_feat: 글로벌 특징 벡터 (B, 256)
            local_features: 각 레이어의 로컬 특징
            hf_feat: 고주파수 특징 지도
            scale: 정규화 스케일 (B, 1)
            loc: 정규화 위치 (B, 3)
            camera_k: 카메라 파라미터
            camera_rt: 카메라 파라미터
        """
        sampled_feats = []
        
        # 로컬 특징에 3D 좌표 투영
        for grid in local_features:
            feat, _ = self.query_feature(grid, points, scale, loc, camera_k, camera_rt)
            sampled_feats.append(feat)
        local_feat = torch.cat(sampled_feats, dim=-1)
        
        # 고주파수 특징 지도 투영 및 z축 깊이 반환
        hf_feat_sampled, z_cam = self.query_feature(hf_feat, points, scale, loc, camera_k, camera_rt)
        
        # 3D 좌표 임베딩
        point_embed = self.point_mlp(points)    # (B, N, 3) -> (B, N, 64)
        z_embed = self.z_mlp(z_cam)             # (B, N, 1) -> (B, N, 16)
        
        # 글로벌 특징 모양 변환
        B, N, _ = points.shape
        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, N, -1)
        
        # Attention 기반 고주파수 특징 게이트
        gate_input = torch.cat([global_feat_expanded, hf_feat_sampled, z_embed], dim=-1).transpose(1, 2)
        final_3d_mask = self.depth_gate(gate_input).transpose(1, 2) # (B, N, 1)
        local_feat = local_feat * (final_3d_mask * 2.0)
        
        # 특징 융합 (로컬 960 + 좌표임베딩 64 + 글로벌 256 = 1280차원)
        combined_feat = torch.cat([local_feat, point_embed, global_feat_expanded], dim=-1)
        net = combined_feat.transpose(1, 2) # (B, N, 1280) -> (B, 1280, N)
        
        net = self.relu(self.fc_in(net)) # (B, 1280, N) -> (B, 512, N)
        for block in self.blocks:   # 잔차학습
            res = net
            net = block(net)
            net = self.relu(net + res) 
            
        out = self.fc_out(net) # (B, 512, N) -> (B, 1, N)
        
        return out.squeeze(1) # (B, N)