#########################################
#  Loss Function                        #
# : Occupancy Prediction Loss Function  #
#########################################

import torch.nn as nn

# Loss 함수 정의
class OccupancyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        """
            BCEWithLogitsLoss : 모델이 추정한 logits 값을 확률(0~1)로 변환하고 정답과 비교해서 오차 계산
        """
        # loss 채점 도구 생성
        # reduction='mean': 배치 내 모든 점의 오차 평균을 구함
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, preds, targets):
        """
        Args:
            preds: 모델 예측값 (Logits) -> (B, N, 1)
            targets: 정답 라벨 (0 or 1) -> (B, N) 또는 (B, N, 1)
        
        Returns:
            loss: 스칼라(Scalar) 오차 값
        """
        #==========================
        # 1. BCEWithLogitsLoss
        #==========================
        
        # targets가 (B, N)으로 들어올 경우 (B, N, 1)로 맞춰줌
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)
            
        # targets 값을 float 형태로 변환
        targets = targets.float()
        
        # 오차 계산
        loss = self.criterion(preds, targets)
        
        return loss
