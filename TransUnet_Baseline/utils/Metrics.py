################################################
#  Metrics function                            #
# : Volumetric IoU, Chamfer Distance, F-Score  #
################################################

import torch

class Metrics:
    """
        3D Reconstruction 평가 지표:
        1. calc_iou: Volumetric IoU (부피 겹침 정도)
        2. calc_chamfer: Chamfer Distance (예측된 점과 정답 점 사이의 최단 거리 평균)
        3. calc_f_score: F-Score (거리 기반의 정밀도/재현율 조화 평균)
    """

    @staticmethod
    def calc_iou(pred_logits, gt_occ, threshold=0.5):
        """
            Volumetric IoU
            : Occupancy 예측값과 정답 사이의 Intersection over Union 계산
        
        Args:
            pred_logits: 모델 예측값 (Logits) [B, N, 1] 또는 [B, N]
            gt_occ: 정답 Occupancy (0 또는 1) [B, N]
            threshold: 0.5 기준
        Returns:
            avg_iou: 배치 평균 IoU (Scalar)
        """
        
        # 차원 변환, [B, N, 1] -> [B, N]
        if pred_logits.dim() == 3:
            pred_logits = pred_logits.squeeze(-1)
            
        # 예측값 확률로 변환 및 이진화
        probs = torch.sigmoid(pred_logits)
        pred_binary = (probs >= threshold).float()
        
        # 교집합 & 합집합
        intersection = (pred_binary * gt_occ).sum(dim=1)
        union = pred_binary.sum(dim=1) + gt_occ.sum(dim=1) - intersection
        
        # IOU 계산
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        # 배치 1개의 IOU 평균 반환
        return iou.mean().item()
