######################################
#  Train Loop                        #
# : Train, Validation per One Epoch  #
######################################

import torch
from tqdm import tqdm # 진행률 바 표시용
from utils.Metrics import Metrics

#==========================
# Train One Epoch
#==========================
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
        one epoch 동안 모델 학습을 수행
    Args:
        model: 학습할 네트워크(TransUNet 기반 3D 복원 모델)
        dataloader: 학습 데이터 로더(train_Loader, val_Loader)
        optimizer: 옵티마이저(Adam)
        criterion: 손실 함수
        device: 실행 디바이스(GPU)
        epoch: 현재 epoch 수
    Returns:
        avg_loss: train 한 epoch 동안 전체 데이터 셋의 평균 Loss
        avg_acc: train 한 epoch 동안 전체 데이터 셋의 평균 정확도
        avg_iou: train 한 epoch 동안 전체 데이터 셋의 평균 IoU
    """
    # 학습 모드 활성화
    model.train()
    
    # 손실 누적 변수 초기화
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    
    # tqbm: 진행 상태를 표시하는 라이브러리, desc: 진행 바 앞에 표시할 텍스트
    loop = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    
    # 배치 단위로 학습
    for batch in loop:

        # 배치 데이터를 GPU로 이동
        images = batch['img'].to(device)         # (B, 3, 224, 224)
        points = batch['points'].to(device)      # (B, N, 3)
        occ = batch['occupancies'].to(device)    # (B, N)
        k = batch['inputs_k'].to(device)         # (B, 3, 3)
        rt = batch['inputs_rt'].to(device)       # (B, 3, 4)
        
        # occ 차원 맞추기 (B, N) -> (B, N, 1)
        if occ.dim() == 2:
            occ_target = occ.unsqueeze(-1)
        else:
            occ_target = occ
            
        # Gradient 초기화
        optimizer.zero_grad()
        
        # 모델 Forward 연산
        preds = model(images, points, k, rt)
        
        # 오차 계산
        loss = criterion(preds, occ_target)
        
        # 오차 기반 Backpropagation 및 가중치 업데이트
        loss.backward()  # 기울기 계산
        optimizer.step() # 가중치 수정
        
        # IoU 계산
        batch_iou = Metrics.calc_iou(preds, occ_target)
        
        with torch.no_grad():
            # Logit > 0.0 이면 내부(1), 아니면 외부(0)
            predicted_labels = (preds > 0.0).float()
            
            # 정확도 계산 (정답과 똑같은 점의 개수 / 전체 점 개수)
            correct = (predicted_labels == occ_target).float().sum()
            accuracy = correct / occ_target.numel()
        
        # 누적 계산
        total_loss += loss.item()
        total_iou += batch_iou
        total_acc += accuracy.item()
        
        # 진행률 바에 현재 Loss 및 정확도 표시
        loop.set_postfix(loss=loss.item(), iou=batch_iou, acc=accuracy.item())

    # 전체 평균 계산
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    
    # 1 epoch 당 평균 Loss, IoU 및 정확도 반환
    return avg_loss, avg_iou, avg_acc


#==========================
# Validate One Epoch
#==========================
def validate_one_epoch(model, dataloader, criterion, device, epoch):
    """
        one epoch 동안 모델 평가를 수행
    Args:
        model: 학습할 네트워크(TransUNet 기반 3D 복원 모델)
        dataloader: 학습 데이터 로더(train_Loader, val_Loader)
        criterion: 손실 함수
        device: 실행 디바이스(GPU)
        epoch: 현재 epoch 수
    Returns:
        avg_loss: Validation 한 epoch 동안 전체 데이터 셋의 평균 Loss
        avg_acc: Validation 한 epoch 동안 전체 데이터 셋의 평균 정확도
        avg_iou: Validation 한 epoch 동안 전체 데이터 셋의 평균 IoU
    """
    # 평가 모드 활성화
    model.eval()
    
    # 손실 누적 변수 초기화
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    
    # Gradient 계산 비활성화, 검증 단계에서는 가중치 업데이트 안함
    with torch.no_grad():
        # tqbm: 진행 상태를 표시하는 라이브러리, desc: 진행 바 앞에 표시할 텍스트
        loop = tqdm(dataloader, desc=f"Val Epoch {epoch}")
        
        # 배치 단위로 학습
        for batch in loop:
            
            # 배치 데이터를 GPU로 이동
            images = batch['img'].to(device)
            points = batch['points'].to(device)
            occ = batch['occupancies'].to(device)
            k = batch['inputs_k'].to(device)
            rt = batch['inputs_rt'].to(device)
            
            # occ 차원 맞추기 (B, N) -> (B, N, 1)
            if occ.dim() == 2:
                occ_target = occ.unsqueeze(-1)
            else:
                occ_target = occ
                
            # 모델 Forward 연산
            preds = model(images, points, k, rt)
            
            # 오차 계산
            loss = criterion(preds, occ_target)
            
            # IoU 계산
            batch_iou = Metrics.calc_iou(preds, occ_target)
            
            # Logit > 0.0 이면 내부(1), 아니면 외부(0)
            predicted_labels = (preds > 0.0).float()
                
            # 정확도 계산 (정답과 똑같은 점의 개수 / 전체 점 개수)
            correct = (predicted_labels == occ_target).float().sum()
            accuracy = correct / occ_target.numel()
            
            # 누적 계산
            total_loss += loss.item()
            total_iou += batch_iou
            total_acc += accuracy.item()
            
            # 진행률 바에 현재 Loss 및 정확도 표시
            loop.set_postfix(loss=loss.item(), iou=batch_iou, acc=accuracy.item())
          
    # 전체 평균 계산  
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    
    # 1 epoch 당 평균 Loss, IoU 및 정확도 반환
    return avg_loss, avg_iou, avg_acc