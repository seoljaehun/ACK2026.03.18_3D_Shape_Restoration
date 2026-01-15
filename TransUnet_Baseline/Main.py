#############################################################
#  Main loop                                                #
# : Training and validation loop to save the optimal model  #
#############################################################

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom modules
from utils.Config import Config
from model.Network import BaselineTransUNet
from loss.Loss import OccupancyLoss
from dataset.Dataset import ShapeNetDataset
from train.Train import train_one_epoch, validate_one_epoch
from utils.Checkpoint import save_checkpoint, load_checkpoint

if __name__ == "__main__":
    torch.cuda.empty_cache()  # GPU 캐시 초기화
    
    # 시드 고정
    Config.set_seed(Config.SEED)
    
    # device = GPU
    device = Config.device
    
    #==========================
    # Dataset & DataLoader
    #==========================
    
    # 데이터 셋 로드
    train_dataset = ShapeNetDataset(data_root=Config.root_dir, split="Train")
    val_dataset = ShapeNetDataset(data_root=Config.root_dir, split="Val")
    
    # 시드 고정용 generator 생성
    g = torch.Generator()
    g.manual_seed(Config.SEED)
    
    # Train Loader
    # batch_size: batch 수 만큼 데이터를 묶어 GPU에 전달
    # shuffle = True: 매 epoch마다 순서를 섞어 학습 안정화
    # num_workers: 멀티스레드로 데이터 로딩 속도 향상
    # pin_memory=True: GPU로 데이터를 옮길 때 속도 향상
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        worker_init_fn=Config.seed_worker,
        generator=g
    )
    
    # Validation Loader
    # 검증에서는 shuffle = False (순서 안중요함)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        worker_init_fn=Config.seed_worker,
        generator=g
    )
    
    #==========================
    # Model, Loss, Optimizer
    #==========================
    
    # main model load
    model = BaselineTransUNet().to(device)
    
    # Loss function and optimizer(Adam) load
    criterion = OccupancyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    #==========================
    # Checkpoint 설정
    #==========================
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    
    best_val_iou = 0.0 # 초기 IOU = 0.0
    start_epoch = 1
    resume = False   # 중단한 학습을 이어서 할지 여부
    
    # 중단한 학습을 이어서 하는 경우
    if resume:
        # 마지막 학습 상태가 저장된 checkpoint 경로
        last_ckpt_path = os.path.join(Config.checkpoint_dir, "last.pth")
        
        # 마지막 모델의 가중치, optimizer, epoch, loss 등 불러오기
        if os.path.exists(last_ckpt_path):
            loaded_epoch, best_val_iou = load_checkpoint(model, optimizer, last_ckpt_path, device)
            start_epoch = loaded_epoch + 1
            print(f"[INFO] Resuming training from epoch {start_epoch}")
            
    #============================
    # Training & Validation Loop
    #============================
    for epoch in range(start_epoch, Config.epochs + 1):
        print(f"\nEpoch [{epoch}/{Config.epochs}]")
        
        # 1. Training
        train_loss, train_iou, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # 2. Validation
        val_loss, val_iou, val_acc = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        print(f"[Epoch {epoch}] "
              f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Acc: {val_acc:.4f}")
        
        # 3. Checkpoint 저장
        save_checkpoint(
            model, optimizer, epoch, val_iou, 
            os.path.join(Config.checkpoint_dir, "last.pth")
        )
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            
            save_checkpoint(
                model, optimizer, epoch, best_val_iou, 
                os.path.join(Config.checkpoint_dir, "best.pth")
            )
            print("Best model updated!")
            
    print("\nAll Training Finished!")
