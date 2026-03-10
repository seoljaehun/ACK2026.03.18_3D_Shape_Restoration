###########################################
#  Config                                 #
# : directory paths and Hyper parameters  #
###########################################

import os
import torch

class Config:
    #=====================
    # Dataset Path
    #=====================
    root_dir = r"C:\SJH\3D_Shape_Reconstruction"
    
    #=====================
    # Checkpoint Path
    #=====================
    _utils_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_utils_dir)
    checkpoint_dir = os.path.join(_project_root, 'checkpoint') # 학습 중 모델의 가중치 저장 폴더 경로
    
    #=====================
    # Training Settings
    #=====================
    epochs = 100                 # 학습 반복 횟수
    batch_size = 16              # 배치 사이즈
    learning_rate = 1e-4        # optimizer 학습률
    num_workers = 4             # DataLoader에서 데이터 로드할 병렬 스레드 수
    
    #=====================
    # Device
    #=====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 -> cuda 설정
    