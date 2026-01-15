###########################################
#  Config                                 #
# : directory paths and Hyper parameters  #
###########################################

import os
import random
import numpy as np
import torch

class Config:
    #=====================
    # Dataset Path
    #=====================
    root_dir = r"D:\Dataset\3D_Shape_Reconstruction"
    
    #=====================
    # Checkpoint Path
    #=====================
    _utils_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_utils_dir)
    checkpoint_dir = os.path.join(_project_root, 'checkpoint') # 학습 중 모델의 가중치 저장 폴더 경로
    
    #=====================
    # Training Settings
    #=====================
    epochs = 50                 # 학습 반복 횟수
    batch_size = 8              # 배치 사이즈
    learning_rate = 1e-4        # optimizer 학습률
    num_workers = 8             # DataLoader에서 데이터 로드할 병렬 스레드 수
    
    #=====================
    # Device
    #=====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 -> cuda 설정
    
    #=====================
    # 시드 고정 함수
    #=====================
    SEED = 2026
    
    # 재현성을 위해 시드 고정
    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def seed_worker(worker_id):
        # 파이토치 시드 + 워커 ID를 조합해서 새로운 시드를 만듦
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)