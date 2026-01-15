########################################
#  Preprocessed ShapeNet 데이터셋 로드  #
########################################

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ShapeNetDataset(Dataset):
    def __init__(self, data_root, split='Train', points_batch_size=2048):
        """
            데이터 셋 경로 설정 및 전처리 초기화
        Args:
            data_root (str): 데이터셋 최상위 경로
            split (str): 'Train', 'Val', 'Test'
            points_batch_size (int): 3D 포인트 샘플링 개수
        """
        # 인스턴스 변수 선언
        self.data_root = data_root
        self.split = split
        self.points_batch_size = points_batch_size
        self.pointcloud_samples = 100000
        
        # 이미지 크기 설정
        self.org_size = 137
        self.target_size = 224
        
        # 이미지 전처리
        # Resize: 137*137 -> 224*224 늘림
        # ToTensor: [0, 255] -> [0.0, 1.0]
        # Normalize: ImageNet 평균/표준편차로 정규화
        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # train/val/test 폴더 경로 생성
        # 예: data_root/Train
        target_folder = os.path.join(data_root, split)
        
        # 객체 폴더 리스트 로드
        self.items = self._get_files_from_folder(target_folder)
        
        # 데이터가 없으면 경고 출력
        if len(self.items) == 0:
            print(f"경고: '{target_folder}' 경로에서 데이터를 하나도 못 찾았습니다. 경로를 확인하세요.")
        else:
            print(f"[{split}] 총 {len(self.items)}개의 객체 로드 완료.")

    def __getitem__(self, idx):
        """
            1개의 데이터 샘플 로드
        Args:
            idx: dataset index number

        Returns:
            sample: 전처리된 데이터 딕셔너리
        """
        # 폴더 리스트 중 n번째(idx) 객체 선택
        item_path = self.items[idx]
        
        # 뷰(View) 선택
        # Train 폴더일 때는 0~23번 중 하나만 랜덤으로 선택,
        # Val, Test는 0번 뷰로 고정
        if self.split == 'Train':
            view_idx = np.random.randint(0, 24)
        else:
            view_idx = 0 
         
        #=========================================   
        # 2D 이미지 로드 (.jpg)
        #=========================================
        
        # ex) view_idx = 5인 경우 '005.jpg' 파일 로드
        img_filename = f"{view_idx:03d}.jpg" 
        img_path = os.path.join(item_path, 'img_choy2016', img_filename)
        
        # n번째 뷰 이미지 open, 파일이 없으면 0번 뷰로 대체
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            view_idx = 0
            img_path = os.path.join(item_path, 'img_choy2016', "000.jpg")
            image = Image.open(img_path).convert('RGB')
        
        # 이미지 전처리 적용
        image = self.transform(image)

        #=========================================
        # 카메라 파라미터 (.npz)
        #=========================================
        camera_path = os.path.join(item_path, 'img_choy2016', 'cameras.npz')
        camera_data = np.load(camera_path)
        
        # 스케일링 팩터(224 / 137 = 1.635...)
        scale_factor = self.target_size / self.org_size
        
        # 카메라 내부 파라미터 로드
        K = camera_data[f'camera_mat_{view_idx}'].astype(np.float32).copy()
        K[:2, :] *= scale_factor  # 스케일링 적용
        
        # 카메라 외부 파라미터 로드 (스케일링 x)
        RT = camera_data[f'world_mat_{view_idx}'].astype(np.float32)

        #=========================================
        # 3D Points(정답 데이터, .npz)
        #=========================================
        points_path = os.path.join(item_path, 'points.npz')
        point_data = np.load(points_path)
        
        # Point: 3D 공간 상 무작위 점들의 좌표(x, y, z)
        # Occupancy: 각 점이 물체 내부(1)인지 외부(0)인지 여부
        points = point_data['points']
        occupancies = np.unpackbits(point_data['occupancies'])[:points.shape[0]]

        # 안과 밖 point 분리
        inside_idxs = np.where(occupancies == 1)[0]
        outside_idxs = np.where(occupancies == 0)[0]

        # 개수 설정 (1:1 비율)
        # 100000개의 정답 데이터 중 2048개 무작위 선택하여 사용(다 사용하면 메모리 부족)
        n_out = self.points_batch_size // 2
        n_in = self.points_batch_size - n_out
        
        # 내부 점 샘플링 (부족하면 중복 허용)
        if len(inside_idxs) > 0:
            rand_in = np.random.choice(inside_idxs, n_in, replace=(len(inside_idxs) < n_in))
        else:
            # 내부 점이 아예 없을 때 (외부 점에서만 추출)
            rand_in = np.random.choice(outside_idxs, n_in, replace=True)
        
        # 외부 점 샘플링 (부족하면 중복 허용)
        rand_out = np.random.choice(outside_idxs, n_out, replace=(len(outside_idxs) < n_out))

        # 내부/외부 점 합치고 섞기
        idxs = np.concatenate([rand_in, rand_out])
        np.random.shuffle(idxs)
        
        # 최종 데이터 추출
        points = points[idxs].astype(np.float32)
        occupancies = occupancies[idxs].astype(np.float32)
        
        #=========================================
        # 3D Pointcloud (평가지표 데이터, .npz)
        #=========================================
        pcl_path = os.path.join(item_path, 'pointcloud.npz')
        pcl_dict = np.load(pcl_path)
        
        # pcl: 객체 표면의 (x, y, z) 좌표
        # normals: 해당 점의 (nx, ny, nz) 법선 벡터
        pcl = pcl_dict['points'].astype(np.float32)
        normals = pcl_dict['normals'].astype(np.float32)
        
        # 현재 점 개수 확인
        n_pcl = pcl.shape[0]
        
        # pointcloud 개수를 통일 (10만개)
        if n_pcl >= self.pointcloud_samples:
            idx_pcl = np.random.choice(n_pcl, self.pointcloud_samples, replace=False)
        else:
            idx_pcl = np.random.choice(n_pcl, self.pointcloud_samples, replace=True)
        
        # 최종 데이터 추출  
        pcl = pcl[idx_pcl]
        normals = normals[idx_pcl]
                
        # 딕셔너리 형태로 반환
        return {
            'img': image,
            'points': points,
            'occupancies': occupancies,
            'inputs_k': K,
            'inputs_rt': RT,
            'pointcloud': pcl,
            'normals': normals
        }

    def __len__(self):
        return len(self.items)

    #=========================================
    # 객체 폴더 리스트 반환
    #=========================================
    def _get_files_from_folder(self, target_folder):
        items = []
        
        # Train/Val/Test 폴더가 실제로 있는지 확인
        if not os.path.exists(target_folder):
            return items

        # Train/Val/Test 폴더 안에 있는 모든 객체 폴더 리스트 가져오기
        objs = os.listdir(target_folder)
        
        for obj in objs:
            # obj는 "Bench (1450)", "Car (1)" 같은 객체 폴더 이름
            obj_path = os.path.join(target_folder, obj)
            
            # 파일이 아니라 폴더인지 확인
            if not os.path.isdir(obj_path): continue
            
            # 유효성 검사: 그 안에 'img_choy2016/cameras.npz'가 있는지 확인
            # 경로: .../Train/Bench (1450)/img_choy2016/cameras.npz
            if os.path.exists(os.path.join(obj_path, 'img_choy2016', 'cameras.npz')):
                 items.append(obj_path)
                 
        return items
