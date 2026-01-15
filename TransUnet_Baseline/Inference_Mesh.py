################################################
#  inference mesh                              #
# : create mesh from occupancy network output  #
################################################

import os
import torch
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
from PIL import Image
from torchvision import transforms

# Custom Modules
from model.Network import BaselineTransUNet
from utils.Config import Config

#=======================
# Inference Settings
#=======================

# 객체 이미지 경로 (입력 이미지)
BASE_DIR = r"D:\Dataset\3D_Shape_Reconstruction\Train\Bench (1)\img_choy2016"
TARGET_VIEW_IDX = 0

# 체크포인트 경로 (학습된 모델)
CHECKPOINT_PATH = os.path.join(Config.checkpoint_dir, 'best.pth')

# 3D 해상도 (높을수록 표면이 매끄럽지만 느려짐)
RESOLUTION = 128 

# 배치 청크 크기 (한 번에 모델에 넣을 점의 개수, VRAM에 맞춰 조절)
POINTS_BATCH_SIZE = 4096

#=======================
# Mesh Generator Class
#=======================
class MeshGenerator:
    def __init__(self, model, device, resolution=128, threshold=0.5):
        """
            3D 좌표 그리드 생성 및 초기화
        Args:
            model: 학습이 완료된 TransUNet 모델
            device: GPU
            resolution: 해상도(= 128)
            threshold: 물체 존재 확률 임계값(= 0.5)
        """
        self.model = model
        self.device = device
        self.resolution = resolution
        self.threshold = threshold
        
        # 가상 공간 크기 설정(-0.55 ~ 0.55)
        box_size = 1.1 
        # 1차원 눈금자 생성 (128개의 눈금)
        coordinates = np.linspace(-box_size/2, box_size/2, resolution)
        
        # 3D 공간 전체 좌표 생성 -> (128, 128, 128, 3)
        self.grid_points = np.stack(np.meshgrid(coordinates, coordinates, coordinates, indexing='ij'), axis=-1)
        self.grid_points = self.grid_points.reshape(-1, 3).astype(np.float32) # (2097152, 3)
        
    def generate_mesh(self, img, k, rt):
        """
            3D 메쉬 생성
        Args:
            img: 입력 이미지
            k: 카메라 내부 행렬
            rt: 카메라 외부 행렬

        Returns:
            verts: 메쉬 정점 좌표
            faces: 메쉬 삼각형 면
            volume: 3D 확률 볼륨
        """
        # 평가모드 설정
        self.model.eval()
        
        num_points = self.grid_points.shape[0]      # 3D 공간 전체 좌표
        all_logits = []     # 모든 점의 예측 결과 저장
        
        with torch.no_grad():
            # 입력 텐서 GPU로 이동
            img_tensor = img.unsqueeze(0).to(self.device)
            k_tensor = k.unsqueeze(0).to(self.device)
            rt_tensor = rt.unsqueeze(0).to(self.device)
            
            print(f"Inferencing {num_points} points with resolution {self.resolution}^3...")
            
            # 점들을 청크 단위로 나누어 처리 (4096개)
            for i in range(0, num_points, POINTS_BATCH_SIZE):
                # 전체 3D 점 좌표에서 청크 단위로 추출
                chunk_points = self.grid_points[i : i + POINTS_BATCH_SIZE]
                
                # 청크 점 좌표 텐서 생성 및 GPU로 이동
                chunk_tensor = torch.from_numpy(chunk_points).unsqueeze(0).to(self.device)
                
                # 모델에게 예측 요청 후 결과 수집
                preds = self.model(img_tensor, chunk_tensor, k_tensor, rt_tensor)
                all_logits.append(preds.cpu().numpy().flatten())
        
        # 모든 청크의 결과를 하나로 결합        
        all_logits = np.concatenate(all_logits)
        
        # logit 결과를 시그모이드 함수로 확률 변환
        probs = 1 / (1 + np.exp(-all_logits))
        
        # 1D 확률 배열을 3D 볼륨으로 재구성 -> 확률이 담긴 3D 공간 데이터
        volume = probs.reshape(self.resolution, self.resolution, self.resolution)
        
        print("Reconstructing Mesh using Marching Cubes...")
        
        try:    # Marching Cubes 알고리즘 적용
            
            # 임계값을 기준으로 메쉬 생성
            # verts: 점들의 좌표, faces: 점들을 잇는 삼각형 면
            # normals: 법선 벡터, values: 메쉬 값
            verts, faces, normals, values = marching_cubes(volume, level=self.threshold)
            box_size = 1.1
            
            # marching 좌표 -> (-0.55 ~ 0.55) 범위로 정규화
            verts = verts / (self.resolution - 1) * box_size - (box_size / 2)
            
            # 생성된 점, 면, 3D 볼륨 데이터 반환
            return verts, faces, volume
        
        except RuntimeError: # 메쉬 생성 실패 시
            print("Mesh 생성 실패: 물체가 감지되지 않았거나 너무 작습니다.")
            
            return None, None, volume
        
def load_data_by_view_index(base_dir, view_idx, device):
    """
        base_dir와 view_idx를 결합하여 데이터 로드
    Args:
        base_dir: 이미지 및 카메라 정보가 담긴 폴더 경로
        view_idx: 불러올 뷰 인덱스 (예: 5)
        device: GPU
    """
    # 이미지 경로 생성
    img_filename = f"{view_idx:03d}.jpg"
    img_path = os.path.join(base_dir, img_filename)
        
    print(f"Loading image: {img_path}")
    
    # 이미지 로드 및 전처리
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    
    # 카메라 파라미터 로드
    cam_path = os.path.join(base_dir, 'cameras.npz')
    cam_data = np.load(cam_path)
    
    # 스케일링 팩터
    scale_factor = 224 / 137
    
    # 카메라 내부 및 외부 행렬 추출
    K = cam_data[f'camera_mat_{view_idx}'].astype(np.float32)
    K[:2, :] *= scale_factor
    RT = cam_data[f'world_mat_{view_idx}'].astype(np.float32)
    
    # 텐서로 변환하여 반환
    return image, torch.from_numpy(K), torch.from_numpy(RT)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model = BaselineTransUNet().to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        # 체크포인트(best) 로드
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Checkpoint loaded (Best IoU: {checkpoint.get('val_iou', 0):.4f})")
    else:
        print("체크포인트가 없습니다!")
        exit()
        
    # 데이터 로드
    try:
        # 입력 이미지, 카메라 내부 및 외부 파라미터
        img_tensor, k_tensor, rt_tensor = load_data_by_view_index(BASE_DIR, TARGET_VIEW_IDX, device)
    
    except Exception as e:
        print(f"데이터 로드 에러: {e}")
        exit()
        
    # Mesh 생성
    generator = MeshGenerator(model, device, resolution=RESOLUTION)
    verts, faces, volume = generator.generate_mesh(img_tensor, k_tensor, rt_tensor)
    
    # 결과 시각화 및 저장
    if verts is not None:
        print("Visualizing Mesh...")
        # 빈 mesh 객체 생성
        mesh = o3d.geometry.TriangleMesh()
        
        # 정점 및 삼각형 면 설정
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # 면의 각도 계산 후 빛 반사 시뮬
        mesh.compute_vertex_normals()
        
        # 색깔 지정 (회색)
        mesh.paint_uniform_color([0.7, 0.7, 0.7]) 
        
        # 3D mesh 결과 저장
        save_dir = "./result"
        os.makedirs(save_dir, exist_ok=True)
        
        save_name = os.path.join(save_dir, f"result_view_{TARGET_VIEW_IDX:03d}.ply")
        o3d.io.write_triangle_mesh(save_name, mesh)
        
        print(f"Mesh saved to: {os.path.abspath(save_name)}")
        
        # 결과 시각화
        o3d.visualization.draw_geometries([mesh], 
                                          window_name=f"Reconstructed Mesh (View {TARGET_VIEW_IDX})", 
                                          width=800, height=600)
    else:
        print("복원 실패")