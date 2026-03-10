import os
import torch
import trimesh
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.measure import marching_cubes
from collections import defaultdict

from model.Network import OccupancyNetwork
from dataset.Dataset import ShapeNetDataset
from utils.Config import Config

# ==========================================
# 평가 지표 계산 함수
# ==========================================
# 1. Volume IoU
def compute_iou(preds, targets, prob_threshold=0.85):
    """
        Volumetric IoU 계산
    Args:
        preds: 예측 Occupancy
        targets: 정답 Occupancy
        prob_threshold: 표면 임계값
    """
    # Logit -> 0~1 확률값 변환 및 이진화
    probs = torch.sigmoid(preds)
    preds_binary = (probs > prob_threshold).float()
    
    intersection = (preds_binary * targets).sum(dim=1)          # 교집합 (논리곱 AND)
    union = ((preds_binary + targets) > 0).float().sum(dim=1)   # 합집합 (논리합 OR)
    
    # IOU 계산 (분모 0 방지)
    iou = torch.where(union == 0, torch.tensor(1.0, device=preds.device), intersection / union)
    
    return iou.mean().item()    # 배치 내의 IOU 평균값

# 2. Chamfer Distance & F-Score
def compute_cd_and_fscore(pred_verts, gt_pointcloud, device, threshold=0.01, chunk_size=10000):
    """ 
        Chamfer Distance 및 F-Score 계산 (Surface Sampling 기반)
    Args:
        pred_verts: 예측한 형상의 표면 점
        gt_pointcloud: 정답의 표면 점
        device: 연산 장치(GPU)
        threshold: F-score 채점 허용 오차
        chunk_size: 분할 계산 크기
    """
    # 형상 복원을 못했을 시 CD, F-sore 점수 반환 (최하점)
    if len(pred_verts) == 0 or len(gt_pointcloud) == 0:
        return 6.0, 0.0 
    
    # pytorch 텐서로 변환
    pred_tensor = torch.tensor(pred_verts, dtype=torch.float32, device=device)  # (N, 3)
    gt_tensor = torch.tensor(gt_pointcloud, dtype=torch.float32, device=device) # (M, 3)
    
    # 1. Pred -> GT 최단 거리 (Accuracy)
    min_dist_pred_to_gt = []
    
    for i in range(0, len(pred_tensor), chunk_size):
        chunk = pred_tensor[i:i+chunk_size]
        
        # 거리 계산 - 유클리디안 거리 행렬 연산
        dists = torch.cdist(chunk.unsqueeze(0), gt_tensor.unsqueeze(0)).squeeze(0)
        
        # 가장 짧은 거리만 받아옴
        min_dist, _ = torch.min(dists, dim=1)
        min_dist_pred_to_gt.append(min_dist)
        
    min_dist_pred_to_gt = torch.cat(min_dist_pred_to_gt) # (N,)

    # 2. GT -> Pred 최단 거리 (Completeness)
    min_dist_gt_to_pred = []
    
    for i in range(0, len(gt_tensor), chunk_size):
        chunk = gt_tensor[i:i+chunk_size]
        
        # 거리 계산 - 유클리디안 거리 행렬 연산
        dists = torch.cdist(chunk.unsqueeze(0), pred_tensor.unsqueeze(0)).squeeze(0)
        
        # 가장 짧은 거리만 받아옴
        min_dist, _ = torch.min(dists, dim=1)
        min_dist_gt_to_pred.append(min_dist)
        
    min_dist_gt_to_pred = torch.cat(min_dist_gt_to_pred) # (M,)
    
    # Chamfer-L1
    accuracy = min_dist_pred_to_gt.mean().item()        # Pred -> GT 최단 평균 거리
    completeness = min_dist_gt_to_pred.mean().item()    # GT -> Pred 최단 평균 거리
    cd = 0.5 * (accuracy + completeness)
    
    # F-Score (@ 1%): 임계값 안쪽으로 들어오는 점들을 True
    precision = (min_dist_pred_to_gt < threshold).float().mean().item()
    recall = (min_dist_gt_to_pred < threshold).float().mean().item()
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return cd, f_score

# ==========================================
# 메쉬 추출 함수
# ==========================================

# 1. mesh 생성
def extract_mesh_verts(model, batch, features, device, resolution=128, prob_threshold=0.85):
    """ 
        모델의 전체 forward 로직을 사용하여 고해상도 표면 추출
    Args:
        model: 3D 형상 복원 모델
        batch: 데이터 배치
        features: 인코더 특징
        device: 연산 장치
        resolution: 해상도
        prob_threshold: 표면 임계값
    """
    box_size = 1.1 
    
    # -0.55 ~ 0.55 범위의 resolution개수로 쪼갠 좌표 생성(하나의 축)
    coords = torch.linspace(-box_size/2, box_size/2, resolution, device=device)
    
    # coords가 하나의 축으로 입체적인 3D 격자 생성
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
    # 3D 격자를 리스트 혀앹로 변환
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)
    
    # 배치 데이터 GPU 이동
    scale = batch['scale'].to(device)
    loc = batch['loc'].to(device)
    camera_k = batch['inputs_k'].to(device)
    camera_rt = batch['inputs_rt'].to(device)
    
    # 청크 사이즈로 나눠서 계산
    num_points = grid_points.shape[0]
    chunk_size = 32768
    all_probs = []
    
    # Encoder의 특징
    global_feat, local_feat, hf_features = features
    
    with torch.no_grad():
        for i in range(0, num_points, chunk_size):
            p_chunk = grid_points[i:i+chunk_size].unsqueeze(0)
            
            # 1. Global 로짓 (뼈대)
            g_logits = model.decoder(p_chunk, global_feat)
            
            # 2. Local 로짓 (디테일 잔차)
            l_logits = model.local_decoder(
                points=p_chunk, 
                local_features=local_feat, 
                global_feat=global_feat, 
                hf_feat=hf_features,
                scale=scale, 
                loc=loc, 
                camera_k=camera_k, 
                camera_rt=camera_rt
            )
            
            # 3. 합산 및 확률 변환
            probs = torch.sigmoid(g_logits + l_logits)
            all_probs.append(probs.cpu().numpy().flatten())
    
    # 쪼개진 모든 확률 값들을 다시 3D 형태로 변환
    volume = np.concatenate(all_probs).reshape(resolution, resolution, resolution)
    
    try:
        # 지정한 임계값으로 메쉬 껍질 추출(말징 큐브)
        verts, faces, _, _ = marching_cubes(volume, level=prob_threshold)
        
        # 스케일 정렬 -> -0.55 ~ 0.55
        verts = verts / (resolution - 1) * box_size - (box_size / 2)
        
        # 꼭짓점(verts)과 면(faces)을 엮어 3D 메쉬 객체로 만듬
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        if len(mesh.vertices) > 0:
            # mesh 표면 위에 10만개의 점을 고르게 샘플링
            sampled_points, _ = trimesh.sample.sample_surface(mesh, 100000)
            
            return sampled_points
        
        return np.array([])
    
    except Exception:
        return np.array([])

# ==========================================
# 메인 평가 루프
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 테스트 데이터셋 로드
    test_dataset = ShapeNetDataset(data_root=Config.root_dir, split="Test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 모델 로드
    model = OccupancyNetwork().to(device)
    checkpoint_path = os.path.join(Config.checkpoint_dir, 'last.pth')
    
    # 저장된 가중치 로드
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model Loaded! (Val IoU: {checkpoint.get('val_iou', 0):.4f})")
    else:
        print("Checkpoint not found. Exiting.")
        exit()
    
    # 평가모드
    model.eval()
    
    # 변수 초기화
    total_iou, total_cd, total_fscore = 0.0, 0.0, 0.0
    num_samples = len(test_loader)
    category_metrics = defaultdict(lambda: {'iou': 0.0, 'cd': 0.0, 'fscore': 0.0, 'count': 0})
    
    loop = tqdm(test_loader, total=num_samples, desc="Evaluating")

    with torch.no_grad():
        for batch in loop:
            # 배치 데이터 GPU 이동
            occupancies = batch['occupancies'].to(device).float()
            gt_pointcloud = batch['pointcloud'].squeeze(0).cpu().numpy()
            images = batch['img'].to(device)
            points = batch['points'].to(device)
            scale = batch['scale'].to(device)
            loc = batch['loc'].to(device)
            camera_k = batch['inputs_k'].to(device)
            camera_rt = batch['inputs_rt'].to(device)
            category_name = batch['category'][0]
            
            # Encoder 특징 추출
            global_feat, local_feat, hf_features = model.encoder(images)
            features = (global_feat, local_feat, hf_features)
            
            # Decoder 형상 복원
            global_logits = model.decoder(points, global_feat)
            local_logits = model.local_decoder(
                points=points,
                local_features=local_feat,
                global_feat=global_feat,
                hf_feat=hf_features,
                scale=scale,
                loc=loc,
                camera_k=camera_k,
                camera_rt=camera_rt
            )
            
            # 최종 형상 재구성
            final_preds = global_logits + local_logits
            
            # IoU
            iou = compute_iou(final_preds, occupancies)
            
            # CD & F-Score
            pred_verts = extract_mesh_verts(model, batch, features, device, resolution=128)
            cd, f_score = compute_cd_and_fscore(pred_verts, gt_pointcloud, device)
            
            # 평가지표 합산
            total_iou += iou
            total_cd += cd
            total_fscore += f_score
            
            category_metrics[category_name]['iou'] += iou
            category_metrics[category_name]['cd'] += cd
            category_metrics[category_name]['fscore'] += f_score
            category_metrics[category_name]['count'] += 1
            
            loop.set_postfix(IoU=f"{iou:.4f}", CD=f"{cd:.5f}")

    # 최종 결과 출력
    print("\n" + "="*60)
    print(f" [Category-wise Test Results] ")
    print("="*60)
    
    # 카테고리별 출력
    for cat, metrics in sorted(category_metrics.items()): # 알파벳 순 정렬
        count = metrics['count']
        if count > 0:
            avg_iou = metrics['iou'] / count
            avg_cd = metrics['cd'] / count
            avg_fscore = metrics['fscore'] / count
            print(f" {cat:<10} | Samples: {count:<4} | IoU: {avg_iou:.4f} | CD: {avg_cd:.5f} | F-Score: {avg_fscore:.4f}")

    print("-" * 60)
    # 전체 평균 출력
    print(f" TOTAL AVG  | Samples: {num_samples:<4} | IoU: {total_iou/num_samples:.4f} | CD: {total_cd/num_samples:.5f} | F-Score: {total_fscore/num_samples:.4f}")
    print("="*60)