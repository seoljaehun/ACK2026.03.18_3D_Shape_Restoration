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
def compute_iou(preds, targets, prob_threshold=0.20):
    """
        Volumetric IoU 계산
    Args:
        preds: 예측 Occupancy
        targets: 정답 Occupancy
        prob_threshold: 표면 임계값
    """
    probs = torch.sigmoid(preds)
    preds_binary = (probs > prob_threshold).float()
    
    intersection = (preds_binary * targets).sum(dim=1)
    union = ((preds_binary + targets) > 0).float().sum(dim=1)
    
    iou = torch.where(union == 0, torch.tensor(1.0, device=preds.device), intersection / union)
    return iou.mean().item()

# 2. Chamfer Distance & F-Score
def compute_cd_and_fscore(pred_verts, gt_pointcloud, device, threshold=0.01, chunk_size=10000):
    """ 
        Chamfer Distance 및 F-Score 계산 (Surface Sampling 기반)
    """
    if len(pred_verts) == 0 or len(gt_pointcloud) == 0:
        return 6.0, 0.0 
    
    pred_tensor = torch.tensor(pred_verts, dtype=torch.float32, device=device)
    gt_tensor = torch.tensor(gt_pointcloud, dtype=torch.float32, device=device)
    
    # 1. Pred -> GT 최단 거리
    min_dist_pred_to_gt = []
    for i in range(0, len(pred_tensor), chunk_size):
        chunk = pred_tensor[i:i+chunk_size]
        dists = torch.cdist(chunk.unsqueeze(0), gt_tensor.unsqueeze(0)).squeeze(0)
        min_dist, _ = torch.min(dists, dim=1)
        min_dist_pred_to_gt.append(min_dist)
    min_dist_pred_to_gt = torch.cat(min_dist_pred_to_gt)

    # 2. GT -> Pred 최단 거리
    min_dist_gt_to_pred = []
    for i in range(0, len(gt_tensor), chunk_size):
        chunk = gt_tensor[i:i+chunk_size]
        dists = torch.cdist(chunk.unsqueeze(0), pred_tensor.unsqueeze(0)).squeeze(0)
        min_dist, _ = torch.min(dists, dim=1)
        min_dist_gt_to_pred.append(min_dist)
    min_dist_gt_to_pred = torch.cat(min_dist_gt_to_pred)
    
    # Chamfer-L1
    accuracy = min_dist_pred_to_gt.mean().item()
    completeness = min_dist_gt_to_pred.mean().item()
    cd = 0.5 * (accuracy + completeness)
    
    # F-Score (@ 1%)
    precision = (min_dist_pred_to_gt < threshold).float().mean().item()
    recall = (min_dist_gt_to_pred < threshold).float().mean().item()
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return cd, f_score

# ==========================================
# 메쉬 추출 함수 (Baseline 전용)
# ==========================================
def extract_mesh_verts(model, global_feat, device, resolution=128, prob_threshold=0.20):
    box_size = 1.1 
    coords = torch.linspace(-box_size/2, box_size/2, resolution, device=device)
    
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)
    
    num_points = grid_points.shape[0]
    chunk_size = 32768 
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, num_points, chunk_size):
            p_chunk = grid_points[i:i+chunk_size].unsqueeze(0)
            
            logits = model.decoder(p_chunk, global_feat)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy().flatten())
    
    volume = np.concatenate(all_probs).reshape(resolution, resolution, resolution)
    
    try:
        verts, faces, _, _ = marching_cubes(volume, level=prob_threshold)
        verts = verts / (resolution - 1) * box_size - (box_size / 2)
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        if len(mesh.vertices) > 0:
            sampled_points, _ = trimesh.sample.sample_surface(mesh, 100000)
            
            return sampled_points
        
        return np.array([])

    except Exception:
        return np.array([])

# ==========================================
# 메인 평가 루프 (Baseline 전용)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_dataset = ShapeNetDataset(data_root=Config.root_dir, split="Test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = OccupancyNetwork().to(device)
    
    # ==========================================
    # 사전 훈련된 원본 가중치 로드
    # ==========================================
    checkpoint_path = os.path.join(Config.checkpoint_dir, 'onet_img2mesh_3-f786b04a.pt')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict, strict=True)
        
        print(" Baseline (OccNet) Pretrained Weight Loaded Successfully!")
        
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Exiting.")
        exit()
    
    model.eval()
    
    total_iou, total_cd, total_fscore = 0.0, 0.0, 0.0
    num_samples = len(test_loader)
    category_metrics = defaultdict(lambda: {'iou': 0.0, 'cd': 0.0, 'fscore': 0.0, 'count': 0})
    
    loop = tqdm(test_loader, total=num_samples, desc="Evaluating Baseline")

    with torch.no_grad():
        for batch in loop:
            occupancies = batch['occupancies'].to(device).float()
            gt_pointcloud = batch['pointcloud'].squeeze(0).cpu().numpy()
            images = batch['img'].to(device)
            points = batch['points'].to(device)
            category_name = batch['category'][0]
            
            global_feat = model.encoder(images)
            
            final_preds = model.decoder(points, global_feat)
            
            iou = compute_iou(final_preds, occupancies)
            
            pred_verts = extract_mesh_verts(model, global_feat, device, resolution=128)
            cd, f_score = compute_cd_and_fscore(pred_verts, gt_pointcloud, device)
            
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
    print(f" [Baseline (OccNet) Category-wise Test Results] ")
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