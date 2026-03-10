import os
import torch
from model.Network import OccupancyNetwork  # 경로에 맞게 수정
from utils.Config import Config

def test_weight_loading():
    print("1. OccupancyNetwork 뼈대 생성 중...")
    model = OccupancyNetwork()
    
    # 다운로드 받은 가중치 파일 경로
    weight_path = os.path.join(Config.checkpoint_dir, 'onet_img2mesh_3-f786b04a.pt')
    
    print(f"2. 사전 학습된 가중치 로드 시도: {weight_path}")
    try:
        # 가중치 파일 열기
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        
        # 원본 OccNet 가중치는 'model' 딕셔너리 안에 들어있음
        state_dict = checkpoint.get('model', checkpoint)
        
        # strict=True: 단 하나의 키 이름이나 차원 크기라도 다르면 즉시 에러 발생!
        model.load_state_dict(state_dict, strict=True)
        
        print("\n" + "="*50)
        print("🎉 [대성공] 매핑 코드 없이 가중치가 100% 완벽하게 다이렉트로 로드되었습니다! 🎉")
        print("="*50)
        
    except Exception as e:
        print("\n❌ [실패] 가중치 매칭에 실패했습니다. 키 이름이나 차원을 다시 확인하세요.")
        print(e)

if __name__ == "__main__":
    test_weight_loading()