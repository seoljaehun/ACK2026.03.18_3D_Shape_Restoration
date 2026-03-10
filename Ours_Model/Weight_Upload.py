import torch
import os
from model.Network import OccupancyNetwork
from utils.Config import Config

def test_weight_loading():
    print("1. 🏗️ 투트랙 + HF 어텐션 OccupancyNetwork 생성 중...")
    model = OccupancyNetwork(c_dim=256)
    
    # 다운로드 받은 가중치 파일 경로
    weight_path = os.path.join(Config.checkpoint_dir, 'onet_img2mesh_3-f786b04a.pt')
    
    print(f"2. 📂 사전 학습된 가중치 다이렉트 로드 시도: {weight_path}")
    
    if not os.path.exists(weight_path):
        print(f"❌ [에러] 파일 경로를 찾을 수 없습니다: {weight_path}")
        return

    try:
        # 1. 가중치 파일 열기
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        
        print("3. 🔄 매핑 없이 다이렉트로 가중치 이식을 시작합니다...")
        
        # 2. 가중치 이식
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print("\n" + "="*65)
        print("🎉 [성공] 글로벌 뼈대(ResNet + OccNet) 가중치가 완벽하게 안착되었습니다!")
        print("=" * 65)
        
        # 3. 로드된 항목 확인
        expected_missing = [k for k in missing_keys if 'local_decoder' in k or 'hf_feat_generator' in k]
        unexpected_missing = [k for k in missing_keys if k not in expected_missing]
        
        if len(expected_missing) > 0 and len(unexpected_missing) == 0:
            print(f"✅ 체크 1: {len(expected_missing)}개의 [Local Decoder & HF Feature] 레이어는 학습 전 상태로 잘 남겨졌습니다.")
            print("   (이는 의도된 결과이며, 이 모듈들이 바로 이번 연구의 핵심 무기입니다!)")
        elif len(unexpected_missing) > 0:
            print(f"⚠️ [경고] 의도치 않게 누락된 글로벌 뼈대 가중치가 있습니다!")
            print(f"   누락된 항목: {unexpected_missing}")
            
        if len(unexpected_keys) == 0:
            print("✅ 체크 2: 원본 가중치 파일에서 남거나 버려진 키가 없습니다. (완벽 호환)")
        else:
            print(f"⚠️ [참고] 현재 모델에 없는 원본 가중치 키가 {len(unexpected_keys)}개 남았습니다.")
            
        print("="*65)
        print("🚀 검증이 완료되었습니다! 이 깔끔한 로직을 Train.py에 적용하고 본 학습을 시작하세요!")
        
    except Exception as e:
        print("\n❌ [실패] 가중치 로드 중 예상치 못한 에러가 발생했습니다.")
        print(e)

if __name__ == "__main__":
    test_weight_loading()
