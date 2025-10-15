"""
디버깅 스크립트: 모델이 생성하는 액션을 자세히 분석
"""

import torch
import numpy as np
import gymnasium as gym
from metadrive.envs.top_down_env import TopDownMetaDrive
from ppo_metadrive3 import Actor, Critic, ImageVAE_CNN840Micro_LinearDec, ZValue, NNPolicy, obs_batch_to_tensor, deviceof

# 환경 등록
gym.register(id='MetaDrive-topdown', entry_point=TopDownMetaDrive, kwargs=dict(config={}))

def debug_model_actions():
    """모델이 생성하는 액션을 자세히 분석"""
    print("🔍 모델 액션 디버깅 시작...")
    
    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('ppo_metadrive_best_model.pth', map_location=device)
    
    actor = Actor(input_channels=5, z_dim=40).to(device)
    critic = Critic(input_channels=5, z_dim=40).to(device)
    vae_guide = ImageVAE_CNN840Micro_LinearDec(recon_activation=None).to(device)
    
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    vae_guide.load_state_dict(checkpoint['vae_state_dict'])
    
    actor.eval()
    critic.eval()
    vae_guide.eval()
    
    print("✅ 모델 로드 완료!")
    
    # 환경 생성
    env = gym.make('MetaDrive-topdown', config={
        'use_render': False,  # 렌더링 비활성화로 빠른 테스트
        'horizon': 20,
        'num_scenarios': 5
    })
    
    obs, info = env.reset()
    print(f"✅ 환경 리셋 완료!")
    print(f"관찰 공간: {obs.shape}, 범위: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # 정책 생성
    policy = NNPolicy(actor, vae=vae_guide, use_recon_for_policy=True)
    
    print("\n🎮 액션 분석 시작...")
    actions = []
    
    for i in range(20):
        # 모델에서 액션 생성
        action = policy(obs)
        actions.append(action)
        
        print(f"스텝 {i+1:2d}: throttle={action[0]:.6f}, steering={action[1]:.6f}")
        
        # 환경에서 스텝 실행
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"    에피소드 종료! (terminated={terminated}, truncated={truncated})")
            break
    
    env.close()
    
    # 액션 통계 분석
    actions = np.array(actions)
    throttle_actions = actions[:, 0]
    steering_actions = actions[:, 1]
    
    print(f"\n📊 액션 통계 분석:")
    print(f"Throttle (가속):")
    print(f"  평균: {np.mean(throttle_actions):.6f}")
    print(f"  표준편차: {np.std(throttle_actions):.6f}")
    print(f"  범위: [{np.min(throttle_actions):.6f}, {np.max(throttle_actions):.6f}]")
    print(f"  고유값 개수: {len(np.unique(throttle_actions))}")
    
    print(f"\nSteering (조향):")
    print(f"  평균: {np.mean(steering_actions):.6f}")
    print(f"  표준편차: {np.std(steering_actions):.6f}")
    print(f"  범위: [{np.min(steering_actions):.6f}, {np.max(steering_actions):.6f}]")
    print(f"  고유값 개수: {len(np.unique(steering_actions))}")
    
    # 문제 진단
    print(f"\n🔍 문제 진단:")
    if np.std(throttle_actions) < 0.01:
        print("❌ Throttle 액션이 거의 변하지 않음 - 자동차가 가속하지 않음")
    if np.std(steering_actions) < 0.01:
        print("❌ Steering 액션이 거의 변하지 않음 - 자동차가 방향을 바꾸지 않음")
    
    if np.mean(throttle_actions) < 0.1:
        print("❌ Throttle 평균값이 너무 낮음 - 자동차가 충분히 가속하지 않음")
    
    if abs(np.mean(steering_actions)) < 0.1:
        print("❌ Steering 평균값이 0에 가까움 - 자동차가 직진만 함")
    
    # 해결책 제안
    print(f"\n💡 해결책 제안:")
    print("1. 모델이 제대로 훈련되지 않았을 수 있습니다.")
    print("2. 더 많은 에피소드로 훈련이 필요할 수 있습니다.")
    print("3. 학습률이나 다른 하이퍼파라미터 조정이 필요할 수 있습니다.")
    print("4. 랜덤 액션과 비교해보세요:")
    
    # 랜덤 액션과 비교
    print(f"\n🎲 랜덤 액션 비교:")
    random_throttle = np.random.uniform(0, 1, 20)
    random_steering = np.random.uniform(-1, 1, 20)
    print(f"랜덤 Throttle 범위: [{np.min(random_throttle):.3f}, {np.max(random_throttle):.3f}]")
    print(f"랜덤 Steering 범위: [{np.min(random_steering):.3f}, {np.max(random_steering):.3f}]")

if __name__ == "__main__":
    debug_model_actions()
