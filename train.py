import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from RJSPEnv.Env import RJSPEnv
from stable_baselines3 import A2C, DQN, TD3, PPO
from sb3_contrib import MaskablePPO, TRPO, RecurrentPPO, ARS, QRDQN
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

def train_and_evaluate(selected_algo, total_steps=1000000):
    # 환경 설정
    instance = "5x5"
    machine_config_path = f'instances/Machines/v0-{instance}.json'
    job_config_path = f'instances/Jobs/v0-{instance}-5.json'
    job_repeats_params = [(3, 1)] * 5  # 각 Job의 반복 횟수 (평균, 표준편차)

    # 훈련 환경 생성
    env = RJSPEnv(
        machine_config_path=machine_config_path,
        job_config_path=job_config_path,
        job_repeats_params=job_repeats_params,
        max_time=50
    )

    # 고정된 작업 횟수를 사용하는 테스트 환경 생성
    test_env = RJSPEnv(
        machine_config_path=machine_config_path,
        job_config_path=job_config_path,
        job_repeats_params=job_repeats_params,  # 작업 횟수 고정
        max_time=50,
        test_mode=True
    )

    # 알고리즘 리스트와 각 알고리즘에 적합한 정책 지정
    algorithms = {
        "MaskablePPO": MaskablePPO,
        "A2C": A2C,
        "DQN": DQN,
        "TRPO": TRPO,
        "RecurrentPPO": RecurrentPPO,
        "QRDQN": QRDQN,
        "PPO": PPO
    }

    # 선택한 알고리즘이 유효한지 확인
    if selected_algo not in algorithms:
        raise ValueError(f"알 수 없는 알고리즘: {selected_algo}. 사용 가능한 알고리즘: {list(algorithms.keys())}")

    algo_class = algorithms[selected_algo]

    # 결과를 저장할 리스트 초기화
    results = []

    # 각 알고리즘에 대해 학습 및 모델 저장
    print(f"\n==========================")
    print(f"{selected_algo} 알고리즘으로 학습 시작...")
    print("==========================\n")

    # 정책 설정
    if selected_algo in ["MaskablePPO", "A2C", "DQN", "TRPO", "QRDQN", "PPO"]:
        policy = "MultiInputPolicy"
    elif selected_algo == "RecurrentPPO":
        policy = "MultiInputLstmPolicy"
    else:
        policy = "MlpPolicy"

    # 콜백 설정
    # 모든 알고리즘에 공통으로 사용할 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=total_steps//10,  # total_steps에 비례하여 조정 가능
        save_path='./models/',
        name_prefix=f'model_checkpoint_{selected_algo}'
    )

    # MaskablePPO는 MaskableEvalCallback 사용
    if selected_algo == "MaskablePPO":
        eval_callback = MaskableEvalCallback(
            eval_env=test_env,
            best_model_save_path=f'./models/{selected_algo}_best',
            log_path=f'./logs/{selected_algo}_eval',
            eval_freq=total_steps//10,  # total_steps에 비례하여 조정 가능
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )
        callbacks = CallbackList([checkpoint_callback, eval_callback])
    else:
        # 다른 알고리즘들은 CheckpointCallback만 사용
        callbacks = checkpoint_callback

    # 모델 초기화
    model = algo_class(
        policy=policy,
        env=env,
        verbose=1,
        tensorboard_log=f"./tensorboard/{selected_algo}"
    )

    # 총 total_steps 타임스텝 학습
    batch_size = total_steps // 10  # 학습 단위 설정
    num_batches = total_steps // batch_size
    for step in range(1, num_batches + 1):
        current_step = step * batch_size
        print(f"{selected_algo} 알고리즘: {current_step} 타임스텝 학습 중...")
        model.learn(
            total_timesteps=batch_size,
            reset_num_timesteps=False,
            callback=callbacks
        )
        print(f"{selected_algo} 알고리즘: {current_step} 타임스텝 학습 완료.\n")

    # 최종 모델 저장
    model.save(f"./models/{selected_algo}_final")
    print(f"{selected_algo} 모델이 최종적으로 저장되었습니다.\n")

    # 모델 평가
    num_episodes = 10
    total_rewards = 0.0

    print(f"{selected_algo} 알고리즘 테스트 중...")
    for episode in range(num_episodes):
        obs, _ = test_env.reset()

        done = False
        episode_reward = 0.0

        while not done:
            # 만약 MaskablePPO라면 행동 마스크를 사용하여 predict를 호출해야 함
            if selected_algo == "MaskablePPO":
                # 환경에서 행동 마스크를 가져옴
                action_masks = test_env.action_masks()
                action, _states = model.predict(obs, deterministic=False, action_masks=action_masks)
            else:
                action, _states = model.predict(obs, deterministic=False)
            
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards += episode_reward
        print(f"{selected_algo} - 에피소드 {episode + 1}: 보상 = {episode_reward}")

    average_reward = total_rewards / num_episodes
    print(f"{selected_algo} 알고리즘 - 평균 보상: {average_reward}\n")

    # 결과 기록
    results.append({
        "Algorithm": selected_algo,
        "Timesteps": str(total_steps),
        "Average Reward": average_reward
    })

    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)

    # 결과 출력
    print("\n알고리즘별 학습 및 테스트 결과:")
    print(results_df)

    # 결과를 CSV 파일로 저장
    results_df.to_csv("training_results.csv", index=False)
    print("학습 결과가 'training_results.csv' 파일로 저장되었습니다.\n")

    # 시각화: 알고리즘별 평균 보상을 막대 그래프로 표시
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="Algorithm", y="Average Reward", palette="viridis")
    plt.title("Average Rewards by Algorithm")
    plt.ylabel("Average Reward")
    plt.xlabel("Algorithm")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("average_rewards_plot.png")
    plt.show()

    print("평균 보상 막대 그래프가 'average_rewards_plot.png'에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model.")
    parser.add_argument('--algo', type=str, required=True, help="The algorithm to use for training.")
    args = parser.parse_args()

    # 선택한 알고리즘과 함께 학습 함수 호출
    train_and_evaluate(selected_algo=args.algo, total_steps=1000000)