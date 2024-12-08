import matplotlib.pyplot as plt
import argparse
from RJSPEnv.Env import RJSPEnv
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO, TRPO

def test_model(model, test_env, selected_algo):
    num_episodes = 10
    total_rewards = []

    print(f"{selected_algo} 알고리즘 테스트 중...")
    for episode in range(num_episodes):
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            if selected_algo == "MaskablePPO":
                action_masks = test_env.action_masks()
                action, _states = model.predict(obs, deterministic=False, action_masks=action_masks)
            else:
                action, _states = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        # 환경을 렌더링하여 스케줄링 결과를 시각화
        test_env.render()

        total_rewards.append(episode_reward)
        print(f"{selected_algo} - 에피소드 {episode + 1}: 보상 = {episode_reward}")

    average_reward = sum(total_rewards) / num_episodes
    print(f"{selected_algo} 알고리즘 - 평균 보상: {average_reward}\n")

    return average_reward

def plot_average_rewards(average_rewards):
    plt.figure(figsize=(10, 6))
    algorithms = list(average_rewards.keys())
    rewards = list(average_rewards.values())
    plt.bar(algorithms, rewards, color='skyblue')
    plt.title("Average Rewards by Algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("Average Reward")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a reinforcement learning model.")
    parser.add_argument('--algos', type=str, nargs='+', required=True, help="The algorithms to use for testing.")
    args = parser.parse_args()

    machine_config_path = 'instances/Machines/v0-5x5.json'
    job_config_path = 'instances/Jobs/v0-5x5-5.json'
    job_repeats_params = [(3, 1)] * 5

    test_env = RJSPEnv(
        machine_config_path=machine_config_path,
        job_config_path=job_config_path,
        job_repeats_params=job_repeats_params,
        max_time=50,
        test_mode=True
    )

    algorithms = {
        "MaskablePPO": MaskablePPO,
        "PPO": PPO,
        "TRPO": TRPO
    }

    average_rewards = {}

    for algo in args.algos:
        if algo not in algorithms:
            raise ValueError(f"알 수 없는 알고리즘: {algo}. 사용 가능한 알고리즘: {list(algorithms.keys())}")

        model_path = f"./models/{algo}_final"
        model = algorithms[algo].load(model_path)

        average_reward = test_model(model, test_env, algo)
        average_rewards[algo] = average_reward

    plot_average_rewards(average_rewards)