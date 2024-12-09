# 모델 성능 비교 관련 시각화 코드import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_learning_curves(log_dir):
    configurations = ["NoETDEnv_PPO", "ETDEnv_PPO", "NoETDEnv_MaskablePPO", "ETDEnv_MaskablePPO"]
    plt.figure(figsize=(10, 6))

    for config in configurations:
        csv_path = os.path.join(log_dir, config, "progress.csv")
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            plt.plot(data['time/total_timesteps'], data['rollout/ep_rew_mean'], label=config)

    plt.title("Learning Curves for Different Configurations")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episode Reward Mean")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_learning_curves("./tensorboard/ablation_study3")