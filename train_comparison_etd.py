import argparse
from RJSPEnv.Env import RJSPEnv
from RJSPEnv.NoETDEnv import NoETDEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
import matplotlib.pyplot as plt

def train_model(env, algo_class, total_steps, model_name):
    model = algo_class(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=f"./tensorboard/etd_comparison_{model_name}"
    )

    model.learn(total_timesteps=total_steps)
    model.save(f"./models/etd_comparison_{model_name}_final")
    return model

def test_and_render_model(model, test_env, num_episodes=1):
    for episode in range(num_episodes):
        obs, _ = test_env.reset()
        done = False

        while not done:
            action_masks = test_env.action_masks()
            action, _states = model.predict(obs, deterministic=False, action_masks=action_masks)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

            # 환경을 렌더링하여 스케줄링 결과를 시각화
            test_env.render()

def plot_comparison(etd_reward, no_etd_reward):
    plt.figure(figsize=(8, 5))
    plt.bar(['ETD', 'No ETD'], [etd_reward, no_etd_reward], color=['blue', 'orange'])
    plt.title("Average Reward Comparison")
    plt.xlabel("Environment")
    plt.ylabel("Average Reward")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ETD and No-ETD environments.")
    parser.add_argument('--steps', type=int, default=1000000, help="Total training steps.")
    args = parser.parse_args()

    instance = "12x8"
    machine_config_path = f"instances/Machines/v0-{instance}.json"
    job_config_path = f"instances/Jobs/v0-{instance}-12.json"
    job_repeats_params = [(5, 1)] * 12

    # ETD 환경
    etd_env = RJSPEnv(
        machine_config_path=machine_config_path,
        job_config_path=job_config_path,
        job_repeats_params=job_repeats_params,
        max_time=100,
        test_mode=False
    )

    # No ETD 환경
    no_etd_env = NoETDEnv(
        machine_config_path=machine_config_path,
        job_config_path=job_config_path,
        job_repeats_params=job_repeats_params,
        max_time=100,
        test_mode=False
    )

    # 학습
    etd_model = train_model(etd_env, MaskablePPO, args.steps, "ETD")
    no_etd_model = train_model(no_etd_env, MaskablePPO, args.steps, "No_ETD")

    # 테스트
    etd_reward = test_model(etd_model, etd_env)
    no_etd_reward = test_model(no_etd_model, no_etd_env)

    # 결과 비교
    plot_comparison(etd_reward, no_etd_reward)
