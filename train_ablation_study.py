import argparse
from RJSPEnv.Env import RJSPEnv
from RJSPEnv.NoETDEnv import NoETDEnv
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
import os
from stable_baselines3.common.logger import configure

def train_model(env, algo_class, total_steps, model_name, tensorboard_log, callback):
    model = algo_class(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=tensorboard_log
    )
    new_logger = configure(tensorboard_log, ["tensorboard", "csv", "stdout"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=total_steps, callback=callback)
    model.save(f"./models/ablation_study/{model_name}_final")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train models under four different conditions.")
    parser.add_argument('--steps', type=int, default=1000000, help="Total training steps for each model.")
    args = parser.parse_args()

    conditions = [
        {"env_type": "NoETDEnv", "algo": "PPO"},
        {"env_type": "ETDEnv", "algo": "PPO"},
        {"env_type": "NoETDEnv", "algo": "MaskablePPO"},
        {"env_type": "ETDEnv", "algo": "MaskablePPO"}
    ]

    for condition in conditions:
        env_type = condition["env_type"]
        algo = condition["algo"]
        model_name = f"{env_type}_{algo}"
        tensorboard_log = f"./tensorboard/ablation_study2/{model_name}"

        # 환경 설정
        instance = "5x5"  # 필요에 따라 변경
        machine_config_path = f"instances/Machines/v0-{instance}.json"
        job_config_path = f"instances/Jobs/v0-{instance}-8.json"
        job_repeats_params = [(5, 1)] * 5

        if env_type == "ETDEnv":
            env = RJSPEnv(
                machine_config_path=machine_config_path,
                job_config_path=job_config_path,
                job_repeats_params=job_repeats_params,
                max_time=100,
                test_mode=False
            )
        elif env_type == "NoETDEnv":
            env = NoETDEnv(
                machine_config_path=machine_config_path,
                job_config_path=job_config_path,
                job_repeats_params=job_repeats_params,
                max_time=100,
                test_mode=False
            )
        else:
            raise ValueError(f"알 수 없는 환경 타입: {env_type}")

        # 알고리즘 클래스 선택
        if algo == "PPO":
            algo_class = PPO
        elif algo == "MaskablePPO":
            algo_class = MaskablePPO
        else:
            raise ValueError(f"알 수 없는 알고리즘: {algo}")

        # 콜백 설정
        checkpoint_callback = CheckpointCallback(
            save_freq=args.steps//10,
            save_path='./models/ablation_study2/',
            name_prefix=f'model_checkpoint_{model_name}'
        )

        if algo == "MaskablePPO":
            eval_callback = MaskableEvalCallback(
                eval_env=env,
                best_model_save_path=f'./models/ablation_study2/best_{model_name}',
                log_path=f'./logs/ablation_study2/{model_name}_eval',
                eval_freq=10000,
                n_eval_episodes=5,
                deterministic=False,
                render=False
            )
            callbacks = CallbackList([checkpoint_callback, eval_callback])
        else:
            eval_callback = EvalCallback(
                eval_env=env,
                best_model_save_path=f'./models/ablation_study2/best_{model_name}',
                log_path=f'./logs/ablation_study2/{model_name}_eval',
                eval_freq=10000,
                n_eval_episodes=5,
                deterministic=False,
                render=False
            )
            callbacks = CallbackList([checkpoint_callback, eval_callback])

        # 디렉토리 생성
        os.makedirs('./models/ablation_study2/', exist_ok=True)
        os.makedirs('./tensorboard/ablation_study2/', exist_ok=True)
        os.makedirs('./logs/ablation_study2/', exist_ok=True)

        # 모델 학습
        print(f"\n==========================")
        print(f"{model_name} 학습 시작...")
        print("==========================\n")

        train_model(
            env=env,
            algo_class=algo_class,
            total_steps=args.steps,
            model_name=model_name,
            tensorboard_log=tensorboard_log,
            callback=callbacks
        )

        print(f"{model_name} 학습 완료 및 저장됨.\n")

if __name__ == "__main__":
    main()