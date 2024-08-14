from sb3_contrib import MaskablePPO
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from gymnasium import spaces
from scheduler_env.customEnv_repeat import SchedulingEnv
import torch


def plot_input_weights(model_path, env, save_img=False):
    # Load the model
    model = MaskablePPO.load(model_path)
    # print(dir(model))

    # Get the policy network
    policy = model.policy
    # print(f"Policy network: {policy}")

    mlp_extractor = policy.mlp_extractor
    print(f"MLP extractor: {mlp_extractor}")

    features_extractor = policy.features_extractor
    print(f"Features extractor: {features_extractor}")

    model_feature_names = list(features_extractor.extractors.keys())
    print(f"Model feature names: {model_feature_names}")

    # Create labels for each input feature group and their aggregate importance
    feature_groups = {}
    feature_index = 0
    previous_end_index = -1
    for key, space in env.observation_space.spaces.items():
        # Make sure the feature names match the model feature names
        assert model_feature_names[feature_index] == key, f"Assertion failed: {model_feature_names[feature_index]} != {key}"
        feature_index += 1

        if isinstance(space, spaces.Box):
            num_features = np.prod(space.shape)
            start_index = previous_end_index + 1
            end_index = start_index + num_features - 1
            feature_groups[key] = (start_index, end_index)
            print(f"{key} | range - {start_index}:{end_index}")
            previous_end_index = end_index

    # Find the first linear layer (input layer)
    input_layer = next(layer for layer in mlp_extractor.policy_net if isinstance(layer, nn.Linear))

    # Get the weights of the input layer
    weights = input_layer.weight.detach().numpy()
    print(f"Input shape: {weights.shape}")

    # Make sure the number of input features matches the number of weights
    assert previous_end_index == weights.shape[1] - 1, \
        f"Assertion failed: {previous_end_index} != {weights.shape[1] - 1}"

    # Function to plot and print feature importance
    def plot_and_print_all_features(weights, feature_groups):
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot all points
        for idx in range(weights.shape[1]):
            ax.plot([idx] * weights.shape[0], weights[:, idx], alpha=0.5)

        plt.xlabel('Input Features')
        plt.ylabel('Absolute Weight')
        plt.title('Input Feature Weights')

        # Set xticks and xtick labels
        xticks = []
        xtick_labels = []
        for group, (start_idx, end_idx) in feature_groups.items():
            xticks.append((start_idx + end_idx) / 2)  # Position the label in the middle of the group
            xtick_labels.append(group)
            # Group x-axis labels by feature group
            plt.axvline(x=end_idx, color='gray', linestyle='--')

        plt.xticks(xticks, xtick_labels, rotation=90)  # Rotate labels for better readability

        plt.tight_layout()

        if save_img:
            plt.savefig("input_feature_weights.png", bbox_inches='tight', dpi=100)
        plt.show()

    # Plot and print the feature importance
    plot_and_print_all_features(weights, feature_groups)

    # Calculate the average absolute weight for each input feature
    feature_importance = np.mean(np.abs(weights), axis=0)
    print(f"Feature importance shape: {feature_importance.shape}")

    # Function to plot and print group importance with box plots

    def plot_and_print_groups(feature_groups):
        # Extract importance data for each group
        importance_data = [(group, feature_importance[start_idx:end_idx])
                           for group, (start_idx, end_idx) in feature_groups.items()]

        # Sort the groups by average importance
        importance_data.sort(key=lambda x: np.mean(x[1]), reverse=True)

        # Separate the sorted groups and their importance data
        sorted_groups = [group for group, _ in importance_data]
        sorted_importance_data = [data for _, data in importance_data]

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(sorted_importance_data, vert=True, patch_artist=True,
                   tick_labels=sorted_groups)

        # Customize the plot
        ax.set_xlabel('Input Feature Groups')
        ax.set_ylabel('Importance Distribution')
        ax.set_title("Input Feature Group Importance")
        ax.set_xticklabels(sorted_groups, rotation=45, ha='right')

        # Adjust layout and save as PNG
        plt.tight_layout()
        if save_img:
            plt.savefig("Input Feature Group Importance.png", bbox_inches='tight', dpi=100)
        plt.show()

    # Plot and print the group importance
    plot_and_print_groups(feature_groups)


if __name__ == "__main__":
    # model_path = "./experiments/tmp/2/run_0-{'clip_range': 0.1, 'learning_rate': 0.0005}/best_model.zip"
    model_path = "./logs/tmp/seojun/MP_Env_8_12_3_1_v10.zip"
    env = SchedulingEnv(machine_config_path="instances/Machines/v0-8.json",
                        job_config_path="instances/Jobs/v0-12-repeat-easy.json", job_repeats_params=[(3, 1)] * 12)
    env.reset()
    plot_input_weights(model_path, env, save_img=False)
