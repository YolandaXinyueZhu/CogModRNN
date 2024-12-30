import argparse
import os
import pickle
import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 5)
pd.set_option('expand_frame_repr', False)
import seaborn as sns
import warnings
import scipy.io as sio
import copy
from datetime import datetime
from typing import Optional, List
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from disentangled_rnns.library import disrnn, rnn_utils
from train_disrnn import load_data, to_onehot, zs_to_onehot, dataset_size, create_train_test_datasets, preprocess_data

warnings.filterwarnings("ignore")

def compute_log_likelihood_and_accuracy(xs, ys, model_fun, params):
    n_trials_per_session, n_sessions = ys.shape[:2]
    model_outputs, _ = rnn_utils.eval_network(model_fun, params, xs)
    predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))

    log_likelihood = 0
    correct_predictions = 0
    n = 0  # Total number of trials across sessions.

    for sess_i in range(n_sessions):
        for trial_i in range(n_trials_per_session):
            actual_choice = int(ys[trial_i, sess_i])
            if (actual_choice >= 0) and (0 <= actual_choice < predicted_log_choice_probabilities.shape[2]):
                log_likelihood += predicted_log_choice_probabilities[trial_i, sess_i, actual_choice]
                predicted_choice = np.argmax(predicted_log_choice_probabilities[trial_i, sess_i])
                if predicted_choice == actual_choice:
                    correct_predictions += 1
                n += 1

    normalized_likelihood = np.exp(log_likelihood / n)
    accuracy = correct_predictions / n
    print(f'Model Normalized Likelihood for Human-Chosen Actions: {100 * normalized_likelihood}%')
    print(f'Model Thresholded Classifier Accuracy for Human-Chosen Actions: {accuracy * 100}%')
    return normalized_likelihood, accuracy


def evaluate(xs, actual_choices, model_fun, params):
    model_outputs, _ = rnn_utils.eval_network(model_fun, params, xs)
    predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))
    return predicted_log_choice_probabilities, actual_choices, xs


def plot_session(choices: np.ndarray,
                 rewards: np.ndarray,
                 timeseries: np.ndarray,
                 timeseries_name: str,
                 save_folder_name,
                 labels: Optional[List[str]] = None,
                 fig_ax: Optional = None):
    choose_high = choices == 1
    choose_low = choices == 0
    rewarded = rewards == 1

    y_high = np.max(timeseries) + 0.1
    y_low = np.min(timeseries) - 0.1

    if labels is not None:
        if timeseries.ndim == 1:
            timeseries = timeseries[:, None]
        if len(labels) != timeseries.shape[2]:
            raise ValueError('labels length must match timeseries.shape[2].')
        for i in range(timeseries.shape[2]):
            fig, ax = plt.subplots(1, figsize=(20, 10))
            ax.plot(timeseries[:, :, i], label=labels[i])
            ax.legend(bbox_to_anchor=(1, 1))

            if choices.max() <= 1:
                # Rewarded high
                ax.scatter(
                    np.argwhere(choose_high & rewarded),
                    y_high * np.ones(np.sum(choose_high & rewarded)),
                    color='green',
                    marker=3)
                ax.scatter(
                    np.argwhere(choose_high & rewarded),
                    y_high * np.ones(np.sum(choose_high & rewarded)),
                    color='green',
                    marker='|')
                # Omission high
                ax.scatter(
                    np.argwhere(choose_high & (1 - rewarded)),
                    y_high * np.ones(np.sum(choose_high & (1 - rewarded))),
                    color='red',
                    marker='|')

                # Rewarded low
                ax.scatter(
                    np.argwhere(choose_low & rewarded),
                    y_low * np.ones(np.sum(choose_low & rewarded)),
                    color='green',
                    marker='|')
                ax.scatter(
                    np.argwhere(choose_low & rewarded),
                    y_low * np.ones(np.sum(choose_low & rewarded)),
                    color='green',
                    marker=2)
                # Omission Low
                ax.scatter(
                    np.argwhere(choose_low & (1 - rewarded)),
                    y_low * np.ones(np.sum(choose_low & (1 - rewarded))),
                    color='red',
                    marker='|')

            save_dir = save_folder_name
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, f"plot_{labels[i]}.png")
            ax.set_xlabel('Trial')
            ax.set_ylabel(f"{timeseries_name} ({labels[i]})")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close(fig)
            print(f"Plot for {labels[i]} saved to {plot_path}")
    else:
        fig, ax = plt.subplots(1, figsize=(20, 10))
        ax.plot(timeseries)

        if choices.max() <= 1:
            # Rewarded high
            ax.scatter(
                np.argwhere(choose_high & rewarded),
                y_high * np.ones(np.sum(choose_high & rewarded)),
                color='green',
                marker=3)
            ax.scatter(
                np.argwhere(choose_high & rewarded),
                y_high * np.ones(np.sum(choose_high & rewarded)),
                color='green',
                marker='|')
            # Omission high
            ax.scatter(
                np.argwhere(choose_high & (1 - rewarded)),
                y_high * np.ones(np.sum(choose_high & (1 - rewarded))),
                color='red',
                marker='|')

            # Rewarded low
            ax.scatter(
                np.argwhere(choose_low & rewarded),
                y_low * np.ones(np.sum(choose_low & rewarded)),
                color='green',
                marker='|')
            ax.scatter(
                np.argwhere(choose_low & rewarded),
                y_low * np.ones(np.sum(choose_low & rewarded)),
                color='green',
                marker=2)
            # Omission Low
            ax.scatter(
                np.argwhere(choose_low & (1 - rewarded)),
                y_low * np.ones(np.sum(choose_low & (1 - rewarded))),
                color='red',
                marker='|')

        save_dir = save_folder_name
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, "plot_session.png")
        ax.set_xlabel('Trial')
        ax.set_ylabel(timeseries_name)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Session plot saved to {plot_path}")


def plot_latent_activations_for_session(sess_i, xs_train, pred_train, act_train, bitResponseAIsCorr_train, P_A_train, disrnn_params, make_disrnn, save_folder_name,
                                        train_test_split_variables):
    trials, _, features = xs_train.shape

    session_xs_train = xs_train[:, sess_i, :]  # (trials, features)
    session_xs_train = session_xs_train[:, np.newaxis, :]
    trial_end = session_xs_train.shape[0]

    network_outputs, network_states = rnn_utils.eval_network(make_disrnn, disrnn_params, session_xs_train[:trial_end, :])
    network_states = np.array(network_states)
    disrnn_activations = network_states[:trial_end, :]
    labels = [f'Latent_{i+1}' for i in range(disrnn_activations.shape[2])]

    pred_session = pred_train[:, sess_i, :][:trial_end, :]
    pred_session = pred_session[:, np.newaxis, :]

    act_session = act_train[:, sess_i, :][:trial_end, :]
    act_session = act_session[:, np.newaxis, :]

    bitResponseAIsCorr_session = bitResponseAIsCorr_train[:, sess_i, :][:trial_end].squeeze()
    P_A_session = P_A_train[:, sess_i, :][:trial_end].squeeze()

    # Now we use the unified create_dataframe_for_plotting without train_test_split_variables for session only
    df_session = create_dataframe_for_plotting(
        predictions=pred_session,
        actions=act_session,
        xs=session_xs_train[:trial_end, :, :],
        bitResponseAIsCorr=bitResponseAIsCorr_session,
        P_A=P_A_session,
        train_test_split_variables=None
    )

    choices = df_session["act"].values
    rewards = df_session["rwd"].values

    latent_sigmas = 2 * jax.nn.sigmoid(
        jnp.array(disrnn_params['hk_dis_rnn']['latent_sigmas_unsquashed'])
    )
    latent_sigma_order = np.argsort(disrnn_params['hk_dis_rnn']['latent_sigmas_unsquashed'])
    latent_sigmas = latent_sigmas[latent_sigma_order]
    latent_names = [f'Latent_{i + 1}' for i in range(disrnn_activations.shape[2])]
    labels_with_sigmas = [f"{name} (σ={sigma:.4f})" for name, sigma in zip(latent_names, latent_sigmas)]
    disrnn_activations_sorted = disrnn_activations[:, :, latent_sigma_order]

    plot_session(choices, rewards, timeseries=disrnn_activations_sorted, timeseries_name='Network Activations', labels=labels_with_sigmas,
                 save_folder_name=save_folder_name)


def plot_single_checkpoint_performance(metrics, save_folder, args_dict):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    labels = ['Training', 'Test', 'Shuffled']
    likelihoods = [metrics['train_norm_likelihood'], metrics['test_norm_likelihood'], metrics['shuffled_norm_likelihood']]
    upper_bounds = [metrics['train_upper_bound'], metrics['test_upper_bound']]

    bar_positions = np.arange(len(labels))
    bar_width = 0.35
    bars = ax.bar(bar_positions, likelihoods, bar_width, label='Normalized Likelihood')
    ax.axhline(y=upper_bounds[0], color='orange', linestyle='--', label='Training Upper Bound')
    ax.axhline(y=upper_bounds[1], color='gray', linestyle='-.', label='Test Upper Bound')

    ax.set_xlabel('Dataset Type')
    ax.set_ylabel('Normalized Likelihood')
    ax.set_title('Model Performance Metrics')
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels)
    ax.legend(fontsize='small')

    hyperparams_text = "\n".join([f"{key}: {value}" for key, value in args_dict.items()])
    plt.text(1.05, 0.5, hyperparams_text, transform=ax.transAxes, fontsize=10, va='center',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray", alpha=0.5))

    save_path = os.path.join(save_folder, 'checkpoint_performance.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)
    print(f"Performance plot saved to {save_path}")


def plot_training_results_one_session(df_train, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    session_df = (
        df_train
        .sort_values(['blockN', 'trialNInBlock'])
        .drop_duplicates(subset=['blockN', 'trialNInBlock'], keep='first')
    )
    session_df['trial_num_in_session'] = range(len(session_df))

    # Probability of optimal actions over trials in the session
    plt.figure(figsize=(16, 8))
    sns.lineplot(
        x='trial_num_in_session', y='rwd', data=session_df,
        label='Human Proportion of Optimal Choices (Blue Line)', color='blue'
    )
    sns.lineplot(
        x='trial_num_in_session', y='model_est_optimal_prob', data=session_df,
        label='Model-Estimated Probability of Optimal Action (Orange Line)', color='orange'
    )
    plt.legend()
    plt.xlabel('Trial Number in Session')
    plt.ylabel('Probability')
    plt.xlim(left=0)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title('Session: Probability of Optimal Choices over Trials')
    plt.tight_layout()
    plot_path1 = os.path.join(save_folder, 'probability_optimal_choices_session.png')
    plt.savefig(plot_path1)
    plt.close()

    # Probability of chosen actions over trials in the session
    plt.figure(figsize=(16, 8))
    sns.lineplot(
        x='trial_num_in_session', y='human_chosen_action_prob', data=session_df,
        label='Human Chosen Action Probability (Green Line)', color='green'
    )
    sns.lineplot(
        x='trial_num_in_session', y='pred_acc', data=session_df,
        label='Model Chosen Action Probability (Red Line)', color='red'
    )
    plt.legend()
    plt.xlabel('Trial Number in Session')
    plt.ylabel('Probability')
    plt.xlim(left=0)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title('Session: Probability of Chosen Actions over Trials')
    plt.tight_layout()
    plot_path2 = os.path.join(save_folder, 'probability_chosen_actions_session.png')
    plt.savefig(plot_path2)
    plt.close()

    print(f"Plots for the session saved to {save_folder}")


def plot_training_results_per_block(df_train, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    unique_blocks = df_train['blockN'].unique()

    for block in unique_blocks:
        df_block = df_train[df_train['blockN'] == block]

        # Plot 1
        plt.figure(figsize=(16, 8))
        sns.lineplot(
            x='trialNInBlock', y='human_chosen_action_prob', 
            data=df_block, label='Human Proportion of Chosen Actions', color='blue'
        )
        sns.lineplot(
            x='trialNInBlock', y='pred_acc', 
            data=df_block, label='Model-Estimated Probability of Chosen Actions', color='orange'
        )
        plt.legend()
        plt.xlabel('Trial Number within Block')
        plt.ylabel('Probability of Chosen Action')
        plt.xlim(left=1)
        plt.ylim(bottom=-0.1, top=1.1)
        plt.title(f'Block {block}: Probability of Chosen Action vs. Trial Number')
        plt.tight_layout()
        plot_path = os.path.join(save_folder, f'prob_chosen_action_vs_trial_in_block_block_{block}.png')
        plt.savefig(plot_path)
        plt.close()

        # Plot 2
        plt.figure(figsize=(16, 8))
        sns.lineplot(
            x='trialNInBlock', y='rwd', 
            data=df_block, label='Human Proportion of Optimal Actions', color='purple'
        )
        sns.lineplot(
            x='trialNInBlock', y='model_est_optimal_prob', 
            data=df_block, label='Model-Estimated Probability of Optimal Actions', color='red'
        )
        plt.legend()
        plt.xlabel('Trial Number within Block')
        plt.ylabel('Probability of Optimal Action')
        plt.xlim(left=1)
        plt.ylim(bottom=-0.1, top=1.1)
        plt.title(f'Block {block}: Probability of Optimal Action vs. Trial Number')
        plt.tight_layout()
        plot_path = os.path.join(save_folder, f'prob_optimal_action_vs_trial_in_block_block_{block}.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Plots for block {block} saved to {save_folder}")


def plot_training_results(df_train, dataset_train, save_folder):
    _trial_num = np.tile(np.arange(dataset_train._xs.shape[0]), [dataset_train._xs.shape[1], 1]).transpose()

    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='pred_acc', label='Normalized Log-Likelihood')
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='model_est_optimal_prob', label='Upper Bound', linestyle='--')
    plt.legend()
    plt.xlabel('Trial Number')
    plt.ylabel('Normalized Likelihood')
    plt.xlim(left=1)
    plt.ylim(bottom=0, top=1.1)
    plt.title('Training Data: Normalized Log-Likelihood vs Upper Bound per Trial')
    plt.savefig(os.path.join(save_folder, 'normalized_log_likelihood_vs_upper_bound_train.png'))
    plt.close()

    # Probability of optimal choices over trials
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='rwd', label='Human Proportion of Optimal Choices (Blue Line)')
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='model_est_optimal_prob', label='Model-Estimated Probability of Optimal Action (Orange Line)')
    plt.legend()
    plt.xlabel('Trial Number')
    plt.ylabel('Probability')
    plt.xlim(left=1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title('Training Data: Probability of Optimal Choices over Trials')
    plt.savefig(os.path.join(save_folder, 'probability_optimal_choices_over_trials_train.png'))
    plt.close()

    # Probability of chosen actions over trials
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='human_chosen_action_prob', label='Human Chosen Action Probability (Green Line)', color='green')
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='pred_acc', label='Model Chosen Action Probability (Red Line)', color='red')
    plt.legend()
    plt.xlabel('Trial Number')
    plt.ylabel('Probability')
    plt.xlim(left=1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title('Training Data: Probability of Chosen Actions over Trials')
    plt.savefig(os.path.join(save_folder, 'probability_chosen_actions_over_trials_train.png'))
    plt.close()

    # Probability per trial within block
    plt.figure(figsize=(16, 8))
    df_agg = df_train.groupby('trialNInBlock').agg({
        'human_chosen_action_prob':'mean',
        'pred_acc':'mean'
    }).reset_index()

    sns.lineplot(x='trialNInBlock', y='human_chosen_action_prob', data=df_agg, label='Human Proportion of Chosen Actions', color='blue')
    sns.lineplot(x='trialNInBlock', y='pred_acc', data=df_agg, label='Model-Estimated Probability of Chosen Actions', color='orange')
    
    plt.legend()
    plt.xlabel('Trial Number within Block')
    plt.ylabel('Probability of Optimal Action')
    plt.xlim(left=1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title('Training Data: Probability of Optimal Action vs. Trial Number within Block')
    plt.tight_layout()
    plot_path4 = os.path.join(save_folder, 'prob_chosen_action_vs_trial_in_block_train.png')
    plt.savefig(plot_path4)
    plt.close()

    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='human_chosen_action_prob', label='Human Chosen Action Probability (Green Line)', color='green')
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='pred_acc', label='Model Chosen Action Probability (Red Line)', color='red')
    plt.legend()
    plt.xlabel('Trial Number')
    plt.ylabel('Probability')
    plt.xlim(left=1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title('Training Data: Probability of Chosen Actions over Trials')
    plt.savefig(os.path.join(save_folder, 'probability_chosen_actions_over_trials_train.png'))
    plt.close()

    # Probability of optimal action vs trial in block
    plt.figure(figsize=(16, 8))
    df_agg = df_train.groupby('trialNInBlock').agg({
        'rwd': 'mean',
        'model_est_optimal_prob': 'mean',
    }).reset_index()

    sns.lineplot(x='trialNInBlock', y='rwd', data=df_agg, label='Human Proportion of Optimal Actions', color='purple')
    sns.lineplot(x='trialNInBlock', y='model_est_optimal_prob', data=df_agg, label='Model-Estimated Probability of Optimal Actions', color='red')
    
    plt.legend()
    plt.xlabel('Trial Number within Block')
    plt.ylabel('Probability of Optimal Action')
    plt.xlim(left=1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title('Training Data: Probability of Optimal Action vs. Trial Number within Block')
    plt.tight_layout()
    plot_path5 = os.path.join(save_folder, 'prob_optimal_action_vs_trial_in_block_train.png')
    plt.savefig(plot_path5)
    plt.close()

    print(f"Training result plots saved to {save_folder}")


def plot_bottlenecks(params, save_folder_name, sort_latents=True, obs_names=None):
    import matplotlib.colors as mcolors
    save_dir = save_folder_name
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, "bottleneck.png")

    params_disrnn = params['hk_dis_rnn']
    latent_dim = params_disrnn['latent_sigmas_unsquashed'].shape[0]
    obs_dim = params_disrnn['update_mlp_sigmas_unsquashed'].shape[0] - latent_dim

    if obs_names is None:
        if obs_dim == 2:
            obs_names = ['Choice', 'Reward']
        elif obs_dim == 5:
            obs_names = ['A', 'B', 'C', 'D', 'Reward']
        else:
            obs_names = [f'Obs_{i}' for i in range(1, obs_dim + 1)]

    latent_sigmas = 2 * jax.nn.sigmoid(jnp.array(params_disrnn['latent_sigmas_unsquashed']))
    update_sigmas = 2 * jax.nn.sigmoid(np.transpose(params_disrnn['update_mlp_sigmas_unsquashed']))

    if sort_latents:
        latent_sigma_order = np.argsort(params_disrnn['latent_sigmas_unsquashed'])
        latent_sigmas = latent_sigmas[latent_sigma_order]
        update_sigma_order = np.concatenate((np.arange(0, obs_dim, 1), latent_sigma_order + obs_dim), axis=0)
        update_sigmas = update_sigmas[latent_sigma_order, :]
        update_sigmas = update_sigmas[:, update_sigma_order]

    latent_names = [f'Latent_{i + 1}' for i in range(latent_dim)]

    fig, axes = plt.subplots(1, 2, figsize=(25, 12))
    cmap = 'Oranges'

    ax1 = axes[0]
    im1 = ax1.imshow(latent_sigmas.reshape(1, -1), cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax1.set_yticks([0])
    ax1.set_yticklabels(['Latent #'])
    ax1.set_xticks(range(latent_dim))
    ax1.set_xticklabels(latent_names, rotation=90)
    ax1.set_title('Latent Bottlenecks (Sigma)')

    for i in range(latent_dim):
        ax1.text(i, 0, f"{latent_sigmas[i]:.4f}", ha='center', va='center', color='black', fontsize=12, weight='bold')

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Sigma Value')

    ax2 = axes[1]
    im2 = ax2.imshow(update_sigmas, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax2.set_yticks(range(latent_dim))
    ax2.set_yticklabels(latent_names)
    xlabels = obs_names + latent_names
    ax2.set_xticks(range(len(xlabels)))
    ax2.set_xticklabels(xlabels, rotation=90)
    ax2.set_title('Update MLP Bottlenecks (Sigma)')

    for i in range(latent_dim):
        for j in range(len(xlabels)):
            ax2.text(j, i, f"{update_sigmas[i, j]:.4f}", ha='center', va='center', color='black', fontsize=10, weight='bold')

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Sigma Value')

    fig.text(0.5, 0.95, 'Darker Colors → Higher Sigma Values', ha='center', fontsize=14, color='black')

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Bottleneck plot saved to {plot_path}")

    print("\n=== Latent Bottleneck Sigmas ===")
    for i, sigma in enumerate(latent_sigmas):
        print(f"Latent {i + 1}: Sigma = {sigma:.4f}")

    print("\n=== Update MLP Bottleneck Sigmas ===")
    for i in range(update_sigmas.shape[0]):
        print(f"Latent {i + 1} Update MLP Sigmas:")
        for j, sigma in enumerate(update_sigmas[i]):
            print(f"  Input {j + 1} ({xlabels[j]}): Sigma = {sigma:.4f}")

    latent_sigmas_path = os.path.join(save_dir, "latent_sigmas.txt")
    update_sigmas_path = os.path.join(save_dir, "update_mlp_sigmas.txt")

    with open(latent_sigmas_path, 'w') as f:
        f.write("Latent Bottleneck Sigmas:\n")
        for i, sigma in enumerate(latent_sigmas):
            f.write(f"Latent {i + 1}: {sigma:.6f}\n")

    with open(update_sigmas_path, 'w') as f:
        f.write("Update MLP Bottleneck Sigmas:\n")
        for i in range(update_sigmas.shape[0]):
            f.write(f"Latent {i + 1} Update MLP Sigmas:\n")
            for j, sigma in enumerate(update_sigmas[i]):
                f.write(f"  Input {j + 1} ({xlabels[j]}): {sigma:.6f}\n")
            f.write("\n")

    print(f"\nSigma values have been saved to:\n- {latent_sigmas_path}\n- {update_sigmas_path}")
    return fig


def create_dataframe_for_plotting(predictions, 
                                  actions, 
                                  xs,
                                  train_test_split_variables=None,
                                  bitResponseAIsCorr=None,
                                  P_A=None):
    """
    Unified function to create a DataFrame for plotting.

    If train_test_split_variables is provided, it uses those to build a full dataframe (as originally in create_dataframe_for_plotting).
    If train_test_split_variables is None, it assumes session-level plotting (similar to create_dataframe_for_plotting_session) and requires bitResponseAIsCorr and P_A.
    """

    action_0 = actions == 0
    action_1 = actions == 1
    action_others = ~np.isin(actions, [0, 1])
    action_oh = np.concatenate([action_0, action_1, action_others], axis=2)
    nan = np.empty([*predictions.shape[:2], 1])
    nan.fill(np.nan)
    predictions_expanded = np.concatenate([predictions, nan], axis=2)
    prediction_prob = predictions_expanded[action_oh]
    prediction_prob = prediction_prob.reshape(actions.shape)

    if train_test_split_variables is not None:
        # Full data scenario
        context = train_test_split_variables["context_train"][:, :, 0].ravel()
        blockN = train_test_split_variables["blockN_train"][:, :, 0].ravel()
        trialNInBlock = train_test_split_variables["trialNInBlock_train"][:, :, 0].ravel()
        bitResponseAIsCorr_full = train_test_split_variables["bitResponseAIsCorr_train"][:, :, 0].ravel()
        P_A_full = train_test_split_variables["P_A_train"][:, :, 0].ravel()

        # Include stimulusSlotID and stimulusSlotID_encounterNInBlock if available
        if "stimulusSlotID_train" in train_test_split_variables:
            stimulusSlotID = train_test_split_variables["stimulusSlotID_train"][:, :, 0].ravel()
        else:
            stimulusSlotID = np.zeros_like(context)  # Placeholder if not present

        if "stimulusSlotID_encounterNInBlock_train" in train_test_split_variables:
            stimulusSlotID_encounterNInBlock = train_test_split_variables["stimulusSlotID_encounterNInBlock_train"][:, :, 0].ravel()
        else:
            stimulusSlotID_encounterNInBlock = np.zeros_like(context)  # Placeholder if not present

        # Note: The original create_dataframe_for_plotting uses rwd from xs with a bottom stack of zeros
        # 'rwd': np.vstack((xs[1:, :, 0], np.zeros((1, xs.shape[1])))).ravel(),
        # Let's keep this consistent.
        rwd = np.vstack((xs[1:, :, 0], np.zeros((1, xs.shape[1])))).ravel()

        df = pd.DataFrame({
            'context': context,
            'blockN': blockN,
            'trialNInBlock': trialNInBlock,
            'act': actions[:, :, 0].ravel(),
            'pred': np.exp(predictions[:, :, 1].ravel()),
            'pred_acc': np.exp(prediction_prob.ravel()),
            'bitResponseAIsCorr': bitResponseAIsCorr_full,
            'P_A': P_A_full,
            'rwd': rwd,
            'stimulusSlotID': stimulusSlotID,
            'stimulusSlotID_encounterNInBlock': stimulusSlotID_encounterNInBlock
        })

        df.to_csv("./test_df.csv", index=False)

        df['model_est_optimal_prob'] = np.where(df['bitResponseAIsCorr'] == 1, df['pred'], 1 - df['pred'])
        df['human_chosen_action_prob'] = np.where(df['act'] == 1, df['P_A'], 1 - df['P_A'])
        df['rwd_manual'] = np.where(df['act'] == df['bitResponseAIsCorr'], 1, 0)

        return df
    else:
        # Session-level scenario (previously create_dataframe_for_plotting_session)
        # Requires bitResponseAIsCorr and P_A
        if bitResponseAIsCorr is None or P_A is None:
            raise ValueError("bitResponseAIsCorr and P_A must be provided if train_test_split_variables is None.")

        # Use the session logic:
        # 'rwd': np.vstack((np.zeros((1, xs.shape[1])), xs[1:, :, 0])).ravel()
        # This was the original indexing in create_dataframe_for_plotting_session
        rwd = np.vstack((np.zeros((1, xs.shape[1])), xs[1:, :, 0])).ravel()

        df = pd.DataFrame({
            'act': actions[:, :, 0].ravel(),
            'pred': np.exp(predictions[:, :, 1].ravel()),
            'pred_acc': np.exp(prediction_prob.ravel()),
            'bitResponseAIsCorr': bitResponseAIsCorr,
            'P_A': P_A,
            'rwd': rwd
        })

        df['model_est_optimal_prob'] = np.where(df['bitResponseAIsCorr'] == 1, df['pred'], 1 - df['pred'])
        df['human_chosen_action_prob'] = np.where(df['act'] == 1, df['P_A'], 1 - df['P_A'])
        df['rwd_manual'] = np.where(df['act'] == df['bitResponseAIsCorr'], 1, 0)

        return df


### New Plotting Function ###
def plot_average_block_3_model_probability(df, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    
    # Filter to block 3 and first encounters
    df_block3 = df[(df['blockN'] == 3) & (df['stimulusSlotID_encounterNInBlock'] == 1)].copy()
    
    # Sort by trial number to get the order of first-encountered stimuli
    df_block3 = df_block3.sort_values('trialNInBlock')
    print(df_block3)
    
    # Check that the number of rows is a multiple of 4
    n = len(df_block3)
    
    # Assign encounter_order 1 through 4 repeatedly
    df_block3['encounter_order'] = np.tile([1,2,3,4], n // 4)
    
    # Now average model probability by encounter_order
    df_mean = df_block3.groupby('encounter_order', as_index=False)['model_est_optimal_prob'].mean()
    
    # Plot these 4 averaged values
    plt.figure(figsize=(6,4))
    sns.lineplot(
        data=df_mean,
        x='encounter_order',
        y='model_est_optimal_prob',
        marker='o',
        color='blue'
    )

    # Label the x-axis ticks as A, B, C, D
    plt.xticks([1, 2, 3, 4], ['A', 'B', 'C', 'D'])

    plt.xlabel('Stimulus')
    plt.ylabel('Average Model Probability of Optimal Action')
    plt.title('Average Model Probability for First-Encounter Stimuli in Block 3')
    plt.ylim(0, 1)
    plt.tight_layout()

    plot_path = os.path.join(save_folder, 'average_block_3_model_probabilities.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")


def main(seed, saved_checkpoint_pth):
    np.random.seed(seed)

    dataset_type = 'RealWorldKimmelfMRIDataset'  
    dataset_path = "dataset/tensor_for_dRNN_desc-syn_nSubs-2000_nSessions-1_nBlocks-7_nTrialsPerBlock-50_b-0.11_NaN_10.5_0.93_0.45_NaN_NaN_20241104.mat"
    dataset_train, dataset_test, train_test_split_variables = preprocess_data(dataset_type, dataset_path, 0.1)

    # Load the checkpoint parameters
    with open(saved_checkpoint_pth, 'rb') as file:
        checkpoint = pickle.load(file)
    
    args_dict = checkpoint['args_dict']
    disrnn_params = checkpoint['disrnn_params']
    print(f'Loaded disrnn_params from {saved_checkpoint_pth}')

    def make_disrnn():
      model = disrnn.HkDisRNN(
          obs_size=6,
          target_size=2,
          latent_size=args_dict['latent_size'],
          update_mlp_shape=args_dict['update_mlp_shape'],
          choice_mlp_shape=args_dict['choice_mlp_shape'],
          eval_mode=0.0,
          beta_scale=args_dict['beta_scale'],
          activation=jax.nn.relu
      )
      return model

    xs_train, ys_train = next(dataset_train)
    xs_test, ys_test = next(dataset_test)
    shuffled_dataset_test = copy.deepcopy(dataset_test)
    shuffled_dataset_test._ys = np.random.permutation(shuffled_dataset_test._ys)
    shuffled_xs_test, shuffled_ys_test = next(shuffled_dataset_test)

    # Compute log-likelihood and accuracy for training and test datasets
    print('Normalized Likelihoods and Accuracies for disRNN')
    print('Training Dataset')
    train_norm_likelihood, _ = compute_log_likelihood_and_accuracy(xs_train, ys_train, make_disrnn, disrnn_params)

    pred_train, ys_train, xs_train = evaluate(xs_train, ys_train, make_disrnn, disrnn_params)
    pred_test, ys_test, xs_test = evaluate(xs_test, ys_test, make_disrnn, disrnn_params)
    df_train = create_dataframe_for_plotting(pred_train, ys_train, xs_train, train_test_split_variables=train_test_split_variables)

    save_folder_name = os.path.join('plots', saved_checkpoint_pth.split('checkpoints/')[1].split('.pkl')[0])
    os.makedirs(save_folder_name, exist_ok=True)

    # plot_training_results_one_session(df_train, save_folder_name)
    plot_training_results(df_train, dataset_train, save_folder_name)
    # plot_training_results_per_block(df_train, save_folder_name)
    
    plot_bottlenecks(disrnn_params, save_folder_name)
    session_i = 0
    
    plot_latent_activations_for_session(sess_i = session_i,
                                        xs_train=xs_train, 
                                        pred_train=pred_train, 
                                        act_train=ys_test, 
                                        bitResponseAIsCorr_train=train_test_split_variables["bitResponseAIsCorr_train"], 
                                        P_A_train=train_test_split_variables['P_A_train'], 
                                        disrnn_params=disrnn_params, 
                                        make_disrnn=make_disrnn,
                                        save_folder_name = save_folder_name,
                                        train_test_split_variables = train_test_split_variables
                                        )
    
    # New plot:
    # plot_average_block_3_model_probability(df_train, save_folder_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", nargs=1, type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--saved_checkpoint_pth", type=str, required=True, help="Saved checkpoint path for evaluation.")
    args = parser.parse_args()

    main(args.seed, args.saved_checkpoint_pth)
