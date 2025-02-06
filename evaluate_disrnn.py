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



################################################################
### 1) NEW: Side-by-side inference performance (Human vs. Model)
###          (First encounters, excluding the very first)
################################################################
def plot_inference_performance_human_vs_model_side_by_side(df: pd.DataFrame, save_folder: str):
    """
    Side-by-side subplots for 'first encounters' (excluding the first introduced stimulusSlotN=1),
    comparing Human vs. Model performance by block.

    LEFT SUBPLOT: Human's mean rwd
    RIGHT SUBPLOT: Model's mean model_est_optimal_prob
    """
    # Exclude invalid trials
    valid_mask = (df['act'] != -1) & (df['bitResponseAIsCorr'] != -1)
    df = df[valid_mask].copy()

    # Filter to first encounters (except the very first stimulus in a block)
    inference_df = df[
        (df['stimulusSlotID_encounterNInBlock'] == 1) &
        (df['stimulusSlotN'] > 1)
    ].copy()

    # Group for Human
    grouped_human = (
        inference_df
        .groupby('blockN')['rwd']
        .mean()
        .reset_index()
    )
    # Group for Model
    grouped_model = (
        inference_df
        .groupby('blockN')['model_est_optimal_prob']
        .mean()
        .reset_index()
    )

    unique_blocks = sorted(inference_df['blockN'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # LEFT SUBPLOT: Human
    axes[0].plot(grouped_human['blockN'], grouped_human['rwd'], marker='o', label='Human rwd')
    axes[0].set_title("Inference Perf (First Encounters, Excl. 1st Stim)\nHuman")
    axes[0].set_xlabel("Block Number")
    axes[0].set_ylabel("Mean rwd (Accuracy)")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xticks(unique_blocks)
    axes[0].legend()

    # RIGHT SUBPLOT: Model
    axes[1].plot(grouped_model['blockN'], grouped_model['model_est_optimal_prob'], marker='o', color='orange', label='Model prob')
    axes[1].set_title("Inference Perf (First Encounters, Excl. 1st Stim)\nModel")
    axes[1].set_xlabel("Block Number")
    axes[1].set_ylabel("Mean model_est_optimal_prob")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(unique_blocks)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    sidebyside_path = os.path.join(save_folder, "inference_performance_human_vs_model_side_by_side.png")
    plt.savefig(sidebyside_path)
    plt.close()
    print(f"Side-by-side inference performance (human vs. model) saved to {sidebyside_path}")

###############################################
### Human & Model First-Encounter Plot
###############################################
def plot_inference_curve_by_block_and_encounter_human_vs_model_side_by_side(df: pd.DataFrame, save_folder: str):
    """
    Plots, side by side, the performance (Human = 'rwd', Model = 'model_est_optimal_prob')
    as a function of encounter number within each block. One separate line per blockN.

    Specifically:
      - On the x-axis: stimulusSlotID_encounterNInBlock (the 'encounter number')
      - On the left y-axis: mean rwd (Human)
      - On the right y-axis: mean model_est_optimal_prob (Model)
    """

    # 1) Exclude invalid trials
    valid_mask = (df['act'] != -1) & (df['bitResponseAIsCorr'] != -1)
    df = df[valid_mask].copy()

    # 2) Group by blockN AND by the encounter index within that block
    grouped_human = (
        df
        .groupby(['blockN', 'stimulusSlotID_encounterNInBlock'])['rwd']
        .mean()
        .reset_index()
    )
    grouped_model = (
        df
        .groupby(['blockN', 'stimulusSlotID_encounterNInBlock'])['model_est_optimal_prob']
        .mean()
        .reset_index()
    )

    # 3) Plot side-by-side subplots, share y-axis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # -- LEFT SUBPLOT: Human (rwd) --
    unique_blocks = sorted(grouped_human['blockN'].unique())
    for blk in unique_blocks:
        df_blk = grouped_human[grouped_human['blockN'] == blk].copy()
        df_blk = df_blk.sort_values('stimulusSlotID_encounterNInBlock')
        axes[0].plot(
            df_blk['stimulusSlotID_encounterNInBlock'],
            df_blk['rwd'],
            marker='o',
            label=f'Block {blk}'
        )
    axes[0].set_title("Human (rwd): Encounter # in Each Block")
    axes[0].set_xlabel("Encounter Number (stimulusSlotID_encounterNInBlock)")
    axes[0].set_ylabel("Mean rwd (Accuracy)")
    axes[0].set_ylim(0, 1)
    axes[0].legend(title="BlockN")

    # -- RIGHT SUBPLOT: Model (model_est_optimal_prob) --
    unique_blocks_model = sorted(grouped_model['blockN'].unique())
    for blk in unique_blocks_model:
        df_blk_m = grouped_model[grouped_model['blockN'] == blk].copy()
        df_blk_m = df_blk_m.sort_values('stimulusSlotID_encounterNInBlock')
        axes[1].plot(
            df_blk_m['stimulusSlotID_encounterNInBlock'],
            df_blk_m['model_est_optimal_prob'],
            marker='o',
            label=f'Block {blk}'
        )
    axes[1].set_title("Model (model_est_optimal_prob): Encounter # in Each Block")
    axes[1].set_xlabel("Encounter Number (stimulusSlotID_encounterNInBlock)")
    axes[1].set_ylabel("Mean model_est_optimal_prob")
    axes[1].set_ylim(0, 1)
    axes[1].legend(title="BlockN")

    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    out_path = os.path.join(save_folder, "inference_curve_by_block_and_encounter_human_vs_model_side_by_side.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved to {out_path}")



################################################################
### 2) NEW: Side-by-side first-encounter comparison (Human vs. Model)
################################################################
def plot_inference_curve_by_block_and_order_human_vs_model_side_by_side(df: pd.DataFrame, save_folder: str):
    """
    Side-by-side subplots for the *first encounter* in each block,
    comparing Human (rwd) vs. Model (model_est_optimal_prob) by stimulusSlotN (1..4).
    """
    # Exclude invalid
    valid_mask = (df['act'] != -1) & (df['bitResponseAIsCorr'] != -1)
    df = df[valid_mask].copy()

    # Filter to only first-encounter trials
    df_first_enc = df[df['stimulusSlotID_encounterNInBlock'] == 1].copy()

    # Group for human
    grouped_human = (
        df_first_enc
        .groupby(['blockN', 'stimulusSlotN'])['rwd']
        .mean()
        .reset_index()
    )
    # Group for model
    grouped_model = (
        df_first_enc
        .groupby(['blockN', 'stimulusSlotN'])['model_est_optimal_prob']
        .mean()
        .reset_index()
    )

    blocks = sorted(df_first_enc['blockN'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # LEFT SUBPLOT: Human
    for blk in blocks:
        df_blk = grouped_human[grouped_human['blockN'] == blk].copy().sort_values('stimulusSlotN')
        axes[0].plot(
            df_blk['stimulusSlotN'],
            df_blk['rwd'],
            marker='o',
            label=f'Block {blk}'
        )
    axes[0].set_title("Human (rwd): 1st Encounter by StimulusSlotN")
    axes[0].set_xlabel("StimulusSlotN")
    axes[0].set_ylabel("Mean rwd")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlim(1, 4)
    axes[0].legend(title="BlockN")

    # RIGHT SUBPLOT: Model
    for blk in blocks:
        df_blk = grouped_model[grouped_model['blockN'] == blk].copy().sort_values('stimulusSlotN')
        axes[1].plot(
            df_blk['stimulusSlotN'],
            df_blk['model_est_optimal_prob'],
            marker='o',
            label=f'Block {blk}'
        )
    axes[1].set_title("Model (model_est_optimal_prob): 1st Encounter by StimulusSlotN")
    axes[1].set_xlabel("StimulusSlotN")
    axes[1].set_ylabel("Mean model_est_optimal_prob")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlim(1, 4)
    axes[1].legend(title="BlockN")

    plt.tight_layout()
    out_path = os.path.join(save_folder, "inference_curve_by_block_and_order_human_vs_model_side_by_side.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Side-by-side first-encounter (Human vs. Model) plot saved to {out_path}")



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

    plot_session(
        choices,
        rewards,
        timeseries=disrnn_activations_sorted,
        timeseries_name='Network Activations',
        labels=labels_with_sigmas,
        save_folder_name=save_folder_name
    )


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

    If train_test_split_variables is provided, it uses those to build a full dataframe 
    (as originally in create_dataframe_for_plotting). 

    If train_test_split_variables is None, it assumes session-level plotting 
    and requires bitResponseAIsCorr and P_A.
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
        context = train_test_split_variables["context_train"][:, :, 0].ravel()
        blockN = train_test_split_variables["blockN_train"][:, :, 0].ravel()
        trialNInBlock = train_test_split_variables["trialNInBlock_train"][:, :, 0].ravel()
        bitResponseAIsCorr_full = train_test_split_variables["bitResponseAIsCorr_train"][:, :, 0].ravel()
        P_A_full = train_test_split_variables["P_A_train"][:, :, 0].ravel()

        stimulusSlotN = train_test_split_variables["stimulusSlotN_train"][:, :, 0].ravel()
        stimulusSlotID = train_test_split_variables["stimulusSlotID_train"][:, :, 0].ravel()
        stimulusSlotID_encounterNInBlock = train_test_split_variables["stimulusSlotID_encounterNInBlock_train"][:, :, 0].ravel()

        # "rwd" logic
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
            'stimulusSlotN': stimulusSlotN,
            'stimulusSlotID': stimulusSlotID,
            'stimulusSlotID_encounterNInBlock': stimulusSlotID_encounterNInBlock
        })

        df['model_est_optimal_prob'] = np.where(df['bitResponseAIsCorr'] == 1, df['pred'], 1 - df['pred'])
        df['human_chosen_action_prob'] = np.where(df['act'] == 1, df['P_A'], 1 - df['P_A'])
        df['rwd_manual'] = np.where(df['act'] == df['bitResponseAIsCorr'], 1, 0)
        df["pred_action"] = (df["pred"] > 0.5).astype(int)

    else:
        if bitResponseAIsCorr is None or P_A is None:
            raise ValueError("bitResponseAIsCorr and P_A must be provided if train_test_split_variables is None.")

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


def plot_average_block_3_model_probability(df, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    
    df_block3 = df[(df['blockN'] == 3) & (df['stimulusSlotID_encounterNInBlock'] == 1)].copy()
    
    df_block3 = df_block3.sort_values('trialNInBlock')
    print(df_block3)
    
    n = len(df_block3)
    df_block3['encounter_order'] = np.tile([1,2,3,4], n // 4)
    
    df_mean = df_block3.groupby('encounter_order', as_index=False)['model_est_optimal_prob'].mean()
    
    plt.figure(figsize=(6,4))
    sns.lineplot(
        data=df_mean,
        x='encounter_order',
        y='model_est_optimal_prob',
        marker='o',
        color='blue'
    )

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
    dataset_path = "dataset/tensor_for_dRNN_desc-syn_nSubs-2000_nSessions-2_nBlocks-7_nTrialsPerBlock-25_b-multiple_20250131.mat"
    dataset_train, dataset_test, train_test_split_variables = preprocess_data(dataset_type, dataset_path, 0.1)

    with open(saved_checkpoint_pth, 'rb') as file:
        checkpoint = pickle.load(file)
    
    args_dict = checkpoint['args_dict']
    print(args_dict)
    disrnn_params = checkpoint['disrnn_params']
    print(f'Loaded disrnn_params from {saved_checkpoint_pth}')

    def make_disrnn():
      model = disrnn.HkDisRNN(
          obs_size=10,
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

    print('Normalized Likelihoods and Accuracies for disRNN')
    print('Training Dataset')
    train_norm_likelihood, _ = compute_log_likelihood_and_accuracy(xs_train, ys_train, make_disrnn, disrnn_params)

    pred_train, ys_train, xs_train = evaluate(xs_train, ys_train, make_disrnn, disrnn_params)
    pred_test, ys_test, xs_test = evaluate(xs_test, ys_test, make_disrnn, disrnn_params)
    df_train = create_dataframe_for_plotting(pred_train, ys_train, xs_train, train_test_split_variables=train_test_split_variables)
    log_probs_train = np.log(df_train['human_chosen_action_prob'])
    normalized_likelihood_upper_bound_train = np.exp(np.mean(log_probs_train))
    print("Model Normalized likelihood Upperbound for training", normalized_likelihood_upper_bound_train)

    save_folder_name = os.path.join('plots', saved_checkpoint_pth.split('checkpoints_0128/')[1].split('.pkl')[0])
    os.makedirs(save_folder_name, exist_ok=True)

    plot_training_results_one_session(df_train, save_folder_name)
    plot_training_results(df_train, dataset_train, save_folder_name)
    plot_training_results_per_block(df_train, save_folder_name)
    plot_bottlenecks(disrnn_params, save_folder_name)

    ####################################################
    # 1) Inference performance side-by-side (Human vs. Model),
    #    first encounters, excluding the 1st introduced stimulus
    ####################################################
    plot_inference_performance_human_vs_model_side_by_side(df_train, save_folder_name)

    ####################################################
    # 2) Side-by-side first-encounter comparison
    ####################################################
    plot_inference_curve_by_block_and_order_human_vs_model_side_by_side(df_train, save_folder_name)

    ####################################################
    # 3) Side-by-side all-encounters comparison
    ####################################################
    plot_inference_curve_by_block_and_encounter_human_vs_model_side_by_side(df_train, save_folder_name)

    ####################################################
    # Original single-line plots
    ####################################################
    #plot_inference_performance(df_train, save_folder_name)
    #plot_inference_curve_by_block_and_order(df_train, save_folder_name)
    #plot_inference_curve_by_block_and_order_human(df_train, save_folder_name)

    session_i = 0
    plot_latent_activations_for_session(
        sess_i=session_i,
        xs_train=xs_train, 
        pred_train=pred_train, 
        act_train=ys_test, 
        bitResponseAIsCorr_train=train_test_split_variables["bitResponseAIsCorr_train"], 
        P_A_train=train_test_split_variables['P_A_train'], 
        disrnn_params=disrnn_params, 
        make_disrnn=make_disrnn,
        save_folder_name=save_folder_name,
        train_test_split_variables=train_test_split_variables
    )

    # Optional
    # plot_average_block_3_model_probability(df_train, save_folder_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", nargs=1, type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--saved_checkpoint_pth", type=str, required=True, help="Saved checkpoint path for evaluation.")
    args = parser.parse_args()

    main(args.seed, args.saved_checkpoint_pth)
