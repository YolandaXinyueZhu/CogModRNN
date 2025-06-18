import os
import sys
import pickle
import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
import warnings
import scipy.io as sio
import copy
from datetime import datetime
from typing import Optional, List
import math
import argparse
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


# Import custom libraries
from disentangled_rnns_new.disentangled_rnns.library import disrnn  # 2025 version
from disentangled_rnns_new.disentangled_rnns.library import rnn_utils
from disentangled_rnns_new.disentangled_rnns.library import multisubject_disrnn
from train_disrnn_new import (
    load_data, 
    to_onehot, 
    zs_to_onehot,
    create_train_test_datasets, 
    preprocess_data
)
from train_disrnn_new import build_disrnn_config

# Suppress warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
# Fontsizes and formatting for plots.
small = 15
medium = 18
large = 20
mpl.rcParams['grid.color'] = 'none'
mpl.rcParams['axes.facecolor'] = 'white'
plt.rcParams['svg.fonttype'] = 'none'

def dataset_size(xs, ys):
    """Calculate the size of the dataset in terms of number of samples."""
    return xs.shape[1], ys.shape[1] # 1800

def compute_log_likelihood_and_accuracy(xs, ys, model_fun, params):
    """Compute the normalized likelihood and accuracy of the model on (xs, ys)."""
    n_trials_per_session, n_sessions = ys.shape[:2]
    model_outputs, _ = rnn_utils.eval_network(model_fun, params, xs)
    predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))

    log_likelihood = 0
    correct_predictions = 0
    n = 0  # Total number of valid trials.

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
    print(f'Model Normalized Likelihood for Human-Chosen Actions: {100 * normalized_likelihood:.3f}%')
    print(f'Model Thresholded Classifier Accuracy for Human-Chosen Actions: {accuracy * 100:.3f}%')
    return normalized_likelihood, accuracy


def evaluate(xs, actual_choices, model_fun, params):
    """Return model predictions (logits) and the original actual_choices, xs."""
    model_outputs, _ = rnn_utils.eval_network(model_fun, params, xs)
    predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))
    return predicted_log_choice_probabilities, actual_choices, xs


def plot_inference_performance(df: pd.DataFrame, save_folder: str):
    """
    Side-by-side subplots for 'first encounters' (excluding the very first introduced stimulusSlotN=1),
    comparing Human vs. Model performance by block—overlaid across all sessions in the same plot.

    LEFT SUBPLOT: Human's mean rwd (one line per session)
    RIGHT SUBPLOT: Model's mean model_est_optimal_prob (one line per session)
    """
    # Exclude any invalid trials
    valid_mask = (df['human_chosen_action'] != -1) & (df['correct_optimal_choice'] != -1)
    df = df[valid_mask].copy()

    # Filter to first encounters (except the very first stimulus in each block)
    inference_df = df[
        (df['stimulusSlotID_encounterNInBlock'] == 1) &
        (df['stimulusSlotN'] > 1)
    ].copy()

    # Make sure we have a sessionN column
    if 'sessionN' not in inference_df.columns:
        raise ValueError("plot_inference_performance_human_vs_model_side_by_side requires 'sessionN' in df.")

    # Identify unique sessions
    unique_sessions = sorted(inference_df['sessionN'].unique())
    if not unique_sessions:
        print("No sessions found for first-encounter trials.")
        return

    # Prepare subplots: left for Human, right for Model
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Loop over each session and plot on the same axes
    for sess in unique_sessions:
        df_sess = inference_df[inference_df['sessionN'] == sess]

        # Group for Human (this session, by block)
        grouped_human = (
            df_sess
            .groupby('blockN')['rwd']
            .mean()
            .reset_index()
        )
        # Group for Model (this session, by block)
        grouped_model = (
            df_sess
            .groupby('blockN')['model_est_optimal_prob']
            .mean()
            .reset_index()
        )

        unique_blocks = sorted(df_sess['blockN'].unique())

        # LEFT SUBPLOT: Human (one line per session)
        axes[0].plot(
            grouped_human['blockN'],
            grouped_human['rwd'],
            marker='o',
            label=f'Session {sess}'
        )

        # RIGHT SUBPLOT: Model (one line per session)
        axes[1].plot(
            grouped_model['blockN'],
            grouped_model['model_est_optimal_prob'],
            marker='o',
            label=f'Session {sess}'
        )

    # Final formatting for LEFT SUBPLOT: Human
    axes[0].set_title("Human Inference Performance\n(First Encounters, Excl. 1st Stimulus)")
    axes[0].set_xlabel("Block Number")
    axes[0].set_ylabel("Mean rwd (Accuracy)")
    axes[0].set_ylim(0, 1.0)
    # Collect all block numbers across sessions for consistent ticks
    all_blocks = sorted(inference_df['blockN'].unique())
    axes[0].set_xticks(all_blocks)
    axes[0].legend(title="Session", fontsize='small')

    # Final formatting for RIGHT SUBPLOT: Model
    axes[1].set_title("Model Inference Performance\n(First Encounters, Excl. 1st Stimulus)")
    axes[1].set_xlabel("Block Number")
    axes[1].set_ylabel("Mean model_est_optimal_prob")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(all_blocks)
    axes[1].legend(title="Session", fontsize='small')

    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    out_path = os.path.join(save_folder, "inference_performance_all_sessions.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Overlaid inference performance (human vs. model) across all sessions saved to {out_path}")


def plot_inference_curve_by_block_and_order_human_vs_model_side_by_side(df: pd.DataFrame, save_folder: str):
    """
    Side-by-side subplots for the *first encounter* in each block,
    comparing Human (rwd) vs. Model (model_est_optimal_prob) by stimulusSlotN (1..4).
    """
    # Exclude invalid
    valid_mask = (df['human_chosen_action'] != -1) & (df['correct_optimal_choice'] != -1)
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


def plot_session(choices: np.ndarray,
                 rewards: np.ndarray,
                 timeseries: np.ndarray,
                 timeseries_name: str,
                 save_folder_name,
                 labels: Optional[List[str]] = None,
                 fig_ax: Optional = None,
                 session_number: Optional[int] = None,
                 block_changes: Optional[np.ndarray] = None,
                 block_numbers: Optional[np.ndarray] = None):
    """
    Utility to plot the timeseries of latents (or other signals) along with 
    choice/reward markers. Also marks block transitions if provided.
    """
    choose_high = choices == 1
    choose_low = choices == 0
    rewarded = rewards == 1

    # Just a guess for plotting markers
    y_high = np.max(timeseries) + 0.1 if timeseries.size > 0 else 1.1
    y_low = np.min(timeseries) - 0.1 if timeseries.size > 0 else -0.1

    if labels is not None:
        # If we have multiple latents in the last dimension:
        if timeseries.ndim == 2:  # (trials, latents)
            timeseries = timeseries[:, None, :]  # shape: (trials, 1, latents)

        if len(labels) != timeseries.shape[2]:
            raise ValueError('labels length must match timeseries.shape[2].')

        for i in range(timeseries.shape[2]):
            fig, ax = plt.subplots(1, figsize=(20, 10))
            ax.plot(timeseries[:, :, i], label=labels[i])
            ax.legend(bbox_to_anchor=(1, 1))

            # Add vertical lines for block changes
            if block_changes is not None and block_numbers is not None:
                for change_idx in block_changes:
                    ax.axvline(x=change_idx, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                    block_num = block_numbers[change_idx]
                    ax.text(
                        change_idx, y_high, f'Block {block_num}', rotation=90,
                        verticalalignment='top', horizontalalignment='right', 
                        fontsize=8, color='gray', alpha=0.7
                    )

            # Add session number to title
            title = f'Session {session_number} - {timeseries_name} ({labels[i]})' if session_number is not None else f'{timeseries_name} ({labels[i]})'
            ax.set_title(title)

            if choices.max() <= 1:
                # Rewarded high
                ax.scatter(
                    np.argwhere(choose_high & rewarded),
                    y_high * np.ones(np.sum(choose_high & rewarded)),
                    color='green',
                    marker=3
                )
                ax.scatter(
                    np.argwhere(choose_high & rewarded),
                    y_high * np.ones(np.sum(choose_high & rewarded)),
                    color='green',
                    marker='|'
                )
                # Omission high
                ax.scatter(
                    np.argwhere(choose_high & (1 - rewarded)),
                    y_high * np.ones(np.sum(choose_high & (1 - rewarded))),
                    color='red',
                    marker='|'
                )

                # Rewarded low
                ax.scatter(
                    np.argwhere(choose_low & rewarded),
                    y_low * np.ones(np.sum(choose_low & rewarded)),
                    color='green',
                    marker='|'
                )
                ax.scatter(
                    np.argwhere(choose_low & rewarded),
                    y_low * np.ones(np.sum(choose_low & rewarded)),
                    color='green',
                    marker=2
                )
                # Omission Low
                ax.scatter(
                    np.argwhere(choose_low & (1 - rewarded)),
                    y_low * np.ones(np.sum(choose_low & (1 - rewarded))),
                    color='red',
                    marker='|'
                )

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
        # Single-latent or single-signal case
        fig, ax = plt.subplots(1, figsize=(20, 10))
        ax.plot(timeseries)

        # Add vertical lines for block changes
        if block_changes is not None and block_numbers is not None:
            for change_idx in block_changes:
                ax.axvline(x=change_idx, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                block_num = block_numbers[change_idx]
                ax.text(
                    change_idx, y_high, f'Block {block_num}', rotation=90,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=8, color='gray', alpha=0.7
                )

        # Add session number to title
        title = f'Session {session_number} - {timeseries_name}' if session_number is not None else timeseries_name
        ax.set_title(title)

        if choices.max() <= 1:
            # Rewarded high
            ax.scatter(
                np.argwhere(choose_high & rewarded),
                y_high * np.ones(np.sum(choose_high & rewarded)),
                color='green',
                marker=3
            )
            ax.scatter(
                np.argwhere(choose_high & rewarded),
                y_high * np.ones(np.sum(choose_high & rewarded)),
                color='green',
                marker='|'
            )
            # Omission high
            ax.scatter(
                np.argwhere(choose_high & (1 - rewarded)),
                y_high * np.ones(np.sum(choose_high & (1 - rewarded))),
                color='red',
                marker='|'
            )

            # Rewarded low
            ax.scatter(
                np.argwhere(choose_low & rewarded),
                y_low * np.ones(np.sum(choose_low & rewarded)),
                color='green',
                marker='|'
            )
            ax.scatter(
                np.argwhere(choose_low & rewarded),
                y_low * np.ones(np.sum(choose_low & rewarded)),
                color='green',
                marker=2
            )
            # Omission Low
            ax.scatter(
                np.argwhere(choose_low & (1 - rewarded)),
                y_low * np.ones(np.sum(choose_low & (1 - rewarded))),
                color='red',
                marker='|'
            )

        save_dir = save_folder_name
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, "plot_session.png")
        ax.set_xlabel('Trial')
        ax.set_ylabel(timeseries_name)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Session plot saved to {plot_path}")


def create_dataframe_for_plotting(predictions, 
                                  ys_train, 
                                  xs,
                                  train_test_split_variables=None,
                                  bitResponseAIsCorr=None,
                                  P_A=None):

    action_codes = ys_train[:, :, 0].copy()
    other_choices = (action_codes != 0) & (action_codes != 1)
    action_codes[other_choices] = 2
    action_codes = action_codes.astype(int)  

    nan_column = np.full(predictions.shape[:2] + (1,), np.nan)
    predictions_padded = np.concatenate((predictions, nan_column), axis=2)

    trial, sequence = action_codes.shape
    prediction_prob = np.empty((trial, sequence))
    for t in range(trial):
        for s in range(sequence):
            code = action_codes[t, s]
            prediction_prob[t, s] = predictions_padded[t, s, code]

    prediction_prob = prediction_prob.reshape(ys_train.shape)

    context = train_test_split_variables["context_train"][:, :, 0].ravel()
    blockN = train_test_split_variables["blockN_train"][:, :, 0].ravel()
    sessionN = train_test_split_variables["sessionN_train"][:, :, 0].ravel()
    trialNInBlock = train_test_split_variables["trialNInBlock_train"][:, :, 0].ravel()
    correct_optimal_choice_full = train_test_split_variables["bitResponseAIsCorr_train"][:, :, 0].ravel()
    prob_choosing_act_1_A_full = train_test_split_variables["P_A_train"][:, :, 0].ravel()

    stimulusSlotN = train_test_split_variables["stimulusSlotN_train"][:, :, 0].ravel()
    stimulusSlotID = train_test_split_variables["stimulusSlotID_train"][:, :, 0].ravel()
    stimulusSlotID_encounterNInBlock = train_test_split_variables["stimulusSlotID_encounterNInBlock_train"][:, :, 0].ravel()

    # "rwd" logic: reward at trial t is the outcome from the next time-step's xs (column 0).
    # This often depends on how your dataset is structured, but here's a typical approach:
    rwd = np.vstack((xs[1:, :, 0], np.zeros((1, xs.shape[1])))).ravel()
    
    df = pd.DataFrame({
        'context': context,
        'blockN': blockN,
        'sessionN': sessionN,
        'trialNInBlock': trialNInBlock,
        'human_chosen_action': ys_train[:, :, 0].ravel(),  # human-chosen action
        'model_predicted_action': np.exp(predictions[:, :, 1].ravel()),  # the prob of "action=1" from model
        'model_predicted_probability_of_the_human_chosen_action': np.exp(prediction_prob.ravel()), 
        'correct_optimal_choice': correct_optimal_choice_full,
        'prob_choosing_act_1_A': prob_choosing_act_1_A_full,
        'rwd': rwd,
        'stimulusSlotN': stimulusSlotN,
        'stimulusSlotID': stimulusSlotID,
        'stimulusSlotID_encounterNInBlock': stimulusSlotID_encounterNInBlock
    })

    df['model_est_optimal_prob'] = np.where(
        df['correct_optimal_choice'] == 1, 
        df['model_predicted_action'], 
        1 - df['model_predicted_action']
    )
    df['human_chosen_action_prob'] = np.where(
        df['human_chosen_action'] == 1, 
        df['prob_choosing_act_1_A'], 
        1 - df['prob_choosing_act_1_A']
    )
    df['rwd_manual'] = np.where(
        df['human_chosen_action'] == df['correct_optimal_choice'], 
        1, 
        0
    )
    df["pred_action"] = (df["model_predicted_action"] > 0.5).astype(int)

    return df


def plot_training_results_one_session(df_train, save_folder):
    """Example: Plotting per-session results from a data frame."""
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
        x='trial_num_in_session', y='rwd_manual', data=session_df,
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
        x='trial_num_in_session', y='model_predicted_probability_of_the_human_chosen_action', data=session_df,
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
    """Example: Plotting results on a per-block basis."""
    os.makedirs(save_folder, exist_ok=True)
    unique_blocks = df_train['blockN'].unique()

    for block in unique_blocks:
        df_block = df_train[df_train['blockN'] == block]

        # Plot 1: Probability of chosen action
        plt.figure(figsize=(16, 8))
        sns.lineplot(
            x='trialNInBlock', y='human_chosen_action_prob', 
            data=df_block, label='Human Proportion of Chosen Actions', color='blue'
        )
        sns.lineplot(
            x='trialNInBlock', y='model_predicted_probability_of_the_human_chosen_action', 
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

        # Plot 2: Probability of optimal action
        plt.figure(figsize=(16, 8))
        sns.lineplot(
            x='trialNInBlock', y='rwd_manual', 
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
    """Generic training result plots across all trials."""
    _trial_num = np.tile(np.arange(dataset_train._xs.shape[0]), [dataset_train._xs.shape[1], 1]).transpose()

    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='model_predicted_probability_of_the_human_chosen_action', label='Normalized Log-Likelihood')
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
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='rwd_manual', label='Human Proportion of Optimal Choices (Blue Line)')
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
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='model_predicted_probability_of_the_human_chosen_action', label='Model Chosen Action Probability (Red Line)', color='red')
    plt.legend()
    plt.xlabel('Trial Number')
    plt.ylabel('Probability')
    plt.xlim(left=1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title('Training Data: Probability of Chosen Actions over Trials')
    plt.savefig(os.path.join(save_folder, 'probability_chosen_actions_over_trials_train.png'))
    plt.close()

    # Probability per trial within block (human vs. model chosen action)
    plt.figure(figsize=(16, 8))
    df_agg = df_train.groupby('trialNInBlock').agg({
        'human_chosen_action_prob': 'mean',
        'model_predicted_probability_of_the_human_chosen_action': 'mean'
    }).reset_index()

    sns.lineplot(x='trialNInBlock', y='human_chosen_action_prob', data=df_agg, label='Human Proportion of Chosen Actions', color='blue')
    sns.lineplot(x='trialNInBlock', y='model_predicted_probability_of_the_human_chosen_action', data=df_agg, label='Model-Estimated Probability of Chosen Actions', color='orange')
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

    # Probability of optimal action vs. trial in block
    plt.figure(figsize=(16, 8))
    df_agg = df_train.groupby('trialNInBlock').agg({
        'rwd_manual': 'mean',
        'model_est_optimal_prob': 'mean',
    }).reset_index()

    sns.lineplot(x='trialNInBlock', y='rwd_manual', data=df_agg, label='Human Proportion of Optimal Actions', color='purple')
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



def plot_model_probs_by_block_and_session(df_train, save_folder):
    """
    Like plot_avg_probs_by_block_and_session, but only shows the model:
      • top panel: model_est_optimal_prob
      • bottom panel: model_predicted_probability_of_the_human_chosen_action

    X‐axis is concatenated trial-in-block → block → session, with session & block boundaries.
    """
    os.makedirs(save_folder, exist_ok=True)

    # copy & ensure integer indices
    df = df_train.copy()
    df['blockN']        = df['blockN'].astype(int)
    df['trialNInBlock'] = df['trialNInBlock'].astype(int)
    df['sessionN']      = df['sessionN'].astype(int)

    # sizes
    T_block = df['trialNInBlock'].max()
    B       = df['blockN'].nunique()
    T_sess  = T_block * B

    # zero-based session
    df['sess0'] = df['sessionN'] - 1

    # concatenated x
    df['x'] = (
          df['sess0'] * T_sess
        + (df['blockN'] - 1) * T_block
        + df['trialNInBlock']
    )
    df = df[df['x'] >= 1]

    # average
    df_mean = df.groupby('x').agg({
        'model_est_optimal_prob':                                 'mean',
        'model_predicted_probability_of_the_human_chosen_action': 'mean'
    }).reset_index()

    n_sessions = int(df['sess0'].max()) + 1
    max_x      = df_mean['x'].max()

    fig, (ax_opt, ax_ch) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ─── Top panel: Model Optimal‐Choice Probability ─────────────────────────
    ax_opt.plot(df_mean['x'], df_mean['model_est_optimal_prob'],
                label='Model P(optimal)', color='C1', linewidth=2)
    ax_opt.set_ylabel('P(optimal choice)')
    ax_opt.set_title('Model Optimal‐Choice Probability\nby Trial→Block→Session')
    ax_opt.legend(loc='lower right')

    # ─── Bottom panel: Model Chosen‐Action Probability ────────────────────────
    ax_ch.plot(df_mean['x'],
               df_mean['model_predicted_probability_of_the_human_chosen_action'],
               label='Model P(chosen action)', color='C2', linewidth=2)
    ax_ch.set_ylabel('P(chosen action)')
    ax_ch.set_xlabel('Concatenated trial (blocks→sessions)')
    ax_ch.set_title('Model Chosen‐Action Probability\nby Trial→Block→Session')
    ax_ch.legend(loc='lower right')

    # draw session & block boundaries
    for s in range(n_sessions):
        xsess = s * T_sess
        if 1 <= xsess <= max_x:
            ax_opt.axvline(xsess, color='gray', linestyle=':', lw=1)
            ax_ch.axvline(xsess, color='gray', linestyle=':', lw=1)
        for b in range(1, B+1):
            xblk = s*T_sess + (b-1)*T_block
            if 1 <= xblk <= max_x:
                ax_opt.axvline(xblk, color='lightgray', linestyle='--', lw=0.5)
                ax_ch.axvline(xblk, color='lightgray', linestyle='--', lw=0.5)

    ax_ch.set_xlim(1, max_x)
    plt.tight_layout()

    out = os.path.join(save_folder, 'model_probs_block_and_session.png')
    plt.savefig(out)
    plt.close(fig)
    print(f"Saved model‐only plot to {out}")


def plot_human_probs_by_block_and_session(df_train, save_folder):
    """
    Like plot_avg_probs_by_block_and_session, but only shows the human:
      • top panel: rwd_manual   (P(optimal choice))
      • bottom panel: human_chosen_action_prob

    X‐axis is concatenated trial-in-block → block → session, with session & block boundaries.
    """
    os.makedirs(save_folder, exist_ok=True)

    # make sure our key columns are ints
    df = df_train.copy()
    df['blockN']        = df['blockN'].astype(int)
    df['trialNInBlock'] = df['trialNInBlock'].astype(int)
    df['sessionN']      = df['sessionN'].astype(int)

    # how many trials per block, blocks per session
    T_block = df['trialNInBlock'].max()
    B       = df['blockN'].nunique()
    T_sess  = T_block * B

    # zero-based session index
    df['sess0'] = df['sessionN'] - 1

    # concatenated x
    df['x'] = (
          df['sess0'] * T_sess
        + (df['blockN'] - 1) * T_block
        + df['trialNInBlock']
    )
    df = df[df['x'] >= 1]

    # average across sessions & blocks at each x
    df_mean = df.groupby('x').agg({
        'rwd_manual':                  'mean',
        'human_chosen_action_prob':    'mean',
    }).reset_index()

    n_sessions = int(df['sess0'].max()) + 1
    max_x      = df_mean['x'].max()

    fig, (ax_opt, ax_ch) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ─── Top panel: Human Optimal‐Choice Probability ─────────────────────────
    ax_opt.plot(df_mean['x'], df_mean['rwd_manual'],
                label='Human P(optimal)', color='C0', linewidth=2)
    ax_opt.set_ylabel('P(optimal choice)')
    ax_opt.set_title('Human Optimal‐Choice Probability\nby Trial→Block→Session')
    ax_opt.legend(loc='lower right')

    # ─── Bottom panel: Human Chosen‐Action Probability ────────────────────────
    ax_ch.plot(df_mean['x'], df_mean['human_chosen_action_prob'],
               label='Human P(chosen action)', color='C3', linewidth=2)
    ax_ch.set_ylabel('P(chosen action)')
    ax_ch.set_xlabel('Concatenated trial (blocks→sessions)')
    ax_ch.set_title('Human Chosen‐Action Probability\nby Trial→Block→Session')
    ax_ch.legend(loc='lower right')

    # draw session & block boundaries
    for s in range(n_sessions):
        xsess = s * T_sess
        if 1 <= xsess <= max_x:
            ax_opt.axvline(xsess, color='gray', linestyle=':', lw=1)
            ax_ch.axvline(xsess, color='gray', linestyle=':', lw=1)
        for b in range(1, B+1):
            xblk = s*T_sess + (b-1)*T_block
            if 1 <= xblk <= max_x:
                ax_opt.axvline(xblk, color='lightgray', linestyle='--', lw=0.5)
                ax_ch.axvline(xblk, color='lightgray', linestyle='--', lw=0.5)

    ax_ch.set_xlim(1, max_x)
    plt.tight_layout()

    out = os.path.join(save_folder, 'human_probs_block_and_session.png')
    plt.savefig(out)
    plt.close(fig)
    print(f"Saved human‐only plot to {out}")


def plot_avg_probs_by_block_and_session(df_train, save_folder):
    """
    Plot average human/model P(optimal) and P(chosen)
    concatenated by trial-in-block → block → session,
    without gaps at the start or end.
    """
    os.makedirs(save_folder, exist_ok=True)

    # ensure blockN, trialNInBlock, sessionN are ints
    df = df_train.copy()
    df['blockN']         = df['blockN'].astype(int)
    df['trialNInBlock']  = df['trialNInBlock'].astype(int)
    df['sessionN']       = df['sessionN'].astype(int)

    # get sizes
    T_block = df['trialNInBlock'].max()              # trials per block
    B       = df['blockN'].nunique()                 # blocks per session
    T_sess  = T_block * B                            # trials per session

    # zero-based session index
    df['sess0'] = df['sessionN'] - 1                 # now 0,1,...

    # concatenated x-axis
    df['x'] = (
          df['sess0'] * T_sess
        + (df['blockN'] - 1) * T_block
        + df['trialNInBlock']
    )

    # drop any x ≤ 0 (shouldn't be any after cast, but just in case)
    df = df[df['x'] >= 1]

    # aggregate means
    df_mean = df.groupby('x').agg({
        'rwd_manual':                                             'mean',
        'model_est_optimal_prob':                                 'mean',
        'human_chosen_action_prob':                               'mean',
        'model_predicted_probability_of_the_human_chosen_action': 'mean'
    }).reset_index()

    # how many sessions
    n_sessions = int(df['sess0'].max()) + 1

    # prepare figure
    fig, (ax_opt, ax_ch) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # — Optimal‐choice panel —
    ax_opt.plot(df_mean['x'], df_mean['rwd_manual'],
                label='Human optimal (rwd_manual)',   linewidth=2)
    ax_opt.plot(df_mean['x'], df_mean['model_est_optimal_prob'],
                label='Model optimal', linestyle='--', linewidth=2)
    ax_opt.set_ylabel('P(optimal choice)')
    ax_opt.legend(loc='upper right')
    ax_opt.set_title('Average Optimal‐Choice Probability\nby Trial→Block→Session')

    # — Chosen‐action panel —
    ax_ch.plot(df_mean['x'], df_mean['human_chosen_action_prob'],
               label='Human chosen', linewidth=2)
    ax_ch.plot(df_mean['x'],
               df_mean['model_predicted_probability_of_the_human_chosen_action'],
               label='Model chosen', linestyle='--', linewidth=2)
    ax_ch.set_ylabel('P(chosen action)')
    ax_ch.set_xlabel('Concatenated trial (blocks→sessions)')
    ax_ch.legend(loc='upper right')

    # vertical lines only where they fall in [1, max_x]
    max_x = df_mean['x'].max()
    for s in range(n_sessions):
        # session boundary
        xsess = s * T_sess
        if 1 <= xsess <= max_x:
            ax_opt.axvline(xsess, color='gray', linestyle=':', linewidth=1)
            ax_ch.axvline(xsess, color='gray', linestyle=':', linewidth=1)
        # block boundaries within session
        for b in range(1, B+1):
            xblk = s*T_sess + (b-1)*T_block
            if 1 <= xblk <= max_x:
                ax_opt.axvline(xblk, color='lightgray', linestyle='--', linewidth=0.5)
                ax_ch.axvline(xblk, color='lightgray', linestyle='--', linewidth=0.5)

    # clamp x-axis
    ax_ch.set_xlim(1, max_x)

    plt.tight_layout()
    out = os.path.join(save_folder, 'avg_probs_block_and_session.png')
    plt.savefig(out)
    plt.close(fig)
    print(f"Saved avg‐prob plot to {out}")

def plot_bottlenecks(
    params: hk.Params,
    disrnn_config: disrnn.DisRnnConfig,
    save_folder_name,
    sort_latents: bool = True,
) -> plt.Figure:
  """Plot the bottleneck sigmas from an hk.DisentangledRNN."""

  save_dir = save_folder_name
  os.makedirs(save_dir, exist_ok=True)
  plot_path = os.path.join(save_dir, "bottleneck.png")

  if True:
    params_disrnn = params['hk_disentangled_rnn']
    subject_embedding_size = 0
    update_input_names = disrnn_config.x_names
    # For update_sigmas: concatenate transposed reparameterized sigmas
    # Order of inputs to update nets: observations, latents
    update_obs_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_obs_sigma_params']
        )
    )
    update_latent_sigmas_t = np.transpose(
        disrnn.reparameterize_sigma(
            params_disrnn['update_net_latent_sigma_params']
        )
    )
    update_sigmas = np.concatenate(
        (update_obs_sigmas_t, update_latent_sigmas_t), axis=1)
    choice_sigmas = np.array(
        disrnn.reparameterize_sigma(
            np.transpose(params_disrnn['choice_net_sigma_params'])
        )
    )
  else:
    raise ValueError(
        'plot_bottlenecks only supports DisRnnConfig and'
        ' MultisubjectDisRnnConfig.'
    )

  latent_sigmas = np.array(
      disrnn.reparameterize_sigma(params_disrnn['latent_sigma_params'])
  )

  if sort_latents:
    latent_sigma_order = np.argsort(latent_sigmas)
    latent_sigmas = latent_sigmas[latent_sigma_order]
    print("latent sigmas",latent_sigmas)

    # Sort choice sigmas based on the order of latents, keeping subject
    # embedding dimensions first if they exist.
    choice_sigma_order = np.concatenate(
        (
            np.arange(0, subject_embedding_size),
            subject_embedding_size + latent_sigma_order,
        ),
        axis=0,
    )
    choice_sigmas = choice_sigmas[choice_sigma_order]

    # Sort update sigmas based on the order of latents, keeping subject
    # embedding dimensions first if they exist, then observations, then latents.
    non_latent_input_size = subject_embedding_size + disrnn_config.obs_size
    update_sigma_order = np.concatenate(
        (
            np.arange(0, non_latent_input_size, 1),
            non_latent_input_size + latent_sigma_order,
        ),
        axis=0,
    )
    update_sigmas = update_sigmas[latent_sigma_order, :]
    update_sigmas = update_sigmas[:, update_sigma_order]

  latent_names = np.arange(1, disrnn_config.latent_size + 1)
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))

  # Plot Latent Bottlenecks on axes[0]
  im1 = axes[0].imshow(np.swapaxes([1 - latent_sigmas], 0, 1), cmap='Oranges')
  im1.set_clim(vmin=0, vmax=1)
  axes[0].set_yticks(
      ticks=range(disrnn_config.latent_size),
      labels=latent_names,
      fontsize=small,
  )
  axes[0].set_xticks(ticks=[])
  axes[0].set_ylabel('Latent #', fontsize=medium)
  axes[0].set_title('Latent Bottlenecks', fontsize=large)

  # Plot Choice Bottlenecks on axes[1]
  # These bottlenecks apply to the inputs of the choice network:
  # [subject embeddings, latents]
  choice_input_dim = subject_embedding_size + disrnn_config.latent_size
  choice_input_names = np.concatenate((
      [f'SubjEmb {i+1}' for i in range(subject_embedding_size)],
      [f'Latent {i}' for i in latent_names]
  ))
  im2 = axes[1].imshow(np.swapaxes([1 - choice_sigmas], 0, 1), cmap='Oranges')
  im2.set_clim(vmin=0, vmax=1)
  axes[1].set_yticks(
      ticks=range(choice_input_dim), labels=choice_input_names, fontsize=small
  )
  axes[1].set_xticks(ticks=[])
  axes[1].set_ylabel('Choice Network Input', fontsize=medium)
  axes[1].set_title('Choice Network Bottlenecks', fontsize=large)

  # Plot Update Bottlenecks on axes[2]
  im3 = axes[2].imshow(1 - update_sigmas, cmap='Oranges')
  im3.set_clim(vmin=0, vmax=1)
  cbar = fig.colorbar(im3, ax=axes[2])
  # Y-axis corresponds to the target latent (sorted if sort_latents=True)
  cbar.ax.tick_params(labelsize=small)
  axes[2].set_yticks(
      ticks=range(disrnn_config.latent_size),
      labels=latent_names,
      fontsize=small,
  )
  # X-axis corresponds to the inputs to the update network:
  # [subject embeddings, observations, latents]
  xlabels = update_input_names + [f'Latent {i}' for i in latent_names]
  axes[2].set_xticks(
      ticks=range(len(xlabels)),
      labels=xlabels,
      rotation='vertical',
      fontsize=small,
  )
  axes[2].set_ylabel('Latent #', fontsize=medium)
  axes[2].set_xlabel('Update Network Inputs', fontsize=medium)
  axes[2].set_title('Update Network Bottlenecks', fontsize=large)
  fig.tight_layout()  # Adjust layout to prevent overlap
  plt.savefig(plot_path)
  plt.close(fig)
  print(f"Bottleneck plot saved to {plot_path}")
  return fig

def plot_update_rules(
    params: hk.Params,
    disrnn_config: disrnn.DisRnnConfig,
    subj_ind: Optional[int] = None,
    axis_lim: float = 2.1,
) -> list[plt.Figure]:
  """Generates visualizations of the update rules of a HkDisentangledRNN."""

  # Work on a copy so we can disable noise for plotting
  disrnn_config = copy.deepcopy(disrnn_config)
  disrnn_config.noiseless_mode = True

  # Handle multisubject vs single-subject
  if True:
    if subj_ind is not None:
      print('subj_ind provided, but not in multisubject mode. Ignoring it')
    subj_ind = None

  # Extract update-bottleneck sigmas for each input type
  if True:
    param_prefix = 'hk_disentangled_rnn'
    subj_embedding_size = 0
    obs_names = disrnn_config.x_names
    update_obs_s_t = np.transpose(
        disrnn.reparameterize_sigma(
            params[param_prefix]['update_net_obs_sigma_params']
        )
    )
    update_latent_s_t = np.transpose(
        disrnn.reparameterize_sigma(
            params[param_prefix]['update_net_latent_sigma_params']
        ))
    update_sigmas = np.concatenate((update_obs_s_t, update_latent_s_t), axis=1)

  # Build a single-step version of the network
  def step(xs, state):
    core = (disrnn.HkDisentangledRNN)(disrnn_config)
    output, new_state = core(jnp.expand_dims(jnp.array(xs), axis=0), state)
    return output, new_state

  _, step_hk = hk.transform(step)
  key = jax.random.PRNGKey(0)
  step_hk = jax.jit(step_hk)

  # Initial state for perturbations
  reference_state = np.zeros_like(
        rnn_utils.get_initial_state(lambda: disrnn.HkDisentangledRNN(disrnn_config))
    )


  # Helper for 1D plots
  def plot_update_1d(params, unit_i, observations, titles):
    state_bins = np.linspace(-axis_lim, axis_lim, 20)
    colormap = mpl.colormaps['viridis'].resampled(3)
    color = colormap.colors[1]

    fig, axes = plt.subplots(
        1, len(observations), figsize=(len(observations) * 4, 5.5), sharey=True
    )
    if len(observations) == 1:
      axes = [axes]
    axes[0].set_ylabel('Δ Activity')

    for obs_i, (obs, title) in enumerate(zip(observations, titles)):
      ax = axes[obs_i]
      delta_states = np.zeros((len(state_bins), 1))
      for si, val in enumerate(state_bins):
        state = reference_state.copy()
        state[0, unit_i] = val
        if subj_ind is not None:
          xs = [subj_ind] + obs
        else:
          xs = obs
        _, next_state = step_hk(params, key, xs, state)
        delta_states[si] = next_state[0, unit_i] - val
      ax.plot((-axis_lim, axis_lim), (0, 0), color='black')
      ax.plot(state_bins, delta_states, color=color)
      ax.set_title(title, fontsize=large)
      ax.set_xlim(-axis_lim, axis_lim)
      ax.set_xlabel(f'Latent {unit_i+1} Activity', fontsize=medium)
      ax.set_aspect('equal')
      ax.tick_params(axis='both', labelsize=small)
    return fig

  # Determine latent bottleneck openness
  latent_sigmas = np.array(
      disrnn.reparameterize_sigma(
          params[param_prefix]['latent_sigma_params']
      )
  )

  # Sort latents by openness
  latent_order = np.argsort(latent_sigmas)
  figs = []

  # Main loop over each latent
  for latent_i in latent_order:
    if latent_sigmas[latent_i] < 0.5:
      # Identify which obs dims are open inputs
      obs_start = subj_embedding_size
      sensitive = [idx - obs_start for idx in np.where(
          update_sigmas[latent_i, obs_start:obs_start + disrnn_config.obs_size] < 0.5)[0]]

      # Build baseline and test observations
      baseline = [0]*disrnn_config.obs_size
      observations, titles = [], []
      # Always plot the all-zero case
      observations.append(baseline)
      titles.append('All Trials')
      # Then one curve per sensitive dim
      for j in sensitive:
        obs_vec = baseline.copy()
        obs_vec[j] = 1
        observations.append(obs_vec)
        titles.append(f'{obs_names[j]}: 1')

      # Choose 1D plotting (no multi-latent interactions here)
      fig = plot_update_1d(params, latent_i, observations, titles)
      figs.append(fig)
      fig.tight_layout()

  return figs


def plot_latent_activations_overlayed(
        seq_list: list,
        xs_train: np.ndarray,
        disrnn_params: dict,
        make_disrnn,
        save_folder_name: str,
        train_test_split_variables: dict,
        df_train: pd.DataFrame
    ):
    """
    For exactly four sessions (seq_list), run DisRNN once per session. Then for each latent:
      • plot the activation over trials
      • overlay all four sessions’ latent‐activation curves on a single axes
      • draw a thick black line at the final trial index (session end)

    All blue/green bars and clue bars have been removed.
    """

    if len(seq_list) != 4:
        raise ValueError("seq_list must contain exactly four session indices.")

    # 1) Extract shapes
    n_trials, n_sessions, _ = xs_train.shape

    # 2) Forward‐pass each of the 4 requested sessions to collect latent activations
    activations_by_session = []
    for sess_idx in seq_list:
        # shape (n_trials, 1, obs_dim)
        session_xs = xs_train[:, sess_idx, :][:, None, :]
        _, network_states = rnn_utils.eval_network(make_disrnn, disrnn_params, session_xs)
        # network_states has shape (n_trials, 1, latent_dim)
        activations_by_session.append(np.array(network_states))

    # 3) Compute and sort latent σ’s
    latent_sigmas = disrnn.reparameterize_sigma(
        disrnn_params['hk_disentangled_rnn']['latent_sigma_params'])
    latent_sigma_order = np.argsort(latent_sigmas)
    sorted_sigmas = latent_sigmas[latent_sigma_order]

    # 4) Reorder each session’s activations by that same order
    activations_sorted = [
        arr[:, :, latent_sigma_order] for arr in activations_by_session
    ]
    latent_dim = activations_sorted[0].shape[2]

    os.makedirs(save_folder_name, exist_ok=True)
    session_colors = ['C0', 'C1', 'C2', 'C3']  # distinct colors for the 4 sessions
    session_labels = [f"Session {s}" for s in seq_list]

    # 5) Loop over each latent dimension
    for i in range(latent_dim):
        fig, ax = plt.subplots(figsize=(20, 6))

        # 5.1) Plot all four sessions’ latent‐activation curves on one axes
        for idx, sess_idx in enumerate(seq_list):
            series_i = activations_sorted[idx][:, 0, i]  # shape = (n_trials,)
            ax.plot(
                series_i,
                color=session_colors[idx],
                linewidth=1.5,
                label=session_labels[idx]
            )

        # 5.2) Draw a thick black line at the final trial index (same for all sessions)
        end_idx = n_trials - 1
        ax.axvline(
            x=end_idx,
            color="black",
            linestyle="-",
            linewidth=2.0,
            alpha=0.8
        )
        ax.text(
            end_idx,
            ax.get_ylim()[1],
            "Session End",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=9,
            color="black"
        )

        # 5.3) Final formatting
        ax.set_title(f"Latent {i+1} (σ = {sorted_sigmas[i]:.4f}) — Overlaid Sessions", fontsize=14)
        ax.set_ylabel(f"Activation of Latent {i+1}")
        ax.set_xlabel("Trial Index")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize="small")

        plt.tight_layout()
        filename = f"latent_{i+1}_four_sessions_overlay_one_plot.png"
        plot_path = os.path.join(save_folder_name, filename)
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved overlaid latent {i+1} plot to {plot_path}")



def plot_comparison_heatmap(df_human: pd.DataFrame,
                            df_model: pd.DataFrame,
                            save_folder: str,
                            max_yticks: int = 10):
    """
    Plot side-by-side heatmaps of human accuracy and model estimation by trial (rows) and block (cols).
    - df_human must have columns ['blockN','trial_in_block','human_correct']
    - df_model must have columns ['blockN','trial_in_block','model_est_optimal_prob']
    """
    # Filter out -1 rows
    df_h = df_human[(df_human['trial_in_block'] >= 1) & (df_human['blockN'] >= 1)]
    df_m = df_model[(df_model['trial_in_block'] >= 1) & (df_model['blockN'] >= 1)]

    # Pivot to matrices
    human_mat = (
        df_h.groupby(['trial_in_block','blockN'])['human_correct']
            .mean()
            .unstack()
            .sort_index()
    )
    model_mat = (
        df_m.groupby(['trial_in_block','blockN'])['model_est_optimal_prob']
            .mean()
            .unstack()
            .sort_index()
    )

    # Percentile‐based vmin/vmax separately
    def clip_range(mat):
        arr = mat.values.flatten()
        arr = arr[~np.isnan(arr)]
        return np.percentile(arr, 1), np.percentile(arr, 99)
    hmin, hmax = clip_range(human_mat)
    mmin, mmax = clip_range(model_mat)

    # Create figure + axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot each heatmap
    for ax, mat, (vmin, vmax), title in zip(
        axes,
        [human_mat, model_mat],
        [(hmin, hmax), (mmin, mmax)],
        ['Human Estimated Optimal Prob', 'Model Estimated Optimal Prob']
    ):
        im = ax.imshow(
            mat.values,
            origin='lower',
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
            cmap='viridis'
        )
        ax.set_title(title)
        ax.set_xlabel('Block Number')

        # X‐ticks
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(mat.columns.astype(int), rotation=45)

        # Y‐ticks (downsample if too many)
        n = mat.shape[0]
        yt = np.linspace(0, n-1, min(max_yticks, n)).astype(int)
        ax.set_yticks(yt)
        ax.set_yticklabels(mat.index[yt].astype(int))
        if ax is axes[0]:
            ax.set_ylabel('Trial in Block')

    # Reserve right margin for colorbar
    plt.tight_layout(rect=[0, 0, 0.90, 1])

    # Shared colorbar without overlap
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        fraction=0.046, pad=0.02)
    cbar.set_label('Proportion')

    # Save
    os.makedirs(save_folder, exist_ok=True)
    out = os.path.join(save_folder, 'human_vs_model_heatmap.png')
    plt.savefig(out)
    plt.close(fig)
    print(f"Saved comparison heatmap to {out}")

def plot_first_encounter_heatmap(df_human: pd.DataFrame,
                                 save_folder: str,
                                 max_yticks: int = 10):
    """
    Plot a heatmap of human accuracy on *first-encounter* trials (i.e. where
    stimulusSlotID_encounterNInBlock == 1), by trial (rows) and block (cols).

    - df_human must have columns ['blockN','trial_in_block',
      'human_correct','stimulusSlotID_encounterNInBlock']
    """
    # 1) filter to only first-encounter trials and remove any -1s
    fe = df_human[(df_human['stimulusSlotID_encounterNInBlock'] == 1)
                  & (df_human['trial_in_block'] >= 1)
                  & (df_human['blockN'] >= 1)]

    # 2) pivot to matrix: rows=trial_in_block, cols=blockN
    mat = (
        fe
        .groupby(['trial_in_block', 'blockN'])['human_correct']
        .mean()
        .unstack()
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    # 3) robust vmin/vmax by 1st/99th percentiles
    flat = mat.values.flatten()
    flat = flat[~np.isnan(flat)]
    vmin, vmax = np.percentile(flat, [1, 99])

    # 4) plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        mat.values,
        origin='lower',
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
        cmap='viridis'
    )

    # 5) labels
    ax.set_title('First-Encounter Human Estimated Optimal Prob\n(rows=trial, cols=block)')
    ax.set_xlabel('Block Number')
    ax.set_ylabel('Trial in Block')

    # xticks
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.astype(int), rotation=45)
    # yticks downsampled
    n = mat.shape[0]
    yt = np.linspace(0, n-1, min(max_yticks, n)).astype(int)
    ax.set_yticks(yt)
    ax.set_yticklabels(mat.index[yt].astype(int))

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy (proportion correct)')

    # save
    os.makedirs(save_folder, exist_ok=True)
    out = os.path.join(save_folder, 'first_encounter_heatmap.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"Saved first-encounter heatmap to {out}")


# ------------------------------------------------------------------
#  Seaborn colour‑blind safe palette (Wong, 2011) applied globally
# ------------------------------------------------------------------
def plot_latent_activations(
        seq_list: list,
        xs_train: np.ndarray,
        disrnn_params: dict,
        make_disrnn,
        save_folder_name: str,
        train_test_split_variables: dict,
        df_train: pd.DataFrame
    ):
    """
    For exactly four sequences (seq_list), run DisRNN once per sequence. Then for each latent:
      • plot the activation over trials
      • draw block-boundary lines
      • draw a thick black line at sequence end
      • draw ORANGE bars   (bottom quarter) whenever the model predicted action = 1
      • draw SKY-BLUE bars (next quarter)   whenever the current model choice was correct
      • draw GREEN bars    (third quarter)  whenever the previous HUMAN choice was correct
      • draw NAVY bars     (top quarter)    whenever the previous HUMAN choice == 1
      • draw a thin 4-colour “slot bar”    beneath the latent curve indicating slot
      • draw a thin 8-colour “clue bar”    beneath the latent curve indicating clue
      • draw a thin 2-colour “context bar” beneath the latent curve indicating context
    """
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="colorblind")  # use Seaborn colour‑blind template for all plots

    CB_PALETTE = sns.color_palette("colorblind", 10)
    CB_ORANGE  = CB_PALETTE[1]   # "#de8f05"
    CB_SKYBLUE = CB_PALETTE[9]   # "#56b4e9"
    CB_GREEN   = CB_PALETTE[2]   # "#029e73"
    CB_BLUE    = CB_PALETTE[0]   # "#0173b2"

    if len(seq_list) != 4:
        raise ValueError("seq_list must contain exactly four sequence indices.")

    # ─── extract data arrays ──────────────────────────────────────────────
    n_trials, n_sequences, _ = xs_train.shape
    prev_choice  = train_test_split_variables['bitResponseA_prev_train'][..., 0]
    prev_correct = train_test_split_variables['bitCorr_prev_train'][..., 0]
    slot_ids     = train_test_split_variables['stimulusSlotID_train'][..., 0] - 1
    clue_ids     = train_test_split_variables['state_train'][..., 0]
    context_ids  = train_test_split_variables['context_train'][..., 0]

    # ─── model predictions ────────────────────────────────────────────────
    model_logits, _ = rnn_utils.eval_network(make_disrnn, disrnn_params, xs_train)
    model_logits   = np.array(model_logits)
    pred_probs     = jax.nn.softmax(model_logits[..., :2], axis=-1)
    pred_choices   = np.argmax(np.array(pred_probs), axis=-1)
    opt_choices    = train_test_split_variables['bitResponseAIsCorr_train'][..., 0]
    curr_model_correct = (pred_choices == opt_choices)
    model_choose_one   = (pred_choices == 1)  # ORANGE bars

    # ─── forward-pass to collect activations ──────────────────────────────
    activations_by_sequence = []
    for seq_idx in seq_list:
        seq_xs = xs_train[:, seq_idx, :][:, None, :]
        _, states = rnn_utils.eval_network(make_disrnn, disrnn_params, seq_xs)
        activations_by_sequence.append(np.array(states))

    # ─── sort latents by σ (small → open) ─────────────────────────────────
    latent_sigmas = disrnn.reparameterize_sigma(
        disrnn_params['hk_disentangled_rnn']['latent_sigma_params'])
    order         = np.argsort(latent_sigmas)
    sorted_sigmas = latent_sigmas[order]
    activations_sorted = [a[:, :, order] for a in activations_by_sequence]
    latent_dim         = activations_sorted[0].shape[2]

    # ─── colour maps for slot / clue / context now use colour‑blind palette ─────
    slot_cmap     = mcolors.ListedColormap(sns.color_palette("colorblind", 4))
    slot_norm     = mcolors.BoundaryNorm(boundaries=np.arange(5), ncolors=4)
    clue_cmap     = mcolors.ListedColormap(sns.color_palette("colorblind", 8))
    clue_norm     = mcolors.BoundaryNorm(boundaries=np.arange(9), ncolors=8)
    context_cmap  = mcolors.ListedColormap(sns.color_palette("colorblind", 2))
    context_norm  = mcolors.BoundaryNorm(boundaries=np.arange(3), ncolors=2)

    os.makedirs(save_folder_name, exist_ok=True)
    seq_colors = ['C0', 'C1', 'C2', 'C3']   # now map to colour‑blind palette via seaborn

    for i in range(latent_dim):
        fig = plt.figure(figsize=(25, 16))
        outer = fig.add_gridspec(nrows=4, ncols=1, hspace=0.4)

        # ── legend patches using CB palette ───────────────────────────────
        orange_patch  = Patch(facecolor=CB_ORANGE,  alpha=0.4, label="model_prediction = 1")
        sky_patch     = Patch(facecolor=CB_SKYBLUE, alpha=0.4, label="curr_model_correct = 1")
        green_patch   = Patch(facecolor=CB_GREEN,   alpha=0.4, label="prev_correct = 1")
        blue_patch    = Patch(facecolor=CB_BLUE,    alpha=0.4, label="prev_choice = 1")
        slot_patches  = [Patch(facecolor=slot_cmap(j),    label=f"Slot {j+1}") for j in range(4)]
        clue_patches  = [Patch(facecolor=clue_cmap(j),    label=f"Clue {j+1}") for j in range(8)]
        context_patches = [Patch(facecolor=context_cmap(j), label=f"Context {j+1}") for j in range(2)]
        bar_handles = [orange_patch, sky_patch, green_patch, blue_patch] + slot_patches + clue_patches + context_patches

        for row_idx, seq_idx in enumerate(seq_list):
            inner = outer[row_idx].subgridspec(nrows=4, ncols=1,
                                               height_ratios=[5, 2, 2, 2],
                                               hspace=0.1)

            # ── activation curve ──────────────────────────────────────────
            ax_act = fig.add_subplot(inner[0])
            vals   = activations_sorted[row_idx][:, 0, i]
            y_max, y_min = vals.max(), vals.min()
            Δ = 0.05*(y_max - y_min) if y_max != y_min else 0.1
            ax_act.set_ylim((y_min - 4*Δ, y_max + Δ))

            lh, = ax_act.plot(vals, color=seq_colors[row_idx], lw=1.5,
                              label=f"Sequence {seq_idx}", zorder=4)

            # ── block boundaries ─────────────────────────────────────────
            blocks  = train_test_split_variables['blockN_train'][:, seq_idx, 0]
            changes = np.where(np.diff(blocks) != 0)[0] + 1
            for c in changes:
                ax_act.axvline(c, color=seq_colors[row_idx], ls="--", lw=1,
                               alpha=0.6, zorder=3)
                ax_act.text(c, y_max + 0.5*Δ, f"Block {int(blocks[c])}",
                            rotation=90, va="bottom", ha="center",
                            fontsize=8, color=seq_colors[row_idx], alpha=0.7,
                            zorder=3)

            # sequence end
            end = n_trials - 1
            ax_act.axvline(end, color="black", lw=2, alpha=0.8, zorder=3)
            ax_act.text(end, y_max + 0.8*Δ, "Sequence End", rotation=90,
                        va="top", ha="right", fontsize=9, color="black",
                        zorder=3)

            # ── quarter-height event bars (CB palette) ───────────────────
            for t in np.where(model_choose_one[:, seq_idx])[0]:
                ax_act.axvspan(t, t+1, ymin=0.00, ymax=0.25,
                               facecolor=CB_ORANGE, alpha=0.4, zorder=2)
            for t in np.where(curr_model_correct[:, seq_idx])[0]:
                ax_act.axvspan(t, t+1, ymin=0.25, ymax=0.50,
                               facecolor=CB_SKYBLUE, alpha=0.4, zorder=2)
            for t in np.where(prev_correct[:, seq_idx] == 1)[0]:
                ax_act.axvspan(t, t+1, ymin=0.50, ymax=0.75,
                               facecolor=CB_GREEN, alpha=0.4, zorder=2)
            for t in np.where(prev_choice[:, seq_idx] == 1)[0]:
                ax_act.axvspan(t, t+1, ymin=0.75, ymax=1.00,
                               facecolor=CB_BLUE, alpha=0.4, zorder=2)

            ax_act.set_ylabel(f"Latent {i+1}")
            ax_act.set_title(f"Sequence {seq_idx}")
            ax_act.grid(alpha=0.2)

            # ── slot / clue / context bars (now colour‑blind palette) ────
            ax_slot = fig.add_subplot(inner[1], sharex=ax_act)
            ax_slot.imshow(slot_ids[:, seq_idx].reshape(1, n_trials), aspect="auto",
                           cmap=slot_cmap, norm=slot_norm, origin="lower",
                           extent=(0, n_trials, 0, 1), interpolation="nearest",
                           zorder=1)
            ax_slot.set_yticks([])
            ax_slot.set_ylabel("Slot", labelpad=10)

            ax_clue = fig.add_subplot(inner[2], sharex=ax_act)
            ax_clue.imshow(clue_ids[:, seq_idx].reshape(1, n_trials), aspect="auto",
                           cmap=clue_cmap, norm=clue_norm, origin="lower",
                           extent=(0, n_trials, 0, 1), interpolation="nearest",
                           zorder=1)
            ax_clue.set_yticks([])
            ax_clue.set_ylabel("Clue", labelpad=10)

            ax_ctx = fig.add_subplot(inner[3], sharex=ax_act)
            ax_ctx.imshow(context_ids[:, seq_idx].reshape(1, n_trials), aspect="auto",
                           cmap=context_cmap, norm=context_norm, origin="lower",
                           extent=(0, n_trials, 0, 1), interpolation="nearest",
                           zorder=1)
            ax_ctx.set_yticks([])
            ax_ctx.set_ylabel("Context", labelpad=10)

            # reduce x-tick density
            step = max(n_trials // 10, 1)
            ax_ctx.set_xticks(np.arange(0, n_trials, step))
            ax_ctx.set_xlabel("Trial Index")

            if row_idx == 0:
                ax_act.legend(handles=[lh], loc="upper right", fontsize="small")

        # ── global legend for bars ────────────────────────────────────────
        fig.legend(handles=bar_handles,
                   labels=[h.get_label() for h in bar_handles],
                   loc="lower center", ncol=12,
                   bbox_to_anchor=(0.5, 0.02))

        # ── title, layout, save ──────────────────────────────────────────
        fig.suptitle(f"Latent {i+1} (σ = {sorted_sigmas[i]:.4f}) — Four Sequences",
                     fontsize=14)
        plt.tight_layout(rect=[0, 0.06, 1, 0.96])

        fname = f"latent_{i+1}_four_sequences_separate_bars.png"
        path  = os.path.join(save_folder_name, fname)
        plt.savefig(path)
        plt.close(fig)
        print(f"Saved: {path}")


def main(seed, saved_checkpoint_pth):
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Dataset setup
    dataset_type = 'RealWorldKimmelfMRIDataset'
    dataset_path = "/home/rsw0/Desktop/yolanda/CogModRNN/CogModRNN/dataset/tensor_for_dRNN_desc-syn_nSubs-2000_nSessions-4_nBlocks-7_nTrialsPerBlock-25_b-multiple_20250613.mat"
    dataset_train, dataset_test, train_test_split_variables = preprocess_data(dataset_type, dataset_path, 0.1)

    # Save folder setup
    save_folder_name = os.path.join('plots', saved_checkpoint_pth.split('checkpoints_new_data/')[1].split('.pkl')[0])
    os.makedirs(save_folder_name, exist_ok=True)

    # Load checkpoint data
    with open(saved_checkpoint_pth, 'rb') as file:
        checkpoint = pickle.load(file)

    args_dict = checkpoint['args_dict']
    print(args_dict)
    disrnn_params = checkpoint['params']
    print(f'Loaded disrnn_params from {saved_checkpoint_pth}')

    # ----------------------
    # Model Initialization
    # ----------------------
    cfg = build_disrnn_config(
            obs_size = 18,
            output_size = 2,
            latent_size = args_dict['latent_size'],
            update_mlp_shape = args_dict['update_mlp_shape'],
            choice_mlp_shape = args_dict['choice_mlp_shape'],
            noiseless = True,   # deterministic
            latent_penalty            = 0.0,
            update_net_obs_penalty    = 0.0,
            update_net_latent_penalty = 0.0,
            choice_net_latent_penalty = 0.0,
            l2_scale = 0.0,      # ← no weight-decay in warm-up
        )
    def make_disrnn():
        return disrnn.HkDisentangledRNN(cfg)
   
    # Training and evaluation
    xs_train, ys_train = next(dataset_train)

    print('Normalized Likelihoods and Accuracies for disRNN')
    print('Training Dataset')
    train_norm_likelihood, _ = compute_log_likelihood_and_accuracy(xs_train, ys_train, make_disrnn, disrnn_params)
    pred_train, ys_train, xs_train = evaluate(xs_train, ys_train, make_disrnn, disrnn_params)

    # Create DataFrame for plotting
    df_train = create_dataframe_for_plotting(
        pred_train, ys_train, xs_train, train_test_split_variables=train_test_split_variables
    )
    log_probs_train = np.log(df_train['human_chosen_action_prob'])
    normalized_likelihood_upper_bound_train = np.exp(np.mean(log_probs_train))
    print("Model Normalized likelihood Upperbound for training", normalized_likelihood_upper_bound_train)
    


    # Ensure these columns exist for plotting
    df_train['human_optimal'] = (df_train['human_chosen_action'] == df_train['correct_optimal_choice']).astype(int)
    df_train['model_optimal'] = (df_train['model_est_optimal_prob'] >= 0.5).astype(int)

    # First encounter trials analysis
    first_mask = df_train['stimulusSlotID_encounterNInBlock'] == 1
    human_first_cont = df_train.loc[first_mask, 'rwd_manual'].mean()
    model_first_cont = df_train.loc[first_mask, 'model_est_optimal_prob'].mean()
    human_repeat_cont = df_train.loc[~first_mask, 'rwd_manual'].mean()
    model_repeat_cont = df_train.loc[~first_mask, 'model_est_optimal_prob'].mean()

    print(f"First‐encounter trials:   Human  = {100 * human_first_cont:.2f}%   Model  = {100 * model_first_cont:.2f}%")
    print(f"Repeat‐encounter trials:  Human  = {100 * human_repeat_cont:.2f}%   Model  = {100 * model_repeat_cont:.2f}%")

    # ----------------------
    # Plotting and Saving
    # ----------------------
    plot_bottlenecks(disrnn_params, cfg, save_folder_name)
    plot_comparison_heatmap(
        df_human=df_train[['blockN', 'trialNInBlock', 'human_optimal']].rename(columns={'trialNInBlock': 'trial_in_block', 'human_optimal': 'human_correct'}),
        df_model=df_train[['blockN', 'trialNInBlock', 'model_est_optimal_prob']].rename(columns={'trialNInBlock': 'trial_in_block', 'model_est_optimal_prob': 'model_est_optimal_prob'}),
        save_folder=save_folder_name
    )

    # Inference performance plot
    plot_inference_performance(df_train, save_folder_name)

    figs = plot_update_rules(
        params=disrnn_params,
        disrnn_config=cfg,
        axis_lim=2.1
    )

    # If multisubject and you want to plot for subject #2:
    # figs = plot_update_rules(params, config, subj_ind=2, axis_lim=2.1)

    # 4) Save or display all the figures
    for i, fig in enumerate(figs):
        fig.savefig(f"update_rule_latent_{i+1}.png")
        plt.close(fig)

    # Latent activations plot
    plot_latent_activations(
        seq_list=[1, 2, 10, 15],
        xs_train=xs_train,
        disrnn_params=disrnn_params,
        make_disrnn=make_disrnn,
        save_folder_name=save_folder_name,
        train_test_split_variables=train_test_split_variables,
        df_train=df_train
    )

    plot_latent_activations_overlayed(
        seq_list=[1, 2, 10, 15],
        xs_train=xs_train,
        disrnn_params=disrnn_params,
        make_disrnn=make_disrnn,
        save_folder_name=save_folder_name,
        train_test_split_variables=train_test_split_variables,
        df_train=df_train
    )

    '''
    csv_outpath = os.path.join(save_folder_name, "df_train.csv")
    df_train.to_csv(csv_outpath, index=False)
    print(f"df_train has been saved to CSV at: {csv_outpath}")
    '''


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", nargs='?', type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--saved_checkpoint_pth", type=str, required=True, help="Saved checkpoint path for evaluation.")
    args = parser.parse_args()

    # Run main function
    main(args.seed, args.saved_checkpoint_pth)
