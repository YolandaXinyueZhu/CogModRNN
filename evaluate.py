import os
import pickle
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jax import nn
from CogModelingRNNsTutorial.CogModelingRNNsTutorial import bandits, disrnn, hybrnn, plotting, rat_data, rnn_utils
from de_rnn import load_data_for_one_human, to_onehot, zs_to_onehot, dataset_size

import haiku as hk
import jax
import jax.numpy as jnp
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

# Add the parent directory to the Python path
import sys

# Import the necessary modules from the CogModelingRNNsTutorial package
from CogModelingRNNsTutorial.CogModelingRNNsTutorial import bandits, disrnn, hybrnn, plotting, rat_data, rnn_utils

# Suppress warnings
warnings.filterwarnings("ignore")

def plot_session_dynamic(timeseries: np.ndarray,
                 timeseries_name: str,
                 labels: Optional[List[str]] = None,
                 fig_ax: Optional = None):
    """Plot data from a single behavioral session of the bandit task."""

    y_high = np.max(timeseries) + 0.1
    y_low = np.min(timeseries) - 0.1

    # Make the plot
    if fig_ax is None:
        fig, ax = plt.subplots(1, figsize=(20, 10))
    else:
        fig, ax = fig_ax
    if labels is not None:
        if timeseries.ndim == 1:
            timeseries = timeseries[:, None]
        if len(labels) != timeseries.shape[1]:
            raise ValueError('labels length must match timeseries.shape[1].')
        for i in range(timeseries.shape[1]):
            ax.plot(timeseries[:, i], label=labels[i])
        ax.legend(bbox_to_anchor=(1, 1))
    else:  # Skip legend.
        ax.plot(timeseries)

    ax.set_xlabel('Trial')
    ax.set_ylabel(timeseries_name)
    plt.savefig("plot_session_dynamic.png")

    return fig

# Load the dataset
dataset_type = 'RealWorldKimmelfMRIDataset'  # or other types as needed
is_synthetic = dataset_type.startswith('Synthetic')

if is_synthetic:
    # Handle synthetic data cases (if needed)
    pass
else:
    if dataset_type in ['RealWorldRatDataset', 'RealWorldKimmelfMRIDataset']:
        LOCAL_PATH_TO_FILE = "tensor_for_dRNN_desc-syn_nSubs-2000_nSessions-1_nBlocks-1_nTrialsPerBlock-100_b-0.3_NaN_30_0.93_0.45_NaN_NaN_withOptimalChoice_20240718_fast.mat"
        if not os.path.exists(LOCAL_PATH_TO_FILE):
            raise ValueError('File not found.')
        FNAME_ = LOCAL_PATH_TO_FILE

        gen_alpha = "unknown"
        gen_beta = "unknown"
        sigma = 0.1
        testing_set_proportion = 0.1
        environment = bandits.EnvironmentBanditsDrift(sigma=sigma)

        if dataset_type == 'RealWorldKimmelfMRIDataset':
            xs, ys, zs, _, _, bitResponseAIsCorr, P_A = load_data_for_one_human(FNAME_)
            zs_oh = zs_to_onehot(zs)
            inputs = np.concatenate([xs, zs_oh], axis=-1)

            dataset_train, dataset_test, bitResponseAIsCorr_train, bitResponseAIsCorr_test, P_A_train, P_A_test = format_into_datasets_flexible(
                inputs, ys.copy(), bitResponseAIsCorr, P_A, rnn_utils.DatasetRNN, testing_set_proportion
            )

            # Print the size of the dataset
            train_size, test_size = dataset_size(dataset_train._xs, dataset_test._xs)
            print(f'Training dataset size: {train_size} samples')
            print(f'Testing dataset size: {test_size} samples')


# Print the number of training and testing sessions
print('Training: {} sessions\nTesting: {} sessions'.format(dataset_train._xs.shape[1], dataset_test._xs.shape[1]))

# Example function usage: Print shape of a batch of data
x, y = next(dataset_train)
print(f'x.shape = {x.shape} \ny.shape = {y.shape}')

# Define the directories for saving files
checkpoint_dir = './checkpoints'
plot_dir = './plots'
loss_dir = './losses'

# Load the checkpoint parameters
checkpoint_name = 'checkpoints/ls_32_umlp_(8, 8)_cmlp_(8, 8)_beta_1e-05_penalty_1e-10_20240822_112813/disrnn_params_ls_32_umlp_(8, 8)_cmlp_(8, 8)_penalty_1e-10_beta_1e-05_lr_1e-3.pkl'
with open(checkpoint_name, 'rb') as file:
    disrnn_params = pickle.load(file)

print(f'Loaded disrnn_params from {checkpoint_name}')

def compute_log_likelihood_and_accuracy(dataset, model_fun, params):
    xs, actual_choices = next(dataset)
    n_trials_per_session, n_sessions = actual_choices.shape[:2]
    model_outputs, model_states = rnn_utils.eval_model(model_fun, params, xs)
    predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))

    log_likelihood = 0
    n = 0  # Total number of trials across sessions.
    correct_predictions = 0

    for sess_i in range(n_sessions):
        for trial_i in range(n_trials_per_session):
            actual_choice = int(actual_choices[trial_i, sess_i])
            if (actual_choice >= 0) and (0 <= actual_choice < predicted_log_choice_probabilities.shape[2]):
                # TODO: do it on ground truth log choice propbabilties
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



def make_disrnn():
        model = disrnn.HkDisRNN(
            obs_size=x.shape[2],
            target_size=2,
            latent_size=32,
            update_mlp_shape=(8,8),
            choice_mlp_shape=(8,8),
            eval_mode=0.0,
            beta_scale=1e-05,
            activation=jax.nn.relu
        )
        return model


# Compute quality-of-fit: Held-out Normalized Likelihood
print('Normalized Likelihoods and Accuracies for disRNN')
print('Training Dataset')
training_likelihood, training_accuracy = compute_log_likelihood_and_accuracy(dataset_train, make_disrnn, disrnn_params)
print('Held-Out Dataset')
testing_likelihood, testing_accuracy = compute_log_likelihood_and_accuracy(dataset_test, make_disrnn, disrnn_params)
dataset_test3 = copy.deepcopy(dataset_test)
dataset_test3._ys = np.random.permutation(dataset_test3._ys)
print('Held-Out Shuffled Dataset')
testing_likelihood_shuffled, testing_accuracy_shuffled = compute_log_likelihood_and_accuracy(dataset_test3, make_disrnn, disrnn_params)


def evaluate(dataset, model_fun, params):
    xs, actual_choices = next(dataset)
    n_trials_per_session, n_sessions = actual_choices.shape[:2]
    model_outputs, model_states = rnn_utils.eval_model(model_fun, params, xs)
    predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))
    return predicted_log_choice_probabilities, actual_choices, xs


# Plot example session: latents + choices
'''
xs, ys = next(dataset_test)
sess_i = 0
trial_end = rnn_utils.find_session_end(xs[:, sess_i, 0])
network_outputs, network_states = rnn_utils.eval_model(
    make_disrnn, disrnn_params, xs[:trial_end, sess_i:sess_i+1])
network_states = np.array(network_states)
choices = xs[:trial_end, sess_i, 0]
rewards = xs[:trial_end, sess_i, 1]
disrnn_activations = network_states[:trial_end, sess_i, :]
bandits.plot_session(choices=choices,
                     rewards=rewards,
                     timeseries=disrnn_activations,
                     timeseries_name='Network Activations')
'''
# Example usage:
xs, ys = next(dataset_test)
sess_i = 0
trial_end = rnn_utils.find_session_end(xs[:, sess_i, 0])
network_outputs, network_states = rnn_utils.eval_model(
    make_disrnn, disrnn_params, xs[:trial_end, sess_i:sess_i+1])
network_states = np.array(network_states)
disrnn_activations = network_states[:trial_end, sess_i, :]

labels = [f'Latent_{i+1}' for i in range(disrnn_activations.shape[1])]
plot_session_dynamic(timeseries=disrnn_activations,
             timeseries_name='Network Activations',
             labels=labels)

# Data loaded from file
pred_train, act_train, xs_train = evaluate(dataset_train, make_disrnn, disrnn_params)
pred_test, act_test, xs_test = evaluate(dataset_test, make_disrnn, disrnn_params)

# Gather data into dataframes for plotting
_c0 = act_train == 0
_c1 = act_train == 1
_cn = ~np.isin(act_train, [0, 1])
_c = np.concatenate([_c0, _c1, _cn], axis=2)
_nan = np.empty([*pred_train.shape[:2], 1])
_nan.fill(np.nan)
_pred_train_expand = np.concatenate([pred_train, _nan], axis=2)
_pred_train_acc = _pred_train_expand[_c]
_pred_train_acc = _pred_train_acc.reshape(act_train.shape)

# Create the DataFrame
'''
df_train = pd.DataFrame({
    'act': act_train[:, :, 0].ravel(),
    'pred': np.exp(pred_train[:, :, 1].ravel()),
    'pred_acc': np.exp(_pred_train_acc.ravel()),
    'rwd': np.vstack((np.zeros((1, xs_train.shape[1])), xs_train[1:, :, 0])).ravel()
})
'''

df_train = pd.DataFrame({
    'act': act_train[:, :, 0].ravel(), # human-chosen action
    'pred': np.exp(pred_train[:, :, 1].ravel()), # model-predicted action
    'pred_acc': np.exp(_pred_train_acc.ravel()), # model-predicted probability of the human-chosen action
    'bitResponseAIsCorr': bitResponseAIsCorr_train[:, :, 0].ravel(), # The correct optimal choices of the experiment
    'P_A': P_A_train[:, :, 0].ravel(), # human probability of choosing action 0, human synthetic data is generated from P_A
    'rwd': np.vstack((np.zeros((1, xs_train.shape[1])), xs_train[1:, :, 0])).ravel() # reward for human-chosen action. >0 means human choses optimal choices
})


# Compute the optimal choice
df_train['optimal_choice_manual'] = np.where(df_train['rwd'] == 1, df_train['act'], 1 - df_train['act'])

# Add the model-estimated probability of the optimal action
df_train['model_est_optimal_prob'] = np.where(df_train['bitResponseAIsCorr'] == 1, df_train['pred'],  1 - df_train['pred'])
df_train['human_chosen_action_prob'] = np.where(df_train['act'] == 1, df_train['P_A'], 1 - df_train['P_A'])
log_probs = np.log(df_train['human_chosen_action_prob'])
normalized_likelihood_upper_bound = np.exp(np.mean(log_probs))
print("Model Normalized likelihood Upperbound", normalized_likelihood_upper_bound)


print(df_train.tail(1000))
# Plotting the average accuracy over trials
#for dt, df, ds in [('train', df_train, dataset_train), ('test', df_test, dataset_test)]:
# Plotting the average accuracy over trials and individual cases
for dt, df, ds in [('train', df_train, dataset_train)]:
    _trial_num = np.tile(np.arange(ds._xs.shape[0]), [ds._xs.shape[1], 1]).transpose()

    # Calculate normalized likelihood and upper bound for each trial
    df['log_likelihood'] = np.log(df['pred_acc'])
    df['log_upper_bound'] = np.where(df['act'] == 1, np.log(df['P_A']), np.log(1 - df['P_A']))

    normalized_likelihood_per_trial = np.exp(df['log_likelihood'])
    upper_bound_normalized = np.exp(df['log_upper_bound'])

    # Plotting the normalized likelihood vs upper bound
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df, x=_trial_num.ravel(), y=normalized_likelihood_per_trial, label='Normalized Log-Likelihood')
    sns.lineplot(data=df, x=_trial_num.ravel(), y=upper_bound_normalized, label='Upper Bound', linestyle='--')
    plt.legend()
    plt.xlabel('Trial Number')
    plt.ylabel('Normalized Likelihood')
    plt.xlim(left=1)
    plt.ylim(bottom=0, top=1.1)
    plt.title(f'{dt.capitalize()} Data: Normalized Log-Likelihood vs Upper Bound per Trial')
    plt.savefig(os.path.join(plot_dir, f'normalized_log_likelihood_vs_upper_bound_{dt}.png'))
    plt.close()


    # Plotting the probability of optimal choices over trials
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df, x=_trial_num.ravel(), y='rwd', label='Human Proportion of Optimal Choices (Blue Line)')
    sns.lineplot(data=df, x=_trial_num.ravel(), y='model_est_optimal_prob', label='Model-Estimated Probability of Optimal Action (Orange Line)')
    plt.legend()
    plt.xlabel('Trial Number')
    plt.ylabel('Probability')
    plt.xlim(left=1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title(f'{dt.capitalize()} Data: Probability of Optimal Choices over Trials')
    plt.savefig(os.path.join(plot_dir, f'probability_optimal_choices_over_trials_{dt}.png'))
    plt.close()

    # Plotting the probability of chosen actions over trials
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df, x=_trial_num.ravel(), y='human_chosen_action_prob', label='Human Chosen Action Probability (Green Line)', color='green')
    sns.lineplot(data=df, x=_trial_num.ravel(), y='pred_acc', label='Model Chosen Action Probability (Red Line)', color='red')
    plt.legend()
    plt.xlabel('Trial Number')
    plt.ylabel('Probability')
    plt.xlim(left=1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title(f'{dt.capitalize()} Data: Probability of Chosen Actions over Trials')
    plt.savefig(os.path.join(plot_dir, f'probability_chosen_actions_over_trials_{dt}.png'))
    plt.close()

    # Plotting individual cases for optimal choices
    plt.figure(figsize=(16, 8))
    for i in range(min(1, ds._xs.shape[1])):  # Plotting up to 5 individual cases
        actual_data = df.iloc[i::ds._xs.shape[1]][['rwd']]
        optimal_prob_data = df.iloc[i::ds._xs.shape[1]][['model_est_optimal_prob']]
        
        sns.lineplot(data=actual_data, x=_trial_num[:, i], y='rwd', label=f'Actual Optimal Choice {i}')
        sns.lineplot(data=optimal_prob_data, x=_trial_num[:, i], y='model_est_optimal_prob', label=f'Model-Estimated Optimal Probability {i}')
    plt.legend()
    plt.xlabel('Trial Number')
    plt.ylabel('Probability')
    plt.xlim(left=1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title(f'{dt.capitalize()} Data: Individual Cases Probability of Optimal Choices over Trials')
    plt.savefig(os.path.join(plot_dir, f'individual_cases_probability_optimal_choices_over_trials_{dt}.png'))
    plt.close()

    # Plotting individual cases for chosen actions
    plt.figure(figsize=(16, 8))
    for i in range(min(1, ds._xs.shape[1])):  # Plotting up to 5 individual cases
        human_action_prob_data = df.iloc[i::ds._xs.shape[1]][['human_chosen_action_prob']]
        model_action_prob_data = df.iloc[i::ds._xs.shape[1]][['pred_acc']]
        
        sns.lineplot(data=human_action_prob_data, x=_trial_num[:, i], y='human_chosen_action_prob', label=f'Human Chosen Action Probability {i}')
        sns.lineplot(data=model_action_prob_data, x=_trial_num[:, i], y='pred_acc', label=f'Model Chosen Action Probability {i}')
    plt.legend()
    plt.xlabel('Trial Number')
    plt.ylabel('Probability')
    plt.xlim(left=1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.title(f'{dt.capitalize()} Data: Individual Cases Probability of Chosen Actions over Trials')
    plt.savefig(os.path.join(plot_dir, f'individual_cases_probability_chosen_actions_over_trials_{dt}.png'))
    plt.close()


# Visualize model latents
disrnn.plot_bottlenecks(disrnn_params)
#disrnn.plot_update_rules(disrnn_params, make_disrnn)

