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
import seaborn as sns
import warnings
import scipy.io as sio
import copy
from datetime import datetime
from typing import Optional, List
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CogModelingRNNsTutorial.CogModelingRNNsTutorial import bandits, disrnn, hybrnn, plotting, rat_data, rnn_utils
from de_rnn import load_data_for_one_human, to_onehot, zs_to_onehot, dataset_size, format_into_datasets_flexible, preprocess_data

warnings.filterwarnings("ignore")

# Model-related functions
def make_disrnn():
    model = disrnn.HkDisRNN(
        ### TODO: changed to argument for obs_size
        obs_size=6,
        target_size=2,
        latent_size=16,
        update_mlp_shape=(8, 8),
        choice_mlp_shape=(8, 8),
        eval_mode=0.0,
        beta_scale=3,
        activation=jax.nn.relu
    )
    return model

def compute_log_likelihood_and_accuracy(dataset, model_fun, params):
    xs, actual_choices = next(dataset)
    n_trials_per_session, n_sessions = actual_choices.shape[:2]
    model_outputs, _ = rnn_utils.eval_model(model_fun, params, xs)
    predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))

    log_likelihood = 0
    correct_predictions = 0
    n = 0  # Total number of trials across sessions.

    for sess_i in range(n_sessions):
        for trial_i in range(n_trials_per_session):
            actual_choice = int(actual_choices[trial_i, sess_i])
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

'''
def evaluate(dataset, model_fun, params):
    xs, actual_choices = next(dataset)
    model_outputs, _ = rnn_utils.eval_model(model_fun, params, xs)
    predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))
    return predicted_log_choice_probabilities, actual_choices, xs
'''
def evaluate(xs, actual_choices, model_fun, params):
    model_outputs, _ = rnn_utils.eval_model(model_fun, params, xs)
    predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))
    return predicted_log_choice_probabilities, actual_choices, xs


def plot_session(choices: np.ndarray,
                 rewards: np.ndarray,
                 timeseries: np.ndarray,
                 timeseries_name: str,
                 labels: Optional[List[str]] = None,
                 fig_ax: Optional = None):
  """Plot data from a single behavioral session of the bandit task."""

  choose_high = choices == 1
  choose_low = choices == 0
  rewarded = rewards == 1

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
        np.argwhere(choose_high & 1 - rewarded),
        y_high * np.ones(np.sum(choose_high & 1 - rewarded)),
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
        np.argwhere(choose_low & 1 - rewarded),
        y_low * np.ones(np.sum(choose_low & 1 - rewarded)),
        color='red',
        marker='|')

  ax.set_xlabel('Trial')
  ax.set_ylabel(timeseries_name)
  plt.savefig("plot_session.png")

def plot_latent_activations_for_session(sess_i, xs_train, pred_train, act_train, bitResponseAIsCorr_train, P_A_train, disrnn_params, make_disrnn):
    """
    Plot latent activations from a single behavioral session.

    Parameters:
    - sess_i: Index of the session to analyze.
    - xs_train: Input data array with shape [trials, sessions, features].
    - pred_train: Model predictions with shape [trials, sessions, 2].
    - act_train: Actual actions with shape [trials, sessions, 1].
    - bitResponseAIsCorr_train: Correct optimal choices with shape [trials, sessions, 1].
    - P_A_train: Human probability of choosing action 0 with shape [trials, sessions, 1].
    - disrnn_params: Parameters for the DisRNN model.
    - make_disrnn: Function to create the DisRNN model.
    """

    trials, _, features = xs_train.shape

    # Extract session data from xs_train
    session_xs_train = xs_train[:, sess_i, :]  # Shape: (trials, features)
    session_xs_train = session_xs_train[:, np.newaxis, :]  # Shape: (trials, 1, features)

    trial_end = rnn_utils.find_session_end(session_xs_train[:, 0])

    # Evaluate network activations for the selected session
    network_outputs, network_states = rnn_utils.eval_model(make_disrnn, disrnn_params, session_xs_train[:trial_end, :])
    network_states = np.array(network_states)
    disrnn_activations = network_states[:trial_end, :] 
    labels = [f'Latent_{i+1}' for i in range(disrnn_activations.shape[1])]

    # Prepare session-specific data for plotting
    pred_session = pred_train[:, sess_i, :][:trial_end, :]  # Shape: (trial_end, 2)
    pred_session = pred_session[:, np.newaxis, :]  # Adjusted shape: (trial_end, 1, 2)

    act_session = act_train[:, sess_i, :][:trial_end, :]  # Shape: (trial_end, 1)
    act_session = act_session[:, np.newaxis, :]  # Adjusted shape: (trial_end, 1, 1)

    bitResponseAIsCorr_session = bitResponseAIsCorr_train[:, sess_i, :][:trial_end].squeeze()  # Shape: (trial_end,)
    P_A_session = P_A_train[:, sess_i, :][:trial_end].squeeze()  # Shape: (trial_end,)

    df_session = create_dataframe_for_plotting(pred_session, act_session, bitResponseAIsCorr_session, P_A_session, session_xs_train[:trial_end, :, :])
    choices = df_session["act"]
    rewards = df_session["rwd_manual"]

    plot_session(choices, rewards, timeseries=disrnn_activations, timeseries_name='Network Activations', labels=labels)


def plot_update_rules(params, make_network):
  """Generates visualizations of the update ruled of a disRNN.
  """

  def step(xs, state):
    core = make_network()
    output, new_state = core(jnp.expand_dims(jnp.array(xs), axis=0), state)
    return output, new_state

  _, step_hk = hk.transform(step)
  key = jax.random.PRNGKey(0)
  step_hk = jax.jit(step_hk)

  initial_state = np.array(rnn_utils.get_initial_state(make_network))
  reference_state = np.zeros(initial_state.shape)

  def plot_update_1d(params, unit_i, observations, titles):
    lim = 3
    state_bins = np.linspace(-lim, lim, 20)
    colormap = plt.cm.get_cmap('viridis', 3)
    colors = colormap.colors

    fig, ax = plt.subplots(
        1, len(observations), figsize=(len(observations) * 4, 5.5)
    )
    plt.subplot(1, len(observations), 1)
    plt.ylabel('Updated Activity')

    for observation_i in range(len(observations)):
      observation = observations[observation_i]
      plt.subplot(1, len(observations), observation_i + 1)

      plt.plot((-3, 3), (-3, 3), '--', color='grey')
      plt.plot((-3, 3), (0, 0), color='black')
      plt.plot((0, 0), (-3, 3), color='black')

      delta_states = np.zeros(shape=(len(state_bins), 1))
      for s_i in np.arange(len(state_bins)):
        state = reference_state
        state[0, unit_i] = state_bins[s_i]
        _, next_state = step_hk(
            params, key, observation, state
        )
        next_state = np.array(next_state)
        delta_states[s_i] = next_state[0, unit_i]  # - state[0, unit_i]

      plt.plot(state_bins, delta_states, color=colors[1])

      plt.title(titles[observation_i])
      plt.xlim(-lim, lim)
      plt.ylim(-lim, lim)
      plt.xlabel('Previous Activity')

      if isinstance(ax, np.ndarray):
        ax[observation_i].set_aspect('equal')
      else:
        ax.set_aspect('equal')
    return fig

  def plot_update_2d(params, unit_i, unit_input, observations, titles):
    lim = 3

    state_bins = np.linspace(-lim, lim, 20)
    colormap = plt.cm.get_cmap('viridis', len(state_bins))
    colors = colormap.colors

    fig, ax = plt.subplots(
        1, len(observations), figsize=(len(observations) * 2 + 10, 5.5)
    )
    plt.subplot(1, len(observations), 1)
    plt.ylabel('Updated Latent ' + str(unit_i + 1) + ' Activity')

    for observation_i in range(len(observations)):
      observation = observations[observation_i]
      plt.subplot(1, len(observations), observation_i + 1)

      plt.plot((-3, 3), (-3, 3), '--', color='grey')
      plt.plot((-3, 3), (0, 0), color='black')
      plt.plot((0, 0), (-3, 3), color='black')

      for si_i in np.arange(len(state_bins)):
        delta_states = np.zeros(shape=(len(state_bins), 1))
        for s_i in np.arange(len(state_bins)):
          state = reference_state
          state[0, unit_i] = state_bins[s_i]
          state[0, unit_input] = state_bins[si_i]
          print("s_i", si_i)
          print(state[0, unit_input])
          _, next_state = step_hk(params, key, observation, state)
          next_state = np.array(next_state)
          delta_states[s_i] = next_state[0, unit_i]

        plt.plot(state_bins, delta_states, color=colors[si_i])

      plt.title(titles[observation_i])
      plt.xlim(-lim, lim)
      plt.ylim(-lim, lim)
      plt.xlabel('Latent ' + str(unit_i + 1) + ' Activity')

      if isinstance(ax, np.ndarray):
        ax[observation_i].set_aspect('equal')
      else:
        ax.set_aspect('equal')
    return fig

  latent_sigmas = 2*jax.nn.sigmoid(
      jnp.array(params['hk_dis_rnn']['latent_sigmas_unsquashed'])
      )
  update_sigmas = 2*jax.nn.sigmoid(
      np.transpose(
          params['hk_dis_rnn']['update_mlp_sigmas_unsquashed']
          )
      )
  latent_order = np.argsort(
      params['hk_dis_rnn']['latent_sigmas_unsquashed']
      )
  figs = []

  # Loop over latents. Plot update rules
  for latent_i in latent_order:
    # If this latent's bottleneck is open
    if latent_sigmas[latent_i] < 0.5:
      # Which of its input bottlenecks are open?
      update_mlp_inputs = np.argwhere(update_sigmas[latent_i] < 0.9)
      # TODO: this needs to be checked
      # In our dataset, xs[0] is whether previous choice was correct
      # xs[1] is what was the previous choice
      choice_sensitive = np.any(update_mlp_inputs == 1)
      reward_sensitive = np.any(update_mlp_inputs == 0)
      # Choose which observations to use based on input bottlenecks
      if choice_sensitive and reward_sensitive:
        observations = ([0, 0], [0, 1], [1, 0], [1, 1])
        titles = ('Left, Unrewarded',
                  'Left, Rewarded',
                  'Right, Unrewarded',
                  'Right, Rewarded')
      elif choice_sensitive:
        observations = ([0, 0], [1, 0])
        titles = ('Choose Left', 'Choose Right')
      elif reward_sensitive:
        observations = ([0, 0], [0, 1])
        titles = ('Rewarded', 'Unreward')
      else:
        observations = ([0, 0],)
        titles = ('All Trials',)
      # Choose whether to condition on other latent values
      latent_sensitive = update_mlp_inputs[update_mlp_inputs > 1] - 2 - 4 #-4 because we have 4 other obs
      # Doesn't count if it depends on itself (this'll be shown no matter what)
      latent_sensitive = np.delete(
          latent_sensitive, latent_sensitive == latent_i
      )
      if not latent_sensitive.size:  # Depends on no other latents
        fig = plot_update_1d(params, latent_i, observations, titles)
      else:  # It depends on latents other than itself.
        print("latent sensitive", latent_sensitive)
        print(latent_sensitive[np.argmax(latent_sensitive)])
        fig = plot_update_2d(
            params,
            latent_i,
            latent_sensitive[np.argmax(latent_sensitive)],
            observations,
            titles,
        )
      if len(latent_sensitive) > 1:
        print(
            'WARNING: This update rule depends on more than one '
            + 'other latent. Plotting just one of them'
        )

      figs.append(fig)

  return figs

def plot_training_results(df_train, dataset_train):
    _trial_num = np.tile(np.arange(dataset_train._xs.shape[0]), [dataset_train._xs.shape[1], 1]).transpose()

    # Normalized likelihood vs upper bound
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='pred_acc', label='Normalized Log-Likelihood')
    sns.lineplot(data=df_train, x=_trial_num.ravel(), y='model_est_optimal_prob', label='Upper Bound', linestyle='--')
    plt.legend()
    plt.xlabel('Trial Number')
    plt.ylabel('Normalized Likelihood')
    plt.xlim(left=1)
    plt.ylim(bottom=0, top=1.1)
    plt.title('Training Data: Normalized Log-Likelihood vs Upper Bound per Trial')
    plt.savefig(os.path.join(plot_dir, 'normalized_log_likelihood_vs_upper_bound_train.png'))
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
    plt.savefig(os.path.join(plot_dir, 'probability_optimal_choices_over_trials_train.png'))
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
    plt.savefig(os.path.join(plot_dir, 'probability_chosen_actions_over_trials_train.png'))
    plt.close()

def create_dataframe_for_plotting(predictions, actions, bitResponseAIsCorr, P_A, xs):
    _c0 = actions == 0
    _c1 = actions == 1
    _cn = ~np.isin(actions, [0, 1])
    _c = np.concatenate([_c0, _c1, _cn], axis=2)
    _nan = np.empty([*predictions.shape[:2], 1])
    _nan.fill(np.nan)
    _pred_expand = np.concatenate([predictions, _nan], axis=2)
    _pred_acc = _pred_expand[_c]
    _pred_acc = _pred_acc.reshape(actions.shape)
    
    df = pd.DataFrame({
        'act': actions[:, :, 0].ravel(),  # human-chosen action
        'pred': np.exp(predictions[:, :, 1].ravel()),  # model-predicted action
        'pred_acc': np.exp(_pred_acc.ravel()),  # model-predicted probability of the human-chosen action
        'bitResponseAIsCorr': bitResponseAIsCorr,  # Correct optimal choices of the experiment
        'P_A': P_A,  # Human probability of choosing action 0
        # TODO: Change how dataframe is organized
        # This is whether previous action was correct, because df is organized by trials as opposed to sessions, this is not accurate
        'rwd': np.vstack((np.zeros((1, xs.shape[1])), xs[1:, :, 0])).ravel()
        # np.vstack((np.zeros((1, xs.shape[1])), xs[1:, :, 0])).ravel()  # Reward for human-chosen action
    })
    '''
    df = pd.DataFrame({
        'act': actions[:, :, 0].ravel(),  # human-chosen action
        'pred': np.exp(predictions[:, :, 1].ravel()),  # model-predicted action
        'pred_acc': np.exp(_pred_acc.ravel()),  # model-predicted probability of the human-chosen action
        'bitResponseAIsCorr': bitResponseAIsCorr[:, :, 0].ravel(),  # Correct optimal choices of the experiment
        'P_A': P_A[:, :, 0].ravel(),  # Human probability of choosing action 0
        'rwd': np.vstack((np.zeros((1, xs.shape[1])), xs[1:, :, 0])).ravel()  # Reward for human-chosen action
    })
'''

    # Compute the optimal choice and model-estimated probability
    # df['optimal_choice_manual'] = np.where(df['rwd'] == 1, df['act'], 1 - df['act'])
    df['model_est_optimal_prob'] = np.where(df['bitResponseAIsCorr'] == 1, df['pred'], 1 - df['pred'])
    df['human_chosen_action_prob'] = np.where(df['act'] == 1, df['P_A'], 1 - df['P_A'])
    # rwd_manual
    df['rwd_manual'] = np.where(df['act'] == df['bitResponseAIsCorr'], 1, 0)

    log_probs = np.log(df['human_chosen_action_prob'])
    normalized_likelihood_upper_bound = np.exp(np.mean(log_probs))
    print("Model Normalized likelihood Upperbound", normalized_likelihood_upper_bound)

    return df

def plot_bottlenecks(params, sort_latents=True, obs_names=None):
    """Plot the bottleneck sigmas from an hk.CompartmentalizedRNN."""
    params_disrnn = params['hk_dis_rnn']
    latent_dim = params_disrnn['latent_sigmas_unsquashed'].shape[0]
    obs_dim = params_disrnn['update_mlp_sigmas_unsquashed'].shape[0] - latent_dim

    if obs_names is None:
        if obs_dim == 2:
            obs_names = ['Choice', 'Reward']
        elif obs_dim == 5:
            obs_names = ['A', 'B', 'C', 'D', 'Reward']
        else: 
            obs_names = np.arange(1, obs_dim+1)

    latent_sigmas = 2 * jax.nn.sigmoid(
        jnp.array(params_disrnn['latent_sigmas_unsquashed'])
    )

    update_sigmas = 2 * jax.nn.sigmoid(
        np.transpose(
            params_disrnn['update_mlp_sigmas_unsquashed']
        )
    )

    if sort_latents:
        latent_sigma_order = np.argsort(
            params_disrnn['latent_sigmas_unsquashed']
        )
        latent_sigmas = latent_sigmas[latent_sigma_order]

        update_sigma_order = np.concatenate(
            (np.arange(0, obs_dim, 1), obs_dim + latent_sigma_order), axis=0
        )
        update_sigmas = update_sigmas[latent_sigma_order, :]
        update_sigmas = update_sigmas[:, update_sigma_order]

    latent_names = np.arange(1, latent_dim + 1)
    fig = plt.subplots(1, 2, figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.swapaxes([1 - latent_sigmas], 0, 1), cmap='Oranges')
    plt.clim(vmin=0, vmax=1)
    plt.yticks(ticks=range(latent_dim), labels=latent_names)
    plt.xticks(ticks=[])
    plt.ylabel('Latent #')
    plt.title('Latent Bottlenecks')

    plt.subplot(1, 2, 2)
    plt.imshow(1 - update_sigmas, cmap='Oranges')
    plt.clim(vmin=0, vmax=1)
    plt.colorbar()
    plt.yticks(ticks=range(latent_dim), labels=latent_names)
    xlabels = np.concatenate((np.array(obs_names), latent_names))
    plt.xticks(
        ticks=range(len(xlabels)),
        labels=xlabels,
        rotation='vertical',
    )
    plt.ylabel('Latent #')
    plt.title('Update MLP Bottlenecks')

    plt.tight_layout()
    plt.savefig("plot_bottlenecks.png")


    return fig

def main():
    # Directories for saving files
    checkpoint_dir = './checkpoints'
    plot_dir = './plots'
    loss_dir = './losses'

    dataset_type = 'RealWorldKimmelfMRIDataset'  
    local_file_path = "dataset/tensor_for_dRNN_desc-syn_nSubs-2000_nSessions-1_nBlocks-1_nTrialsPerBlock-100_b-0.3_NaN_30_0.93_0.45_NaN_NaN_withOptimalChoice_20240718_fast.mat"
    dataset_train, dataset_test, bitResponseAIsCorr_train, bitResponseAIsCorr_test, P_A_train, P_A_test = preprocess_data(dataset_type, local_file_path, 0.1)

    # Load the checkpoint parameters
    checkpoint_name = 'checkpoints/ls_16_umlp_[8, 8]_cmlp_[8, 8]_beta_1e-05_penalty_1e-10_20240831_141822/disrnn_params_ls_16_umlp_[8, 8]_cmlp_[8, 8]_penalty_1e-10_beta_1e-05_lr_1e-3.pkl'
    with open(checkpoint_name, 'rb') as file:
        disrnn_params = pickle.load(file)
    print(f'Loaded disrnn_params from {checkpoint_name}')

    # Compute log-likelihood and accuracy for training and test datasets
    print('Normalized Likelihoods and Accuracies for disRNN')
    print('Training Dataset')
    compute_log_likelihood_and_accuracy(dataset_train, make_disrnn, disrnn_params)
    #print('Held-Out Dataset')
    #compute_log_likelihood_and_accuracy(dataset_test, make_disrnn, disrnn_params)

    # Get the first batch of data from the dataset
    xs_train, act_train = next(dataset_train)
    xs_test, act_test = next(dataset_test)

    # Evaluate on the first batch
    pred_train, act_train, xs_train = evaluate(xs_train, act_train, make_disrnn, disrnn_params)
    pred_test, act_test, xs_test = evaluate(xs_test, act_test, make_disrnn, disrnn_params)
    df_train = create_dataframe_for_plotting(pred_train, act_train, bitResponseAIsCorr_train[:, :, 0].ravel(), P_A_train[:, :, 0].ravel(), xs_train)
    df_test = create_dataframe_for_plotting(pred_test, act_test, bitResponseAIsCorr_test[:, :, 0].ravel(), P_A_test[:, :, 0].ravel(), xs_test)

    # Plot results
    plot_training_results(df_train, dataset_train)
    print(df_train.tail(10))

    plot_bottlenecks(disrnn_params)
    session_i = 0
    plot_latent_activations_for_session(sess_i = session_i,
                                        xs_train=xs_train, 
                                        pred_train=pred_train, 
                                        act_train=act_train, 
                                        bitResponseAIsCorr_train=bitResponseAIsCorr_train, 
                                        P_A_train=P_A_train, 
                                        disrnn_params=disrnn_params, 
                                        make_disrnn=make_disrnn)
    # plot_update_rules(disrnn_params, make_disrnn)





    '''
    session_xs_train = xs_train[:, sess_i, :]
    session_act_train = act_train[:, sess_i, :]
    sess_i = 0
    trial_end = rnn_utils.find_session_end(xs_train[:, sess_i, 0])
    network_outputs, network_states = rnn_utils.eval_model(make_disrnn, disrnn_params, xs_train[:trial_end, sess_i:sess_i+1])
    network_states = np.array(network_states)
    disrnn_activations = network_states[:trial_end, sess_i, :]
    labels = [f'Latent_{i+1}' for i in range(disrnn_activations.shape[1])]
    choices = df_train["act"]
    rewards = df_train["rwd_manual"]
    plot_session(choices, rewards, timeseries=disrnn_activations, timeseries_name='Network Activations', labels=labels)
    '''

if __name__ == "__main__":
    main()
