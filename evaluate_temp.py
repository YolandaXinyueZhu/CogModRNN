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
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CogModelingRNNsTutorial.CogModelingRNNsTutorial import bandits, disrnn, hybrnn, plotting, rat_data, rnn_utils
from train_disrnn import load_data, to_onehot, zs_to_onehot, dataset_size, create_train_test_datasets, preprocess_data

warnings.filterwarnings("ignore")

# Model-related functions
def compute_log_likelihood_and_accuracy(xs, ys, model_fun, params):
    n_trials_per_session, n_sessions = ys.shape[:2]
    model_outputs, _ = rnn_utils.eval_model(model_fun, params, xs)
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
    # If this latent's bottleneck is o 
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
    checkpoint_dir = './checkpoints'
    plot_dir = './plots'
    loss_dir = './losses'

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
    action_0 = actions == 0
    action_1 = actions == 1
    action_others = ~np.isin(actions, [0, 1])
    action_oh = np.concatenate([action_0, action_1, action_others], axis=2)
    nan = np.empty([*predictions.shape[:2], 1])
    nan.fill(np.nan)
    predictions_expanded = np.concatenate([predictions, nan], axis=2)
    prediction_prob = predictions_expanded[action_oh] # Prediction_prob shape (20000, )
    prediction_prob = prediction_prob.reshape(actions.shape)
    
    df = pd.DataFrame({
        'act': actions[:, :, 0].ravel(),  # human-chosen action
        'pred': np.exp(predictions[:, :, 1].ravel()),  # model-predicted action
        'pred_acc': np.exp(prediction_prob.ravel()),  # model-predicted probability of the human-chosen action
        'bitResponseAIsCorr': bitResponseAIsCorr,  # Correct optimal choices of the experiment
        'P_A': P_A,  # Human probability of choosing action 0
        # TODO: Change how dataframe is organized
        # This is whether previous action was correct, because df is organized by trials as opposed to sessions, this is not accurate
        'rwd': np.vstack((np.zeros((1, xs.shape[1])), xs[1:, :, 0])).ravel()
        # np.vstack((np.zeros((1, xs.shape[1])), xs[1:, :, 0])).ravel()  # Reward for human-chosen action
    })

    # Compute the optimal choice and model-estimated probability
    # df['optimal_choice_manual'] = np.where(df['rwd'] == 1, df['act'], 1 - df['act'])
    df['model_est_optimal_prob'] = np.where(df['bitResponseAIsCorr'] == 1, df['pred'], 1 - df['pred'])
    df['human_chosen_action_prob'] = np.where(df['act'] == 1, df['P_A'], 1 - df['P_A'])
    # rwd_manual
    df['rwd_manual'] = np.where(df['act'] == df['bitResponseAIsCorr'], 1, 0)

    return df

def plot_bottlenecks(params, save_folder_name, sort_latents=True, obs_names=None):
    """Plot the bottleneck sigmas from an hk.CompartmentalizedRNN and save sigma values to text files.
    
    Additionally, the plot indicates whether darker colors correspond to higher sigma values.
    """
    import matplotlib.colors as mcolors  # Importing here to keep imports organized
    
    plot_dir = 'plots/bottlenecks'
    save_dir = os.path.join(plot_dir, save_folder_name)
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    
    # Extract the base name to use as the plot filename
    base_name = os.path.basename(save_folder_name)
    if not base_name:
        # If save_folder_name ends with a slash, get the second last part
        base_name = os.path.basename(os.path.dirname(save_folder_name))
    
    plot_path = os.path.join(save_dir, f"{base_name}.png")  # Path to save the plot
    
    # Extract parameters for DisRNN
    params_disrnn = params['hk_dis_rnn']
    latent_dim = params_disrnn['latent_sigmas_unsquashed'].shape[0]
    update_mlp_dim = params_disrnn['update_mlp_sigmas_unsquashed'].shape[1]  # Assuming shape [latent, update_mlp_inputs]
    obs_dim = params_disrnn['update_mlp_sigmas_unsquashed'].shape[0] - latent_dim
    
    # Assign observation names if not provided
    if obs_names is None:
        if obs_dim == 2:
            obs_names = ['Choice', 'Reward']
        elif obs_dim == 5:
            obs_names = ['A', 'B', 'C', 'D', 'Reward']
        else: 
            obs_names = [f'Obs_{i}' for i in range(1, obs_dim+1)]
    
    # Compute sigmas using sigmoid activation scaled by 2
    latent_sigmas = 2 * jax.nn.sigmoid(
        jnp.array(params_disrnn['latent_sigmas_unsquashed'])
    )
    
    update_sigmas = 2 * jax.nn.sigmoid(
        np.transpose(
            params_disrnn['update_mlp_sigmas_unsquashed']
        )
    )
    
    # Optionally sort latents based on their sigmas for better visualization
    if sort_latents:
        latent_sigma_order = np.argsort(
            params_disrnn['latent_sigmas_unsquashed']
        )
        latent_sigmas = latent_sigmas[latent_sigma_order]
    
        update_sigma_order = np.concatenate(
            (np.arange(0, obs_dim, 1), latent_sigma_order + obs_dim),
            axis=0
        )
        update_sigmas = update_sigmas[latent_sigma_order, :]
        update_sigmas = update_sigmas[:, update_sigma_order]
    
    # Assign names to latent variables
    latent_names = [f'Latent_{i+1}' for i in range(latent_dim)]
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(25, 12))
    
    # Define a colormap where higher sigma corresponds to more intense (darker) colors
    cmap='Oranges'
    
    # Plot Latent Bottlenecks
    ax1 = axes[0]
    im1 = ax1.imshow(latent_sigmas.reshape(1, -1), cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax1.set_yticks([0])
    ax1.set_yticklabels(['Latent #'])
    ax1.set_xticks(range(latent_dim))
    ax1.set_xticklabels(latent_names, rotation=90)
    ax1.set_title('Latent Bottlenecks (Sigma)')
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Sigma Value')
    
    # Plot Update MLP Bottlenecks
    ax2 = axes[1]
    im2 = ax2.imshow(update_sigmas, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax2.set_yticks(range(latent_dim))
    ax2.set_yticklabels(latent_names)
    xlabels = obs_names + latent_names
    ax2.set_xticks(range(len(xlabels)))
    ax2.set_xticklabels(xlabels, rotation=90)
    ax2.set_title('Update MLP Bottlenecks (Sigma)')
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Sigma Value')
    
    # Add annotation to indicate that darker colors represent higher sigma
    fig.text(0.5, 0.95, 'Darker Colors → Higher Sigma Values', ha='center', fontsize=14, color='black')
    
    plt.tight_layout()
    plt.savefig(plot_path)  # Save the plot
    plt.close(fig)  # Close the figure to free memory
    
    # **Printing Sigmas**
    print("\n=== Latent Bottleneck Sigmas ===")
    for i, sigma in enumerate(latent_sigmas):
        print(f"Latent {i+1}: Sigma = {sigma:.4f}")
    
    print("\n=== Update MLP Bottleneck Sigmas ===")
    for i in range(update_sigmas.shape[0]):
        print(f"Latent {i+1} Update MLP Sigmas:")
        for j, sigma in enumerate(update_sigmas[i]):
            print(f"  Input {j+1} ({xlabels[j]}): Sigma = {sigma:.4f}")
    
    # **Saving Sigmas to Text Files**
    # Define file paths
    latent_sigmas_path = os.path.join(save_dir, "latent_sigmas.txt")
    update_sigmas_path = os.path.join(save_dir, "update_mlp_sigmas.txt")
    
    # Save Latent Sigmas
    with open(latent_sigmas_path, 'w') as f:
        f.write("Latent Bottleneck Sigmas:\n")
        for i, sigma in enumerate(latent_sigmas):
            f.write(f"Latent {i+1}: {sigma:.6f}\n")
    
    # Save Update MLP Sigmas
    with open(update_sigmas_path, 'w') as f:
        f.write("Update MLP Bottleneck Sigmas:\n")
        for i in range(update_sigmas.shape[0]):
            f.write(f"Latent {i+1} Update MLP Sigmas:\n")
            for j, sigma in enumerate(update_sigmas[i]):
                f.write(f"  Input {j+1} ({xlabels[j]}): {sigma:.6f}\n")
            f.write("\n")
    
    print(f"\nSigma values have been saved to:\n- {latent_sigmas_path}\n- {update_sigmas_path}")
    
    return fig


def main(seed, saved_checkpoint_pth):
    np.random.seed(seed)

    dataset_type = 'RealWorldKimmelfMRIDataset'  
    dataset_path = "dataset/tensor_for_dRNN_desc-syn_nSubs-2000_nSessions-1_nBlocks-1_nTrialsPerBlock-100_b-0.3_NaN_30_0.93_0.45_NaN_NaN_withOptimalChoice_20240718_fast.mat"
    dataset_train, dataset_test, bitResponseAIsCorr_train, bitResponseAIsCorr_test, P_A_train, P_A_test = preprocess_data(dataset_type, dataset_path, 0.1)

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

    # Compute log-likelihood and accuracy for training and test datasets
    print('Normalized Likelihoods and Accuracies for disRNN')
    print('Training Dataset')
    compute_log_likelihood_and_accuracy(xs_train, ys_train, make_disrnn, disrnn_params)
    print('Held-Out Dataset')
    compute_log_likelihood_and_accuracy(xs_test, ys_test, make_disrnn, disrnn_params)
    shuffled_dataset_test = copy.deepcopy(dataset_test)
    shuffled_dataset_test._ys = np.random.permutation(shuffled_dataset_test._ys)
    shuffled_xs_test, shuffled_ys_test = next(shuffled_dataset_test)
    print(shuffled_ys_test.shape)
    print('Held-Out Shuffled Dataset')
    compute_log_likelihood_and_accuracy(shuffled_xs_test, shuffled_ys_test, make_disrnn, disrnn_params)


    pred_train, ys_train, xs_train = evaluate(xs_train, ys_train, make_disrnn, disrnn_params)
    pred_test, ys_test, xs_test = evaluate(xs_test, ys_test, make_disrnn, disrnn_params)
    df_train = create_dataframe_for_plotting(pred_train, ys_train, bitResponseAIsCorr_train[:, :, 0].ravel(), P_A_train[:, :, 0].ravel(), xs_train)
    log_probs_train = np.log(df_train['human_chosen_action_prob'])
    normalized_likelihood_upper_bound_train = np.exp(np.mean(log_probs_train))
    print("Model Normalized likelihood Upperbound for training", normalized_likelihood_upper_bound_train)
    df_test = create_dataframe_for_plotting(pred_test, ys_test, bitResponseAIsCorr_test[:, :, 0].ravel(), P_A_test[:, :, 0].ravel(), xs_test)
    log_probs_test = np.log(df_test['human_chosen_action_prob'])
    normalized_likelihood_upper_bound_test = np.exp(np.mean(log_probs_test))
    print("Model Normalized likelihood Upperbound for testing", normalized_likelihood_upper_bound_test)

    # Plot results
    save_folder_name = saved_checkpoint_pth.split('checkpoints/')[1].split('.pkl')[0]
    plot_training_results(df_train, dataset_train)
    print(df_train.tail(10))

    plot_bottlenecks(disrnn_params, save_folder_name)
    session_i = 0
    plot_latent_activations_for_session(sess_i = session_i,
                                        xs_train=xs_train, 
                                        pred_train=pred_train, 
                                        act_train=ys_test, 
                                        bitResponseAIsCorr_train=bitResponseAIsCorr_train, 
                                        P_A_train=P_A_train, 
                                        disrnn_params=disrnn_params, 
                                        make_disrnn=make_disrnn)
    # plot_update_rules(disrnn_params, make_disrnn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", nargs=1, type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--saved_checkpoint_pth", type=str, required=True, help="Saved checkpoint path for evaluation.")

    args = parser.parse_args()

    main(args.seed, args.saved_checkpoint_pth)
