# Imports
import os
import subprocess
import sys

# Imports + default settings
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
import math

# Import the necessary modules from the CogModelingRNNsTutorial package
from CogModelingRNNsTutorial.CogModelingRNNsTutorial import bandits, disrnn, hybrnn, plotting, rat_data, rnn_utils




# Define necessary functions
def load_data_for_one_human(fname, data_dir='./'):
    """Load data for one human subject from a MATLAB file and organize into features, labels, and state information."""
    """
    x.shape = (100, 1800, 18) 
    - xs (Features): This is a 3D array where each element represents a trial's features in a session.
    - Dimensions: (number of trials per session, number of sessions, number of features per trial)
    - Example: xs = np.array([
                        [[-1, -1], [0, 1], [0, 0], ...],
                        [[0, 1], [1, 1], [1, 0], ...],
                        ...
                        ])
    y.shape = (100, 1800, 1)
    - ys (Labels): This is a 3D array where each element represents the label (choice) for a trial in a session.
    - Dimensions: (number of trials per session, number of sessions, 1)
    - Example: ys = np.array([
                        [[-1], [0], [1], ...],
                        [[0], [1], [0], ...],
                        ...
                        ])
    - zs (State Information): This is a 3D array where each element represents the state information for a trial in a session.
    - Dimensions: (number of trials per session, number of sessions, 1)
    - Example: zs = np.array([
                    [[-1], [2], [2], ...],
                    [[1], [3], [0], ...],
                    ...
                    ])
    """
    
    # Load the MATLAB file
    mat_contents = sio.loadmat(os.path.join(data_dir, fname))
    data = mat_contents['tensor']
    var_names = mat_contents['vars_in_tensor']
    vars_for_state = [x.item() for x in mat_contents['vars_for_state'].ravel()]

    # Identify state pages
    is_state_page = (var_names == 'state').flatten()
    zs = data[:, :, is_state_page]
    zs = np.clip(zs - 1, a_min=-1, a_max=None)
    
    # Separate features (xs) and labels (ys)
    xs = data[:, :, ~is_state_page][..., :-1]
    ys = data[:, :, ~is_state_page][..., -1:]
    
    # Validate dimensions
    assert zs.shape[-1] == ys.shape[-1] == 1, "Mismatch in dimensions"
    assert zs.ndim == ys.ndim == xs.ndim == 3, "Data should be 3-dimensional"
    assert zs.max() < 16, "State values should be less than 16"
    
    # Transpose to desired format (trials, sessions, features)
    xs = np.transpose(xs, (1, 0, 2))
    ys = np.transpose(ys, (1, 0, 2))
    zs = np.transpose(zs, (1, 0, 2))
    
    # Print shapes for debugging
    print("features shape:", xs.shape)
    print("labels shape:", ys.shape)
    print("state information shape:", zs.shape)
    
    return xs, ys, zs, fname, vars_for_state

def to_onehot(labels, n):
    """Convert labels to one-hot encoding."""
    return np.eye(n)[labels]

def zs_to_onehot(zs):
    """Convert zs to one-hot encoding."""
    assert zs.shape[-1] == 1
    zs = zs[..., 0]
    minus1_mask = zs == -1
    zs_oh = to_onehot(zs, 16)
    zs_oh[minus1_mask] = 0
    assert np.all((zs_oh.sum(-1) == 0) == (zs == -1))
    return zs_oh

def format_into_datasets_flexible(xs, ys, dataset_constructor, testing_prop=0.5):
    """Format data into training and testing datasets."""
    import math
    if testing_prop == 0.5:
        n = int(xs.shape[1] // 2) * 2
        dataset_train = dataset_constructor(xs[:, :n:2], ys[:, :n:2])
        dataset_test = dataset_constructor(xs[:, 1:n:2], ys[:, 1:n:2])
    else:
        n = int(xs.shape[1])
        n_test = int(math.ceil(float(n) * testing_prop))
        n_train = int(n - n_test)
        assert n_train > 0
        assert n_test > 0
        idx = np.random.permutation(n)
        dataset_train = dataset_constructor(xs[:, idx[:n_train]], ys[:, idx[:n_train]])
        dataset_test = dataset_constructor(xs[:, idx[n_train:]], ys[:, idx[n_train:]])
    return dataset_train, dataset_test

def compute_log_likelihood(dataset, model_fun, params):
    """Compute the log likelihood of the model."""
    xs, actual_choices = next(dataset)
    n_trials_per_session, n_sessions = actual_choices.shape[:2]
    model_outputs, model_states = rnn_utils.eval_model(model_fun, params, xs)
    predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))
    log_likelihood = 0
    n = 0  # Total number of trials across sessions.
    for sess_i in range(n_sessions):
        for trial_i in range(n_trials_per_session):
            actual_choice = int(actual_choices[trial_i, sess_i])
            if (actual_choice >= 0) and (0 <= actual_choice < predicted_log_choice_probabilities.shape[2]):  # Ensure actual_choice is valid
                log_likelihood += predicted_log_choice_probabilities[trial_i, sess_i, actual_choice]
                n += 1
    normalized_likelihood = np.exp(log_likelihood / n)
    print(f'Normalized Likelihood: {100 * normalized_likelihood}%')
    return normalized_likelihood

# Load the dataset
dataset_type = 'RealWorldKimmelfMRIDataset'  # or other types as needed
is_synthetic = dataset_type.startswith('Synthetic')

if is_synthetic:
    if dataset_type == 'SyntheticSRContextTask':
        # Generate synthetic SR Context Task data
        num_part = 41
        num_sessions_per_part = 4
        num_trials_per_session = 120
        min_block_length = 10
        mean_block_length = 25
        max_block_length = 55
        fit_name = 'st_verd_ch_col_alp_sep_upM_all_factRwd_no_PC_1_MTr_all_aScReg_0_shOmg_0_conOmg_0_Inf_normM_0_MConst_0_MFroNorm_1'

        b = np.asarray([
            [0.1139, np.NaN, 10.6907, 0.9298, 0.4389, np.NaN, np.NaN],
            [0.0625, np.NaN, 17.4039, 0.3039, 0.3413, 0.0235, np.NaN],
            [0.0715, np.NaN, 19.2674, 0.6879, 0.7466, 0.0379, np.NaN],
            [0.0682, np.NaN, 22.4885, 0.2005, 0.6009, 0.0429, np.NaN]
        ])

        matlab_call = ("datasets = SR_generate_synthetic_data( " +
          "'num_part',{num_part}," +
          "'num_sessions_per_part',{num_sessions_per_part}," +
          "'num_trials_per_session',{num_trials_per_session}," +
          "'min_block_length',{min_block_length}," +
          "'mean_block_length',{mean_block_length}," +
          "'max_block_length',{max_block_length}," +
          "'b',{b}," +
          "'fit_name',{fit_name});").format(
              num_part = num_part,
              num_sessions_per_part = num_sessions_per_part,
              num_trials_per_session = num_trials_per_session,
              min_block_length = min_block_length,
              mean_block_length = mean_block_length,
              max_block_length = max_block_length,
              b = b,
              fit_name = fit_name,
          )

        # Start MATLAB engine and generate synthetic data
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.eval(matlab_call, nargout=0)
    else:
        # Generate Kevin's synthetic data
        gen_alpha = .25
        gen_beta = 5
        mystery_param = -2
        n_trials_per_session = 200
        n_sessions = 220
        sigma = .1
        environment = bandits.EnvironmentBanditsDrift(sigma=sigma)

        # Set up agent
        agent = {
            'SyntheticVanillaQ': bandits.VanillaAgentQ(gen_alpha, gen_beta),
            'SyntheticMysteryQ': bandits.MysteryAgentQ(gen_alpha, gen_beta, mystery_param=mystery_param),
        }[dataset_type]

        # Create training and testing datasets
        dataset_train, experiment_list_train = bandits.create_dataset(
            agent=agent,
            environment=environment,
            n_trials_per_session=n_trials_per_session,
            n_sessions=n_sessions
        )

        dataset_test, experiment_list_test = bandits.create_dataset(
            agent=agent,
            environment=environment,
            n_trials_per_session=n_trials_per_session,
            n_sessions=n_sessions
        )
else:
    if dataset_type in ['RealWorldRatDataset', 'RealWorldKimmelfMRIDataset']:
        # Load real-world dataset
        LOCAL_PATH_TO_FILE = "tensor_for_dRNN_desc-syn_1block_1session_100tr_nSubs-2000_20240415.mat"
        if not os.path.exists(LOCAL_PATH_TO_FILE):
            raise ValueError('File not found.')
        FNAME_ = LOCAL_PATH_TO_FILE

        gen_alpha = "unknown"
        gen_beta = "unknown"
        sigma = 0.1
        testing_set_proportion = 0.1
        environment = bandits.EnvironmentBanditsDrift(sigma=sigma)

        if dataset_type == 'RealWorldKimmelfMRIDataset':
            xs, ys, zs, *_ = load_data_for_one_human(FNAME_)
            zs_oh = zs_to_onehot(zs)
            inputs = np.concatenate([xs, zs_oh], axis=-1)

            dataset_train, dataset_test = format_into_datasets_flexible(
                inputs, ys.copy(), rnn_utils.DatasetRNN, testing_set_proportion
            )

            n_trials_per_session, n_sessions, _ = dataset_train._xs.shape
            experiment_list_train = None
            experiment_list_test = None

        elif dataset_type == 'RealWorldRatDataset':
            dataset_train, dataset_test = rat_data.format_into_datasets(
                *rat_data.load_data_for_one_rat(FNAME_, '.')[:2], rnn_utils.DatasetRNN
            )
            n_trials_per_session, n_sessions, _ = dataset_train._xs.shape
            experiment_list_train = None
            experiment_list_test = None

# Print the number of training and testing sessions
print('Training: {} sessions\nTesting: {} sessions'.format(dataset_train._xs.shape[1], dataset_test._xs.shape[1]))

# Example function usage: Print shape of a batch of data
x, y = next(dataset_train)
print(f'x.shape = {x.shape} \ny.shape = {y.shape}')


def make_gru(n_hidden=128, n_layers=1, output_size=2):
    layers = [hk.GRU(n_hidden) for _ in range(n_layers)] + [hk.Linear(output_size)]
    model = hk.DeepRNN(layers)
    return model

# Example usage:
n_hidden = 128  # Number of hidden units
n_layers = 1    # Number of GRU layers
output_size = 2 # Output size
print(n_hidden)
print(n_layers)
print(output_size)

# Create the GRU model with specified hyperparameters
gru_model_fun = lambda: make_gru(n_hidden=n_hidden, n_layers=n_layers, output_size=output_size)

'''
# Define GRU model
n_hidden = 64
def make_gru():
    model = hk.DeepRNN(
        [hk.GRU(n_hidden), hk.Linear(output_size=2)]
    )
    return model
'''

# Fit the RNN (GRU) model
n_steps_max = 10000
dataset_to_use = 'data_from_file'  # ['data_from_file', 'Kims_synthetic_data']
optimizer = optax.adam(learning_rate=1e-4)
if dataset_to_use == 'data_from_file':
    _dataset_train = dataset_train
    _dataset_test = dataset_test
elif dataset_to_use == 'Kims_synthetic_data':
    _dataset_train = dataset_train2
    _dataset_test = dataset_test2

gru_params, loss_final, losses_all = rnn_utils.fit_model(
    model_fun=gru_model_fun,
    dataset=_dataset_train,
    optimizer=optimizer,
    convergence_thresh=1e-3,
    n_steps_max=n_steps_max,
    return_all_losses=True,
    penalty_scale=0
)

# Plot loss function
plt.figure(figsize=(8,4))
plt.plot(np.arange(0, len(losses_all) * 10, 10), losses_all)
plt.xlabel('Training steps')
plt.ylabel('Model_loss_n_hidden_{n_hidden}_n_layers_{n_layers}_output_size_{output_size}')
plt.title('Loss Function')
plt.show()

# Compute quality-of-fit: Held-out Normalized Likelihood
print('Normalized Likelihoods for GRU')
print('Training Dataset')
training_likelihood = compute_log_likelihood(_dataset_train, make_gru, gru_params)
print('Held-Out Dataset')
testing_likelihood = compute_log_likelihood(_dataset_test, make_gru, gru_params)

dataset_test3 = copy.deepcopy(_dataset_test)
dataset_test3._ys = np.random.permutation(dataset_test3._ys)
print('Held-Out Shuffled Dataset')
testing_likelihood_shuffled = compute_log_likelihood(dataset_test3, make_gru, gru_params)


# Evaluate model
def evaluate(dataset, model_fun, params):
    xs, actual_choices = next(dataset)
    n_trials_per_session, n_sessions = actual_choices.shape[:2]
    model_outputs, model_states = rnn_utils.eval_model(model_fun, params, xs)
    predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))
    return predicted_log_choice_probabilities, actual_choices, xs

# Data loaded from file
pred_train, act_train, xs_train = evaluate(dataset_train, make_gru, gru_params)
pred_test, act_test, xs_test = evaluate(dataset_test, make_gru, gru_params)

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

df_train = pd.DataFrame({'act': act_train[:, :, 0].ravel(),
                         'pred': np.exp(pred_train[:, :, 1].ravel()),
                         'pred_acc': np.exp(_pred_train_acc.ravel()),
                         'rwd': np.vstack((np.zeros((1, xs_train.shape[1])), xs_train[1:, :, 0])).ravel()})

_c0 = act_test == 0
_c1 = act_test == 1
_cn = ~np.isin(act_test, [0, 1])
_c = np.concatenate([_c0, _c1, _cn], axis=2)
_nan = np.empty([*pred_test.shape[:2], 1])
_nan.fill(np.nan)
_pred_test_expand = np.concatenate([pred_test, _nan], axis=2)
_pred_test_acc = _pred_test_expand[_c]
_pred_test_acc = _pred_test_acc.reshape(act_test.shape)

df_test = pd.DataFrame({'act': act_test[:, :, 0].ravel(),
                        'pred': np.exp(pred_test[:, :, 1].ravel()),
                        'pred_acc': np.exp(_pred_test_acc.ravel()),
                        'rwd': np.vstack((np.zeros((1, xs_test.shape[1])), xs_test[1:, :, 0])).ravel()})

# Save plots
# Save plots
fig1 = sns.displot(df_train, x='pred', col='act').fig
fig1.suptitle('Training Data')
fig1.tight_layout()
fig1.savefig(f'training_data_plot1_n_hidden_{n_hidden}_n_layers_{n_layers}_output_size_{output_size}.png')

fig2 = sns.displot(df_test, x='pred', col='act').fig
fig2.suptitle('Testing Data')
fig2.tight_layout()
fig2.savefig(f'testing_data_plot1_n_hidden_{n_hidden}_n_layers_{n_layers}_output_size_{output_size}.png')

fig3 = sns.displot(df_train, x='pred', col='act', row='rwd').fig
fig3.suptitle('Training Data')
fig3.tight_layout()
fig3.savefig(f'training_data_plot2_n_hidden_{n_hidden}_n_layers_{n_layers}_output_size_{output_size}.png')

fig4 = sns.displot(df_test, x='pred', col='act', row='rwd').fig
fig4.suptitle('Testing Data')
fig4.tight_layout()
fig4.savefig(f'testing_data_plot2_n_hidden_{n_hidden}_n_layers_{n_layers}_output_size_{output_size}.png')



prob_incorrect = 0.0
n_states = 8
prop_invalid = 0.1

def get_choices_rewards(states_int, prob_incorrect=prob_incorrect, prop_invalid=prop_invalid):
    seq_length, batch_size = states_int.shape
    optimal_action = np.array((states_int % 2) == 0, dtype=float)
    is_action_incorrect = np.random.rand(*optimal_action.shape) <= prob_incorrect
    action = optimal_action.copy()
    action[is_action_incorrect] = (1 - optimal_action[is_action_incorrect])
    num_invalid_per_seq = math.ceil(seq_length * prop_invalid)
    foo = np.tile(np.arange(seq_length).T, batch_size).reshape(batch_size, seq_length).T
    rng = np.random.default_rng()
    invalid_idx_row = rng.permuted(foo, axis=0)[:num_invalid_per_seq, ...]
    invalid_idx_col = np.tile(np.arange(batch_size), (num_invalid_per_seq, 1))
    action[(invalid_idx_row, invalid_idx_col)] = -1
    reward = np.array(action == optimal_action, dtype=float)
    return action, reward

def get_dummy_dataset(dataset, n_states=n_states):
    dataset = copy.deepcopy(dataset)
    _xs = dataset._xs[..., -2:].copy()
    choices, rewards = _xs[..., :1], _xs[..., 1:]
    seq_length, batch_size, _ = dataset._xs.shape
    states_int = np.random.randint(0, n_states, (seq_length, batch_size))
    choices, rewards = get_choices_rewards(states_int)
    states_onehot = np.eye(n_states)[states_int]
    prev_choices = np.concatenate([np.zeros((1, batch_size)), choices])[:-1, :, None]
    prev_rewards = np.concatenate([np.zeros((1, batch_size)), rewards])[:-1, :, None]
    new_xs = np.concatenate([states_onehot, prev_choices, prev_rewards], axis=-1)
    dataset._xs = np.array(new_xs.copy(), dtype=float)
    dataset._ys = np.array(choices.copy()[..., None], dtype=float)
    assert np.all(dataset._xs[0, :, -2:] == 0)
    assert np.all(dataset._xs[1:, :, -2] == dataset._ys[:-1, :, 0])
    return dataset

# Generate simple (non-SR) synthetic data
dataset_train2 = get_dummy_dataset(dataset_train)
dataset_test2 = get_dummy_dataset(dataset_test)

# Downsample dataset to last N trials
keep_last_n_trials = 0  # Set to 0 to keep all trials
dataset_train._xs = dataset_train._xs[-keep_last_n_trials:]
dataset_train._ys = dataset_train._ys[-keep_last_n_trials:]
dataset_test._xs = dataset_test._xs[-keep_last_n_trials:]
dataset_test._ys = dataset_test._ys[-keep_last_n_trials:]
(dataset_train._xs.shape, dataset_test._xs.shape)
