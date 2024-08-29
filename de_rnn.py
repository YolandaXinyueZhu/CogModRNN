import argparse
import os
import pickle
import jax
import jax.numpy as jnp
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
# Import the necessary modules from the CogModelingRNNsTutorial package
from CogModelingRNNsTutorial.CogModelingRNNsTutorial import bandits, disrnn, hybrnn, plotting, rat_data, rnn_utils
# Suppress warnings
import warnings
import itertools

warnings.filterwarnings("ignore")


# Define necessary functions
def load_data_for_one_human(fname, data_dir='./'):
    """Load data for one human subject from a MATLAB file and organize into features, labels, and state information."""
    mat_contents = sio.loadmat(os.path.join(data_dir, fname))
    data = mat_contents['tensor']
    var_names = mat_contents['vars_in_tensor']
    print(var_names)
    vars_for_state = [x.item() for x in mat_contents['vars_for_state'].ravel()]

    '''
    is_state_page = (var_names == 'state').flatten()
    zs = data[:, :, is_state_page]
    zs = np.clip(zs - 1, a_min=-1, a_max=None)
    xs = data[:, :, ~is_state_page][..., :-1]
    ys = data[:, :, ~is_state_page][..., -1:]
    '''
    
    is_state_page = (var_names == 'state').flatten()
    zs = data[:, :, is_state_page]
    zs = np.clip(zs - 1, a_min=-1, a_max=None)
    # Extract ys, corresponding to the choice on the present trial, "bitResponseA"
    ys = data[:, :, (var_names == 'bitResponseA').flatten()]
    # Extract the previous choice "bitResponseA_prev"
    bitResponseA_prev = data[:, :, (var_names == 'bitResponseA_prev').flatten()]
    # Extract whether the previous choice was correct "bitCorr_prev"
    bitCorr_prev = data[:, :, (var_names == 'bitCorr_prev').flatten()]
    # Concatenate the vars for the inputs xs
    xs = np.concatenate([bitCorr_prev,bitResponseA_prev], axis=2)
    


    assert zs.shape[-1] == ys.shape[-1] == 1, "Mismatch in dimensions"
    assert zs.ndim == ys.ndim == xs.ndim == 3, "Data should be 3-dimensional"
    assert zs.max() < 16, "State values should be less than 16"

    xs = np.transpose(xs, (1, 0, 2))
    ys = np.transpose(ys, (1, 0, 2))
    zs = np.transpose(zs, (1, 0, 2))

    zs_oh = zs_to_onehot(zs)

    if '_nBlocks-1_' in fname:
        zs_oh = zs_oh[...,:4]
    elif '_nBlocks-2_' in fname:
        zs_oh = zs_oh[...,:8]
    else:
        raise Exception("Dataset format is incorrect or contains more than two blocks of data.")

    # Extract optimal choice indicator and probability of choosing A
    bitResponseAIsCorr = data[:, :, (var_names == 'bitResponseAIsCorr').flatten()]
    P_A = data[:, :, (var_names == 'P_A').flatten()]

    bitResponseAIsCorr = np.transpose(bitResponseAIsCorr, (1, 0, 2))
    P_A = np.transpose(P_A, (1, 0, 2))


    return xs, ys, zs_oh, fname, vars_for_state, bitResponseAIsCorr, P_A

'''
def load_data_for_one_human(fname, data_dir='./'):
    """Load data for one human subject from a MATLAB file and organize into features, labels, and state information."""
    mat_contents = sio.loadmat(os.path.join(data_dir, fname))
    data = mat_contents['tensor']
    var_names = mat_contents['vars_in_tensor']
    vars_for_state = [x.item() for x in mat_contents['vars_for_state'].ravel()]

    #zs contains what the human sees for the current trial
    is_state_page = (var_names == 'state').flatten()
    zs = data[:, :, is_state_page]
    zs = np.clip(zs - 1, a_min=-1, a_max=None)

    # ys contains one thing, the choice that human currently make.
    # Extract ys, corresponding to the choice on the present trial, "bitResponseA"
    ys = data[:, :, (var_names == 'bitResponseA').flatten()]
    
    # xs contain two things: 1. whether the previous choice was correct. 2. what was the previous choice
    # Extract the previous choice "bitResponseA_prev"
    bitResponseA_prev = data[:, :, (var_names == 'bitResponseA_prev').flatten()]
    # Extract whether the previous choice was correct "bitCorr_prev"
    bitCorr_prev = data[:, :, (var_names == 'bitCorr_prev').flatten()]
    # Concatenate the vars for the inputs xs.
    xs = np.concatenate([bitCorr_prev,bitResponseA_prev], axis=2)

    assert zs.shape[-1] == ys.shape[-1] == 1, "Mismatch in dimensions"
    assert zs.ndim == ys.ndim == xs.ndim == 3, "Data should be 3-dimensional"
    assert zs.max() < 16, "State values should be less than 16"

    xs = np.transpose(xs, (1, 0, 2))
    ys = np.transpose(ys, (1, 0, 2))
    zs = np.transpose(zs, (1, 0, 2))

    zs_oh = zs_to_onehot(zs)

    if '_nBlocks-1_' in fname:
        zs_oh = zs_oh[...,:4]
    elif '_nBlocks-2_' in fname:
        zs_oh = zs_oh[...,:8]
    else:
        raise Exception("Dataset format is incorrect or contains more than two blocks of data.")

    return xs, ys, zs_oh, fname
'''

def to_onehot(labels, n):
    labels = np.asarray(labels, dtype=int)  
    return np.eye(n)[labels]

def zs_to_onehot(zs):
    assert zs.shape[-1] == 1
    zs = zs[..., 0]
    minus1_mask = zs == -1
    zs_oh = to_onehot(zs, 16)
    zs_oh[minus1_mask] = 0
    assert np.all((zs_oh.sum(-1) == 0) == (zs == -1))
    return zs_oh
'''
def format_into_datasets_flexible(xs, ys, dataset_constructor, testing_prop=0.5):
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
'''
def format_into_datasets_flexible(xs, ys, optimal_choice, P_A, dataset_constructor, testing_prop=0.5):
    import math
    if testing_prop == 0.5:
        n = int(xs.shape[1] // 2) * 2
        dataset_train = dataset_constructor(xs[:, :n:2], ys[:, :n:2])
        dataset_test = dataset_constructor(xs[:, 1:n:2], ys[:, 1:n:2])

        optimal_choice_train = optimal_choice[:, :n:2]
        optimal_choice_test = optimal_choice[:, 1:n:2]
        P_A_train = P_A[:, :n:2]
        P_A_test = P_A[:, 1:n:2]
        
    else:
        n = int(xs.shape[1])
        n_test = int(math.ceil(float(n) * testing_prop))
        n_train = int(n - n_test)
        assert n_train > 0
        assert n_test > 0
        idx = np.random.permutation(n)
        dataset_train = dataset_constructor(xs[:, idx[:n_train]], ys[:, idx[:n_train]])
        dataset_test = dataset_constructor(xs[:, idx[n_train:]], ys[:, idx[n_train:]])

        optimal_choice_train = optimal_choice[:, idx[:n_train]]
        optimal_choice_test = optimal_choice[:, idx[n_train:]]
        P_A_train = P_A[:, idx[:n_train]]
        P_A_test = P_A[:, idx[n_train:]]
        
    return dataset_train, dataset_test, optimal_choice_train, optimal_choice_test, P_A_train, P_A_test

def compute_log_likelihood(dataset, model_fun, params):
    xs, actual_choices = next(dataset)
    print(f"xs shape: {xs.shape}, actual_choices shape: {actual_choices.shape}")
    n_trials_per_session, n_sessions = actual_choices.shape[:2]
    model_outputs, model_states = rnn_utils.eval_model(model_fun, params, xs)
    print(f"model_outputs shape: {model_outputs.shape}")
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

def dataset_size(xs, ys):
    """Calculate the size of the dataset in terms of number of samples."""
    return xs.shape[0] * xs.shape[1], ys.shape[0] * ys.shape[1]  # (number of timesteps * number of episodes)

def make_disrnn():
        model = disrnn.HkDisRNN(
            obs_size=18,
            target_size=2,
            latent_size=32,
            update_mlp_shape=(8,8),
            choice_mlp_shape=(8,8),
            eval_mode=0.0,
            beta_scale=1e-05,
            activation=jax.nn.relu
        )
        return model

def preprocess_data(dataset_type, LOCAL_PATH_TO_FILE, testing_set_proportion):
    if dataset_type in ['RealWorldRatDataset', 'RealWorldKimmelfMRIDataset']:
        if not os.path.exists(LOCAL_PATH_TO_FILE):
            raise ValueError('File not found.')
        
        xs, ys, zs, _, _, bitResponseAIsCorr, P_A = load_data_for_one_human(LOCAL_PATH_TO_FILE)
        inputs = np.concatenate([xs, zs], axis=-1)

        dataset_train, dataset_test, bitResponseAIsCorr_train, bitResponseAIsCorr_test, P_A_train, P_A_test = format_into_datasets_flexible(
            inputs, ys.copy(), bitResponseAIsCorr, P_A, rnn_utils.DatasetRNN, testing_set_proportion
        )

        train_size, test_size = dataset_size(dataset_train._xs, dataset_test._xs)
        print(f'Training dataset size: {train_size} samples')
        print(f'Testing dataset size: {test_size} samples')

        return dataset_train, dataset_test, bitResponseAIsCorr_train, bitResponseAIsCorr_test, P_A_train, P_A_test
    else:
        raise ValueError('Unsupported dataset type.')
    


def train_model(latent_size, update_mlp_shape, choice_mlp_shape, dataset_train, dataset_test):
    x, y = next(dataset_train)
    beta_values = [1, 3, 10]
    penalty_scales = [1e-10]

    os.makedirs('plots', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('loss', exist_ok=True)

    for beta_scale, penalty_scale in itertools.product(beta_values, penalty_scales):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subdir = f'ls_{latent_size}_umlp_{update_mlp_shape}_cmlp_{choice_mlp_shape}_beta_{beta_scale}_penalty_{penalty_scale}_{timestamp}'
        plot_dir = os.path.join('plots', subdir)
        checkpoint_dir = os.path.join('checkpoints', subdir)
        loss_dir = os.path.join('loss', subdir)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)

        print(f"Training with beta_scale: {beta_scale} and penalty_scale: {penalty_scale}")

        def make_disrnn():
            model = disrnn.HkDisRNN(
                obs_size=x.shape[2],
                target_size=2,
                latent_size=latent_size,
                update_mlp_shape=update_mlp_shape,
                choice_mlp_shape=choice_mlp_shape,
                eval_mode=0.0,
                beta_scale=beta_scale,
                activation=jax.nn.relu
            )
            return model

        optimizer = optax.adam(learning_rate=1e-3)

        disrnn_params, opt_state, losses = rnn_utils.fit_model(
            model_fun=make_disrnn,
            dataset=dataset_train,
            optimizer=optimizer,
            loss_fun='penalized_categorical',
            convergence_thresh=1e-3,
            n_steps_max=10000,
            n_steps_per_call=500,
            return_all_losses=True,
            penalty_scale=penalty_scale
        )

        plt.figure(figsize=(16, 8))
        plt.plot(np.arange(0, len(losses) * 10, 10), losses)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title(f'Loss over Training (beta_scale: {beta_scale}, penalty_scale: {penalty_scale})')
        plt.savefig(os.path.join(loss_dir, f'loss_over_training_ls_{latent_size}_umlp_{update_mlp_shape}_cmlp_{choice_mlp_shape}_penalty_{penalty_scale}_beta_{beta_scale}_lr_1e-3.png'))

        # Save the model parameters
        filename = os.path.join(checkpoint_dir, f'disrnn_params_ls_{latent_size}_umlp_{update_mlp_shape}_cmlp_{choice_mlp_shape}_penalty_{penalty_scale}_beta_{beta_scale}_lr_1e-3.pkl')
        with open(filename, 'wb') as file:
            pickle.dump(disrnn_params, file)

        print(f'Saved disrnn_params to {filename}')
        
        return disrnn_params

def evaluate_model(disrnn_params, dataset_train, dataset_test, latent_size, update_mlp_shape, choice_mlp_shape):
    def evaluate(dataset, model_fun, params):
        xs, actual_choices = next(dataset)
        model_outputs, _ = rnn_utils.eval_model(model_fun, params, xs)
        predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :2]))
        return predicted_log_choice_probabilities, actual_choices, xs

    pred_train, act_train, xs_train = evaluate(dataset_train, make_disrnn, disrnn_params)
    pred_test, act_test, xs_test = evaluate(dataset_test, make_disrnn, disrnn_params)

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

    for dt, df, ds in [('train', df_train, dataset_train), ('test', df_test, dataset_test)]:
        _trial_num = np.tile(np.arange(ds._xs.shape[0]), [ds._xs.shape[1], 1]).transpose()

        plt.figure(figsize=(16, 8))
        _ax = sns.lineplot(data=df, x=_trial_num.ravel(), y='rwd', label='actual')
        sns.lineplot(data=df, x=_trial_num.ravel(), y='pred_acc', ax=_ax, label='pred.')
        plt.legend()
        plt.xlabel('Trial Number')
        plt.ylabel('Proportion Correct')
        plt.xlim(left=1)
        plt.ylim(bottom=0.45)
        plt.title(f'{dt.capitalize()} Data: Accuracy over Trials')
        plt.savefig(os.path.join('plots', f'accuracy_over_trials_{dt}.png'))


def main(latent_size, update_mlp_shape, choice_mlp_shape):
    gpu_devices = jax.devices("gpu")

    if gpu_devices:
        print(f"JAX is using GPU: {gpu_devices}")
    else:
        print("No GPU found, JAX is using CPU.")

    # Preprocess Data
    dataset_type = 'RealWorldKimmelfMRIDataset'
    LOCAL_PATH_TO_FILE = "tensor_for_dRNN_desc-syn_nSubs-2000_nSessions-1_nBlocks-1_nTrialsPerBlock-100_b-0.3_NaN_30_0.93_0.45_NaN_NaN_withOptimalChoice_20240718_fast.mat"
    testing_set_proportion = 0.1

    dataset_train, dataset_test = preprocess_data(dataset_type, LOCAL_PATH_TO_FILE, testing_set_proportion)

    # Train model
    disrnn_params = train_model(latent_size, update_mlp_shape, choice_mlp_shape, dataset_train, dataset_test)

    # Evaluate model
    evaluate_model(disrnn_params, dataset_train, dataset_test, latent_size, update_mlp_shape, choice_mlp_shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_size", type=int, default=16, help="Number of latent units in the model")
    parser.add_argument("--update_mlp_shape", nargs=2, type=int, default=[8, 8], help="Number of hidden units in each of the two layers of the update MLP")
    parser.add_argument("--choice_mlp_shape", nargs=2, type=int, default=[8, 8], help="Number of hidden units in each of the two layers of the choice MLP")

    args = parser.parse_args()

    main(args.latent_size, args.update_mlp_shape, args.choice_mlp_shape)
