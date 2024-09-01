import argparse
import os
import pickle
import sys
import math
import copy
import warnings
from datetime import datetime

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CogModelingRNNsTutorial.CogModelingRNNsTutorial import bandits, disrnn, hybrnn, plotting, rat_data, rnn_utils

warnings.filterwarnings("ignore")

def load_data(fname, data_dir='./'):
    """Load data for one human subject from a MATLAB file and organize into features, labels, and state information.
    
    mat_contents contains the following keys:
    dict_keys(['__header__', '__version__', '__globals__', 'tensor', 'vars_in_tensor', 'col_names', 'vars_for_state'])
    'tensor': A 3D NumPy array containing all the data. The shape is typically (samples, time_steps, features).
    'vars_in_tensor': A nested list of arrays, where each inner array contains a single string representing a variable name.
    For example:
    [[array(['state'], dtype='<U5')],
    [array(['bitResponseAIsCorr'], dtype='<U18')],
    [array(['P_A'], dtype='<U3')],
    [array(['context'], dtype='<U7')],
    [array(['blockN'], dtype='<U6')],
    [array(['trialNInBlock'], dtype='<U13')],
    [array(['bitCorr_prev'], dtype='<U12')],
    [array(['bitResponseA_prev'], dtype='<U17')],
    [array(['bitResponseA'], dtype='<U12')]]
    
    'vars_for_state': A nested list of arrays, where each inner array contains a single string representing a variable name related to state information.
    For example:
    [[array(['sessionN'], dtype='<U8')],
    [array(['stimulusSlotID'], dtype='<U14')]] 
    
    """
    mat_contents = sio.loadmat(os.path.join(data_dir, fname))
    data = mat_contents['tensor']
    var_names = mat_contents['vars_in_tensor']
    print("feature_names", var_names)
    vars_for_state = [x.item() for x in mat_contents['vars_for_state'].ravel()]
    
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

def preprocess_data(dataset_type, LOCAL_PATH_TO_FILE, testing_set_proportion):
    if dataset_type in ['RealWorldRatDataset', 'RealWorldKimmelfMRIDataset']:
        if not os.path.exists(LOCAL_PATH_TO_FILE):
            raise ValueError('File not found.')
        
        xs, ys, zs, _, _, bitResponseAIsCorr, P_A = load_data(LOCAL_PATH_TO_FILE)
        inputs = np.concatenate([xs, zs], axis=-1)

        dataset_train, dataset_test, bitResponseAIsCorr_train, bitResponseAIsCorr_test, P_A_train, P_A_test = create_train_test_datasets(
            inputs, ys.copy(), bitResponseAIsCorr, P_A, rnn_utils.DatasetRNN, testing_set_proportion
        )

        train_size, test_size = dataset_size(dataset_train._xs, dataset_test._xs)
        print(f'Training dataset size: {train_size} samples')
        print(f'Testing dataset size: {test_size} samples')

        return dataset_train, dataset_test, bitResponseAIsCorr_train, bitResponseAIsCorr_test, P_A_train, P_A_test
    else:
        raise ValueError('Unsupported dataset type.')
    
def create_train_test_datasets(xs, ys, optimal_choice, P_A, dataset_constructor, testing_prop=0.5):
    num_trials = int(xs.shape[1])
    num_test_trials = int(math.ceil(float(num_trials) * testing_prop))
    num_train_trials = int(num_trials - num_test_trials)
    
    assert num_train_trials > 0 and num_test_trials > 0, "Invalid train/test split"
    
    idx = np.random.permutation(num_trials)
    dataset_train = dataset_constructor(xs[:, idx[:num_train_trials]], ys[:, idx[:num_train_trials]])
    dataset_test = dataset_constructor(xs[:, idx[num_train_trials:]], ys[:, idx[num_train_trials:]])
    
    optimal_choice_train = optimal_choice[:, idx[:num_train_trials]]
    optimal_choice_test = optimal_choice[:, idx[num_train_trials:]]
    P_A_train = P_A[:, idx[:num_train_trials]]
    P_A_test = P_A[:, idx[num_train_trials:]]
    
    return dataset_train, dataset_test, optimal_choice_train, optimal_choice_test, P_A_train, P_A_test

def dataset_size(xs, ys):
    """Calculate the size of the dataset in terms of number of samples."""
    return xs.shape[0] * xs.shape[1], ys.shape[0] * ys.shape[1]  # (number of timesteps * number of episodes)

def train_model(args_dict,
                dataset_train, 
                latent_size, 
                update_mlp_shape, 
                choice_mlp_shape, 
                beta_scale, 
                penalty_scale,
                n_step_max,
                n_steps_per_call):
    
    x, y = next(dataset_train)

    wandb.init(project="CogModRNN", entity="yolandaz",config={
        "latent_size": latent_size,
        "update_mlp_shape": update_mlp_shape,
        "choice_mlp_shape": choice_mlp_shape,
        "beta_scale": beta_scale,
        "penalty_scale": penalty_scale,
    })

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
        n_steps_max=n_step_max,
        n_steps_per_call=n_steps_per_call,
        return_all_losses=True,
        penalty_scale=penalty_scale
    )

    for step, loss in enumerate(losses):
        wandb.log({"loss": loss, "step": step})

    checkpoint = {
        'args_dict': args_dict,
        'disrnn_params': disrnn_params
    }

    filename = os.path.join(checkpoint_dir, f'disrnn_params_ls_{latent_size}_umlp_{update_mlp_shape}_cmlp_{choice_mlp_shape}_penalty_{penalty_scale}_beta_{beta_scale}_lr_1e-3.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(checkpoint, file)

    print(f'Saved disrnn_params to {filename}')
    
    return disrnn_params

def main(args_dict,
         seed,
         validation_proportion, 
         latent_size, 
         update_mlp_shape, 
         choice_mlp_shape, 
         beta_scale, 
         penalty_scale,
         n_step_max,
         n_steps_per_call):
    
    gpu_devices = jax.devices("gpu")

    if gpu_devices:
        print(f"JAX is using GPU: {gpu_devices}")
    else:
        print("No GPU found, JAX is using CPU.")

    np.random.seed(seed)

    # Preprocess Data
    dataset_type = 'RealWorldKimmelfMRIDataset'
    dataset_path = "dataset/tensor_for_dRNN_desc-syn_nSubs-2000_nSessions-1_nBlocks-1_nTrialsPerBlock-100_b-0.3_NaN_30_0.93_0.45_NaN_NaN_withOptimalChoice_20240718_fast.mat"
    dataset_train, dataset_test, *_ = preprocess_data(dataset_type, dataset_path, validation_proportion)

    # Train model
    disrnn_params = train_model(args_dict,
                                dataset_train, 
                                latent_size, 
                                update_mlp_shape, 
                                choice_mlp_shape, 
                                beta_scale, 
                                penalty_scale, 
                                n_step_max,
                                n_steps_per_call)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", nargs=1, type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--validation_proportion", nargs=1, type=int, default=0.1, help="The percentage for validation dataset.")
    parser.add_argument("--latent_size", type=int, default=16, help="Number of latent units in the model")
    parser.add_argument("--update_mlp_shape", nargs="+", type=int, default=[8, 8], help="Number of hidden units in each of the two layers of the update MLP")
    parser.add_argument("--choice_mlp_shape", nargs="+", type=int, default=[8, 8], help="Number of hidden units in each of the two layers of the choice MLP")
    parser.add_argument("--beta_scale", type=float, required=True, help="Value for the beta scaling parameter")
    parser.add_argument("--penalty_scale", type=float, required=True, help="Value for the penalty scaling parameter")
    parser.add_argument("--n_step_max", type=int, default=10000, help="The maximum number of iterations to run, even if convergence is not reached")
    parser.add_argument("--n_steps_per_call", type=float, default=500, help="The number of steps to give to train_model")

    args = parser.parse_args()

    args_dict = {
        'seed': args.seed,
        'validation_proportion': args.validation_proportion,
        'latent_size': args.latent_size,
        'update_mlp_shape': args.update_mlp_shape,
        'choice_mlp_shape': args.choice_mlp_shape,
        'beta_scale': args.beta_scale,
        'penalty_scale': args.penalty_scale,
        'n_step_max': args.n_step_max,
        'n_steps_per_call': args.n_steps_per_call
    }

    main(args_dict,
         args.seed,
         args.validation_proportion, 
         args.latent_size, 
         args.update_mlp_shape, 
         args.choice_mlp_shape, 
         args.beta_scale, 
         args.penalty_scale,
         args.n_step_max,
         args.n_steps_per_call)
