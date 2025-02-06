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

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from CogModelingRNNsTutorial.CogModelingRNNsTutorial import bandits, disrnn, hybrnn, plotting, rat_data, rnn_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from absl import app
import optax
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import rnn_utils

warnings.filterwarnings("ignore")

def load_data(fname, data_dir='./'):
    """Load data for one human subject from a MATLAB file and organize into a dictionary.
    
    Parameters:
    fname : str
        File name of the MATLAB data.
    data_dir : str, optional
        Directory of the data file (default is './').
    
    Returns:
    dict
        Dictionary containing feature, label, and state information.
    """
    # Load the MATLAB file
    mat_contents = sio.loadmat(os.path.join(data_dir, fname))
    data = mat_contents['tensor']
    var_names = [x.item() for x in mat_contents['vars_in_tensor'].ravel()]
    vars_for_state = [x.item() for x in mat_contents['vars_for_state'].ravel()]
    
    # Create a dictionary to store the data
    data_dict = {}
    
    # Extract data for each variable name
    for var_name in var_names:
        var_idx = (np.array(var_names) == var_name).flatten()
        var_data = data[:, :, var_idx]
        var_data = np.transpose(var_data, (1, 0, 2))  # Transpose to match the desired format
        data_dict[var_name] = var_data

    # Special processing for zs, ys, xs
    if 'state' in data_dict:
        zs = np.clip(data_dict['state'] - 1, a_min=-1, a_max=None)
        data_dict['state'] = zs
        # Need to make this an argument
        data_dict['state_onehot'] = zs_to_onehot(zs)[..., :8]
    if 'bitResponseA' in data_dict:
        ys = data_dict['bitResponseA']
        data_dict['ys'] = ys
    if 'bitCorr_prev' in data_dict and 'bitResponseA_prev' in data_dict:
        bitCorr_prev = data_dict['bitCorr_prev']
        bitResponseA_prev = data_dict['bitResponseA_prev']
        xs = np.concatenate([bitCorr_prev, bitResponseA_prev], axis=2)
        data_dict['xs'] = xs


    data_dict['inputs'] = np.concatenate([data_dict['xs'], data_dict['state_onehot']], axis=-1)

    # Check assertions
    assert data_dict['state'].shape[-1] == data_dict['ys'].shape[-1] == 1, "Mismatch in dimensions"
    assert data_dict['state'].ndim == data_dict['ys'].ndim == data_dict['xs'].ndim == 3, "Data should be 3-dimensional"
    assert data_dict['state'].max() < 16, "State values should be less than 16"
    
    # Add filename and vars_for_state to the dictionary
    data_dict['fname'] = fname
    data_dict['vars_for_state'] = vars_for_state

    return data_dict


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
        
        data_dict = load_data(LOCAL_PATH_TO_FILE)
        dataset_train, dataset_test, train_test_split_variables = create_train_test_datasets(data_dict, rnn_utils.DatasetRNN, testing_set_proportion)

        train_size, test_size = dataset_size(dataset_train._xs, dataset_test._xs)
        print(f'Training dataset size: {train_size} samples')
        print(f'Testing dataset size: {test_size} samples')

        return dataset_train, dataset_test, train_test_split_variables
    else:
        raise ValueError('Unsupported dataset type.')

def create_train_test_datasets(data_dict, dataset_constructor, testing_prop=0.5):
    inputs = data_dict['inputs']
    ys = data_dict['ys']
    num_trials = int(inputs.shape[1])
    num_test_trials = int(math.ceil(float(num_trials) * testing_prop))
    num_train_trials = int(num_trials - num_test_trials)
    
    assert num_train_trials > 0 and num_test_trials > 0, "Invalid train/test split"
    
    idx = np.random.permutation(num_trials)
    dataset_train = dataset_constructor(inputs[:, idx[:num_train_trials]], ys[:, idx[:num_train_trials]])
    dataset_test = dataset_constructor(inputs[:, idx[num_train_trials:]], ys[:, idx[num_train_trials:]])
    
    # Split other variables in data_dict
    train_test_split = {}
    for key in data_dict:
        if key not in ['xs', 'ys', 'zs', 'input', 'state_onehot', 'fname', 'vars_for_state']:
            var_data = data_dict[key]
            train_test_split[f'{key}_train'] = var_data[:, idx[:num_train_trials]]
            train_test_split[f'{key}_test'] = var_data[:, idx[num_train_trials:]]
    
    return dataset_train, dataset_test, train_test_split


def dataset_size(xs, ys):
    """Calculate the size of the dataset in terms of number of samples."""
    return xs.shape[1], ys.shape[1] # 1800


def train_model(args_dict,
                dataset_train, 
                dataset_test,
                latent_size, 
                update_mlp_shape, 
                choice_mlp_shape, 
                beta_scale, 
                penalty_scale,
                n_training_steps,
                n_warmup_steps,
                n_steps_per_call,
                saved_checkpoint_pth=None):

    if saved_checkpoint_pth:
        if os.path.exists(saved_checkpoint_pth):
            print(f"Loading checkpoint from {saved_checkpoint_pth}...")
            with open(saved_checkpoint_pth, 'rb') as f:
                checkpoint = pickle.load(f)
            args_dict = checkpoint['args_dict']
            disrnn_params = checkpoint['disrnn_params']
            print("Checkpoint loaded successfully.")
        else: 
            print("Checkpoint Invalid. Exiting...")
            exit()
    else:
        disrnn_params = None

    x, y = next(dataset_train)
    
    # -------------------------------------------------------------
    # Give your run a more meaningful name:
    # -------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (f"ls_{latent_size}_umlp_{'-'.join(map(str, update_mlp_shape))}"
                f"_cmlp_{'-'.join(map(str, choice_mlp_shape))}"
                f"_beta_{beta_scale}_penalty_{penalty_scale}_{timestamp}")
    
    # Initialize wandb with that name:
    wandb.init(
        project="CogModRNN",
        entity="yolandaz",
        config={
            "latent_size": latent_size,
            "update_mlp_shape": update_mlp_shape,
            "choice_mlp_shape": choice_mlp_shape,
            "beta_scale": beta_scale,
            "penalty_scale": penalty_scale,
        },
        name=run_name,  # <--- more meaningful run name
    )
    
    # Create directories for outputs
    plot_dir = os.path.join('plots_2025', run_name)
    checkpoint_dir = os.path.join('checkpoints_0128', run_name)
    loss_dir = os.path.join('loss', run_name)
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
            beta_scale=beta_scale
        )
        return model

    opt = optax.adam(learning_rate=1e-3)

    # -------------------------------------------------------------
    # 1) Optional warmup (penalty_scale=0)
    # -------------------------------------------------------------
    if saved_checkpoint_pth is None:
        print("Warmup starts!")
        disrnn_params, losses, _ = rnn_utils.train_network(
            make_disrnn,
            training_dataset=dataset_train,
            validation_dataset=dataset_test,
            loss="penalized_categorical",
            params=disrnn_params,
            opt_state=None,
            opt=opt,
            penalty_scale=0,
            n_steps=n_warmup_steps,
            do_plot=False,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=100,
            args_dict=args_dict
        )
    
        # -------------------------------------------------------------
        # If *not* doing real-time logging in train_network, you could
        # log warmup losses here. But we *are* logging real-time now,
        # so we can comment this out to avoid duplication:
        # -------------------------------------------------------------
        """
        num_steps_logged = len(losses.get("training_total_loss", []))
        for step_idx in range(num_steps_logged):
            wandb.log({
                "warmup_categorical_loss": losses["training_cat_loss"][step_idx],
                "warmup_penalty_term": losses["training_penalty"][step_idx],
                "warmup_total_loss": losses["training_total_loss"][step_idx],
                "warmup_step": step_idx
            })
        """

    # -------------------------------------------------------------
    # 2) Main training (penalty_scale=user specified)
    # -------------------------------------------------------------
    disrnn_params, losses, _ = rnn_utils.train_network(
        make_disrnn,
        training_dataset=dataset_train,
        validation_dataset=dataset_test,
        loss="penalized_categorical",
        params=disrnn_params,
        opt_state=None,
        opt=opt,
        penalty_scale=penalty_scale,
        n_steps=n_training_steps,
        do_plot=True,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=100,
        args_dict=args_dict
    )

    # -------------------------------------------------------------
    # Real-time logs are *already* being done inside train_network.
    # So you do NOT need the post-hoc loop below. You can comment it
    # out if you want to avoid duplication:
    # -------------------------------------------------------------
    """
    num_steps_logged = len(losses["training_total_loss"])
    for step_idx in range(num_steps_logged):
        wandb.log({
            "categorical_loss": losses["training_cat_loss"][step_idx],
            "penalty_term":     losses["training_penalty"][step_idx],
            "loss":             losses["training_total_loss"][step_idx],
            "step": step_idx
        })
    """

    # -------------------------------------------------------------
    # Save final checkpoint
    # -------------------------------------------------------------
    final_checkpoint = {
        'args_dict': args_dict,
        'disrnn_params': disrnn_params
    }
    final_filename = os.path.join(
        checkpoint_dir, 
        f'final_disrnn_params_ls_{latent_size}_umlp_{"-".join(map(str, update_mlp_shape))}'
        f'_cmlp_{"-".join(map(str, choice_mlp_shape))}_penalty_{penalty_scale}'
        f'_beta_{beta_scale}_lr_1e-3.pkl'
    )
    with open(final_filename, 'wb') as f:
        pickle.dump(final_checkpoint, f)

    print(f'Saved disrnn_params to {final_filename}')


def main(args_dict,
         seed,
         validation_proportion, 
         latent_size, 
         update_mlp_shape, 
         choice_mlp_shape, 
         beta_scale, 
         penalty_scale,
         n_training_steps,
         n_warmup_steps,
         n_steps_per_call,
         saved_checkpoint_pth=None):

    gpu_devices = jax.devices("gpu")

    if gpu_devices:
        print(f"JAX is using GPU: {gpu_devices}")
    else:
        print("No GPU found, JAX is using CPU.")

    np.random.seed(seed)

    # Preprocess Data
    dataset_type = 'RealWorldKimmelfMRIDataset'
    dataset_path = "dataset/tensor_for_dRNN_desc-syn_nSubs-2000_nSessions-2_nBlocks-7_nTrialsPerBlock-25_b-multiple_20250131.mat"
    dataset_train, dataset_test, *_ = preprocess_data(dataset_type, dataset_path, validation_proportion)

    # Train model
    train_model(args_dict,
                dataset_train, 
                dataset_test,
                latent_size, 
                update_mlp_shape, 
                choice_mlp_shape, 
                beta_scale, 
                penalty_scale, 
                n_training_steps,
                n_warmup_steps,
                n_steps_per_call,
                saved_checkpoint_pth=saved_checkpoint_pth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--validation_proportion", type=float, default=0.1, help="The percentage for validation dataset.")
    parser.add_argument("--latent_size", type=int, default=8, help="Number of latent units in the model")
    parser.add_argument("--update_mlp_shape", nargs="+", type=int, default=[8, 8, 8], help="Number of hidden units in each of the two layers of the update MLP")
    parser.add_argument("--choice_mlp_shape", nargs="+", type=int, default=[3, 3, 3], help="Number of hidden units in each of the two layers of the choice MLP")
    parser.add_argument("--beta_scale", type=float, required=True, help="Value for the beta scaling parameter")
    parser.add_argument("--penalty_scale", type=float, required=True, help="Value for the penalty scaling parameter")
    parser.add_argument("--n_training_steps", type=int, default=1, help="The maximum number of iterations to run, even if convergence is not reached")
    parser.add_argument("--n_warmup_steps", type=int, default=50000, help="The maximum number of iterations to run, even if convergence is not reached")
    parser.add_argument("--n_steps_per_call", type=int, default=500, help="The number of steps to give to train_model")
    parser.add_argument("--saved_checkpoint_pth", type=str, default=None, help="Path to the checkpoint for additional training")

    args = parser.parse_args()

    args_dict = {
        'seed': args.seed,
        'validation_proportion': args.validation_proportion,
        'latent_size': args.latent_size,
        'update_mlp_shape': args.update_mlp_shape,
        'choice_mlp_shape': args.choice_mlp_shape,
        'beta_scale': args.beta_scale,
        'penalty_scale': args.penalty_scale,
        'n_training_steps': args.n_training_steps,
        'n_warmup_steps': args.n_warmup_steps,
        'n_steps_per_call': args.n_steps_per_call,
    }

    main(args_dict,
         args.seed,
         args.validation_proportion, 
         args.latent_size, 
         args.update_mlp_shape, 
         args.choice_mlp_shape, 
         args.beta_scale, 
         args.penalty_scale,
         args.n_training_steps,
         args.n_warmup_steps,
         args.n_steps_per_call,
         saved_checkpoint_pth=args.saved_checkpoint_pth)
