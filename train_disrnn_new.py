#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated training script for DisRNN – adapted to the 2025
`disentangled_rnns.library.disrnn` API.

Data loading, argument parsing, logging and checkpointing are unchanged.
Only the model‑building utilities have been rewritten to use the new
`DisRnnConfig` + `HkDisentangledRNN` pipeline.
"""

import argparse
import copy
import math
import os
import pickle
import sys
import warnings
from datetime import datetime

# ---------------------------------------------------------------------
# Third‑party libs
# ---------------------------------------------------------------------
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
import scipy.io as sio
import wandb

# ---------------------------------------------------------------------
# Local project code
# ---------------------------------------------------------------------
from absl import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from disentangled_rnns_new.disentangled_rnns.library import disrnn        # 2025 version
from disentangled_rnns_new.disentangled_rnns.library import rnn_utils

warnings.filterwarnings("ignore")

# =====================================================================
# -----------------------------  DATA  --------------------------------
# =====================================================================

def load_data(fname, data_dir="./"):
    """Loads a single‐subject MATLAB file and packages variables."""
    mat = sio.loadmat(os.path.join(data_dir, fname))
    data      = mat["tensor"]
    var_names = [v.item() for v in mat["vars_in_tensor"].ravel()]
    vars_for_state = [v.item() for v in mat["vars_for_state"].ravel()]

    # Build dict of [T, E, 1] arrays
    ddict = {}
    for v in var_names:
        idx   = (np.array(var_names) == v).flatten()
        arr   = np.transpose(data[:, :, idx], (1, 0, 2))
        ddict[v] = arr

    # ---- special variables -------------------------------------------------
    if "state" in ddict:
        zs = np.clip(ddict["state"] - 1, a_min=-1, a_max=None)
        ddict["state"]        = zs
        ddict["state_onehot"] = zs_to_onehot(zs)
        print(zs_to_onehot(zs).shape)

    if "bitResponseA" in ddict:
        ddict["ys"] = ddict["bitResponseA"]

    if "bitCorr_prev" in ddict and "bitResponseA_prev" in ddict:
        ddict["xs"] = np.concatenate(
            [ddict["bitCorr_prev"], ddict["bitResponseA_prev"]], axis=2
        )

    ddict["inputs"] = np.concatenate(
        [ddict["xs"], ddict["state_onehot"]], axis=-1
    )

    # quick sanity checks
    assert ddict["state"].shape[-1] == ddict["ys"].shape[-1] == 1
    assert ddict["state"].ndim == ddict["ys"].ndim == ddict["xs"].ndim == 3
    assert ddict["state"].max() < 16

    ddict["fname"] = fname
    ddict["vars_for_state"] = vars_for_state
    return ddict


def to_onehot(labels, n):
    labels = np.asarray(labels, dtype=int)
    return np.eye(n)[labels]


def zs_to_onehot(zs):
    assert zs.shape[-1] == 1
    zs = zs[..., 0]
    minus1 = zs == -1
    oh = to_onehot(zs, 16)
    oh[minus1] = 0
    return oh


# --------------- train / test split utilities ------------------------

def create_train_test_datasets(data_dict,
                               dataset_ctor,
                               testing_prop=0.5):
    xs, ys = data_dict["inputs"], data_dict["ys"]
    n_trials = int(xs.shape[1])
    n_test   = int(math.ceil(n_trials * testing_prop))
    n_train  = n_trials - n_test
    assert n_train > 0 and n_test > 0

    idx = np.random.permutation(n_trials)
    train = dataset_ctor(
        xs[:, idx[:n_train]], ys[:, idx[:n_train]], y_type="categorical"
    )
    test  = dataset_ctor(
        xs[:, idx[n_train:]], ys[:, idx[n_train:]], y_type="categorical"
    )


    # also return a split copy of any extra fields
    split_vars = {}
    for k in data_dict:
        if k in {"xs", "ys", "state_onehot", "inputs",
                 "fname", "vars_for_state"}:
            continue
        arr = data_dict[k]
        split_vars[f"{k}_train"] = arr[:, idx[:n_train]]
        split_vars[f"{k}_test"]  = arr[:, idx[n_train:]]
    return train, test, split_vars


def preprocess_data(dataset_type, path, test_prop):
    if dataset_type != "RealWorldKimmelfMRIDataset":
        raise ValueError("Unsupported dataset type.")

    if not os.path.exists(path):
        raise ValueError(f"File {path} not found.")

    d = load_data(path)
    train, test, meta = create_train_test_datasets(
        d, rnn_utils.DatasetRNN, test_prop
    )
    print(f"Training dataset size: {train._xs.shape[1]}")
    print(f"Testing  dataset size: {test._xs.shape[1]}")
    return train, test, meta


# =====================================================================
# ---------------------------  MODEL BUILD ----------------------------
# =====================================================================

def build_disrnn_config(
    obs_size, output_size, latent_size,
    update_mlp_shape, choice_mlp_shape,
    *,  # ⬅ positional args end here
    noiseless: bool,
    latent_penalty: float,
    update_net_obs_penalty: float,
    update_net_latent_penalty: float,
    choice_net_latent_penalty: float,
    l2_scale: float,
):
    """Factory mapping CLI flags → DisRnnConfig."""
    return disrnn.DisRnnConfig(
        # sizes
        obs_size   = obs_size,
        output_size= output_size,
        latent_size= latent_size,
        # Update-/Choice-net shapes
        update_net_n_units_per_layer = int(update_mlp_shape[0]),
        update_net_n_layers          = len(update_mlp_shape),
        choice_net_n_units_per_layer = int(choice_mlp_shape[0]),
        choice_net_n_layers          = len(choice_mlp_shape),
        activation = "relu",
        noiseless_mode = noiseless,
        # ⬇ individual penalties
        latent_penalty            = latent_penalty,
        update_net_obs_penalty    = update_net_obs_penalty,
        update_net_latent_penalty = update_net_latent_penalty,
        choice_net_latent_penalty = choice_net_latent_penalty,
        l2_scale = l2_scale,
    )


# =====================================================================
# ----------------------------  TRAINING ------------------------------
# =====================================================================
def train_model(
    args_dict,
    dataset_train,
    dataset_test,
    latent_size,
    update_mlp_shape,
    choice_mlp_shape,
    # All penalties
    latent_penalty,
    update_net_obs_penalty,
    update_net_latent_penalty,
    choice_net_latent_penalty,
    l2_scale,
    loss_param,
    n_training_steps,
    n_warmup_steps,
    n_steps_per_call,   # kept for CLI parity (not used)
    saved_checkpoint_pth=None,
    checkpoint_interval=50,  # Save checkpoint every `checkpoint_interval` steps
):

    # -------------------------------------------------
    # (Optional) resume from checkpoint
    # -------------------------------------------------
    if saved_checkpoint_pth and os.path.exists(saved_checkpoint_pth):
        print(f"Loading checkpoint from {saved_checkpoint_pth} …")
        with open(saved_checkpoint_pth, "rb") as f:
            ckpt = pickle.load(f)
        args_dict     = ckpt["args_dict"]
        disrnn_params = ckpt["params"]
        opt_state     = ckpt["opt_state"]
        step          = ckpt["step"]
        print("Checkpoint loaded.")
    else:
        disrnn_params = None
        opt_state = None
        step = 0  # Start from the first step if no checkpoint exists

    # one example batch for shape info
    x, y = next(dataset_train)

    # -------------------------------------------------
    # Run‑name / wandb initialisation
    # -------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"ls_{latent_size}"                                   # latent-state size
        f"_umlp_{'-'.join(map(str, update_mlp_shape))}"       # Update-MLP shape
        f"_cmlp_{'-'.join(map(str, choice_mlp_shape))}"       # Choice-MLP shape
        f"_lparam_{loss_param}"
        f"_lp_{latent_penalty}"                               # latent_penalty
        f"_uobs_{update_net_obs_penalty}"                     # update_net_obs_penalty
        f"_ulat_{update_net_latent_penalty}"                  # update_net_latent_penalty
        f"_clat_{choice_net_latent_penalty}"                  # choice_net_latent_penalty
        f"_l2_{l2_scale}"                                     # L2 weight-decay
        f"_{ts}"                                              # timestamp
    )
    wandb.init(
        project="CogModRNN",
        entity="yolandaz",
        name=run_name,
        config={
            **args_dict,
            "run_name": run_name,
        },
    )

    # dirs
    plot_dir = os.path.join("plots_2025", run_name)
    ckpt_dir = os.path.join("checkpoints_06", run_name)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # -------------------------------------------------
    # Model builders
    # -------------------------------------------------
    def make_disrnn():
        cfg = build_disrnn_config(
            obs_size = x.shape[2],
            output_size = 2,
            latent_size = latent_size,
            update_mlp_shape = update_mlp_shape,
            choice_mlp_shape = choice_mlp_shape,
            noiseless = False,
            latent_penalty            = latent_penalty,
            update_net_obs_penalty    = update_net_obs_penalty,
            update_net_latent_penalty = update_net_latent_penalty,
            choice_net_latent_penalty = choice_net_latent_penalty,
            l2_scale = l2_scale,
        )
        return disrnn.HkDisentangledRNN(cfg)

    def make_disrnn_warmup():
        cfg = build_disrnn_config(
            obs_size = x.shape[2],
            output_size = 2,
            latent_size = latent_size,
            update_mlp_shape = update_mlp_shape,
            choice_mlp_shape = choice_mlp_shape,
            noiseless = True,   # deterministic
            latent_penalty            = 0.0,
            update_net_obs_penalty    = 0.0,
            update_net_latent_penalty = 0.0,
            choice_net_latent_penalty = 0.0,
            l2_scale = 0.0,      # ← no weight-decay in warm-up
        )
        return disrnn.HkDisentangledRNN(cfg)

    opt = optax.adam(1e-3)

    # -------------------------------------------------
    # 1) Warm‑up (no penalties, no noise)
    # -------------------------------------------------s
    if n_warmup_steps != 0 and not saved_checkpoint_pth:
        print("Warm‑up phase …")
        # 1) Warm-up -----------------------------------------
        disrnn_params, opt_state, _ = rnn_utils.train_network(
            make_disrnn_warmup,
            training_dataset=dataset_train,
            validation_dataset=dataset_test,
            loss="penalized_categorical",
            params=disrnn_params,
            opt_state=None,
            opt=opt,
            loss_param=0.0,           
            n_steps=n_warmup_steps,
            do_plot=False,
            log_losses_every=100,      # exact analogue of old checkpoint_interval
            checkpoint_dir="./checkpoints_new_data",  # Specify the directory to save checkpoints
            checkpoint_interval=100,   # Save checkpoint every 100 steps
            run_name=run_name,
            args_dict=args_dict
        )

    # -------------------------------------------------
    # 2) Main training
    # -------------------------------------------------
    print("Main training …")
    disrnn_params, opt_state, losses = rnn_utils.train_network(
        make_disrnn,
        training_dataset=dataset_train,
        validation_dataset=dataset_test,
        loss="penalized_categorical",
        params=disrnn_params,
        opt_state=opt_state,
        opt=opt,
        loss_param=loss_param,  # ← your CLI arg
        n_steps=n_training_steps,
        do_plot=True,
        log_losses_every=100,
        checkpoint_dir="./checkpoints",  # Specify where to save checkpoints
        checkpoint_interval=100,
        run_name=run_name,
        args_dict=args_dict
    )



# =====================================================================
# -----------------------------  MAIN  --------------------------------
# =====================================================================

def main(args):
    # Devices
    devs = jax.devices("gpu")
    print(f"JAX devices: {devs if devs else 'CPU'}")
    np_rng = np.random.default_rng(args.seed)

    # ---------- data ----------
    dataset_type = "RealWorldKimmelfMRIDataset"
    dataset_path = "/home/rsw0/Desktop/yolanda/CogModRNN/CogModRNN/dataset/tensor_for_dRNN_desc-syn_nSubs-2000_nSessions-4_nBlocks-7_nTrialsPerBlock-25_b-multiple_20250613.mat"
    train_ds, test_ds, _ = preprocess_data(
        dataset_type, dataset_path, args.validation_proportion
    )

    # ---------- train ----------
    train_model(
        args_dict       = vars(args),
        dataset_train   = train_ds,
        dataset_test    = test_ds,
        latent_size     = args.latent_size,
        update_mlp_shape= args.update_mlp_shape,
        choice_mlp_shape= args.choice_mlp_shape,
        latent_penalty            = args.latent_penalty,
        update_net_obs_penalty    = args.update_net_obs_penalty,
        update_net_latent_penalty = args.update_net_latent_penalty,
        choice_net_latent_penalty = args.choice_net_latent_penalty,
        l2_scale                  = args.l2_scale,
        loss_param               = args.loss_param,
        n_training_steps= args.n_training_steps,
        n_warmup_steps  = args.n_warmup_steps,
        n_steps_per_call= args.n_steps_per_call,
        saved_checkpoint_pth = args.saved_checkpoint_pth,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_proportion", type=float, default=0.1)
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--update_mlp_shape",
                        nargs="+", type=int, default=[12, 12, 12])
    parser.add_argument("--choice_mlp_shape",
                        nargs="+", type=int, default=[12, 12, 12])
    parser.add_argument("--n_training_steps", type=int, default=50000)
    parser.add_argument("--n_warmup_steps", type=int, default=1000)
    parser.add_argument("--n_steps_per_call", type=int, default=500)
    parser.add_argument("--saved_checkpoint_pth", type=str, default=None)

    parser.add_argument("--latent_penalty",            type=float, default=1e-4)
    parser.add_argument("--update_net_obs_penalty",    type=float, default=2e-3)
    parser.add_argument("--update_net_latent_penalty", type=float, default=2e-3)
    parser.add_argument("--choice_net_latent_penalty", type=float, default=1e-4)
    parser.add_argument("--l2_scale",                  type=float, default=1e-5)
    parser.add_argument("--loss_param",                type=float, default=1.00)

    args = parser.parse_args()

    # Store full CLI in the args dict (useful for provenance)
    args.command_line = " ".join(sys.argv)

    main(args)
