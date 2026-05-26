"""
FPTT (Forward Propagation Through Time) training for the custom async RNN.

Based on: "Training RNNs via Forward Propagation Through Time"
Reference implementation: https://github.com/anilkagak2/FPTT

Core idea
---------
The sequence of length T is split into PARTS chunks.  For chunk p (0-indexed):

    loss_p = beta_p * CE(output_p, y)  +  (1 - beta_p) * oracle_loss_p
    beta_p = (p+1) / PARTS

    oracle_loss_p = -mean( oracle_prob_p · log_softmax(logit_p) )

    oracle_prob_p:
        p < PARTS-1  →  estimate_class_distribution[y, p]   (previous-epoch estimate)
        p == PARTS-1 →  one_hot(y)                          (true label at the end)

A consensus regularizer is added to every chunk loss:

    regularizer = (rho - 1) * sum(param · lm)
                + lambda * 0.5 * alpha * sum(||param - sm||²)

where (sm, lm) are per-parameter running statistics updated *after* each
optimizer step via post_optimizer_updates:

    lm  +=  -alpha * (param - sm)
    sm   =  (1-beta)*sm + beta*param - (beta/alpha)*lm

At the start of each epoch, sm is reset to the current param values
(reset_named_params), so the regularizer pulls params toward their
epoch-start values, preventing large intra-epoch drifts.

Forward-pass equations (multi-layer RNN):

  For each timestep t, events propagate through L hidden layers:

    Layer 0 (input → hidden_0):
        inner_0 = x_t * W_ih[t]  +  z_prev_0  -  o_prev_0  +  o_prev_0 @ W_hh[0]  +  bias[0]

    Layer l (hidden_{l-1} → hidden_l),  l = 1..L-1:
        inner_l = o_{l-1} @ W_ll[l-1]  +  z_prev_l  -  o_prev_l  +  o_prev_l @ W_hh[l]  +  bias[l]

    For each layer:
        z_l     = tanh(inner_l)  if use_tanh else inner_l
        o_l     = ReLU(z_l)  (then sync_rate + firing_nb masks)

    logit += o_{L-1} @ W_out   (only last hidden layer contributes)
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize PyTorch CUDA before JAX touches the GPU
_CUDA_AVAILABLE = torch.cuda.is_available()

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from async_RNN_pytorch_gradient_check import (  # noqa: E402
    TorchRuleRNN,
    init_feedforward_weights,
    init_recurrent_weight,
    keep_top_k_batch_torch,
)
from dataset_helpers.mnist_helper import mnist_loader_manual  # noqa: E402


# ---------------------------------------------------------------------------
#region Data helpers
# ---------------------------------------------------------------------------

def load_neural_decoding_arrays(batch_size: int, data_dir: str,
                                filename: str = "indy_20160622_01.mat",
                                window: int = 50, train_ratio: float = 0.5,
                                collapse_units: bool = True,
                                preserve_exact_times: bool = False):
    """
    Load primate-reaching neural-decoding data and return numpy arrays in the
    same (x, y) shape convention as the other loaders. Regression task: y is
    continuous 2-D velocity, NOT class labels.

    Returns x_train, y_train, x_val, y_val, x_test, y_test, n_input_neurons
    where:
        x shape: (N, max_events, 2)  — padded event format (idx, value); idx=-2 is padding
        y shape: (N, 2)              — float32 (vx, vy)
        n_input_neurons: 96 by default (collapse_units=True)
    """
    from dataset_helpers.primate_reaching_helper import torch_primate_reaching_loader

    (trainloader, _), (valloader, _), (testloader, _), _ = torch_primate_reaching_loader(
        batch_size=batch_size, shuffle=False, data_dir=data_dir,
        filename=filename, window=window, train_ratio=train_ratio,
        collapse_units=collapse_units, preserve_exact_times=preserve_exact_times,
        truncate=True,
    )

    def _collect(loader):
        all_x, all_y = [], []
        for batch in loader:
            data, labels = batch
            all_x.append(np.asarray(data, dtype=np.float32))
            all_y.append(np.asarray(labels, dtype=np.float32))
        return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)

    x_train, y_train = _collect(trainloader)
    x_val,   y_val   = _collect(valloader)
    x_test,  y_test  = _collect(testloader)

    # n_input_neurons: derive from the max idx seen (collapse_units=True → 96 for indy*).
    n_input_neurons = int(max(x_train[..., 0].max(),
                              x_val[..., 0].max(),
                              x_test[..., 0].max()) + 1)
    return x_train, y_train, x_val, y_val, x_test, y_test, n_input_neurons


def load_shd_arrays(batch_size: int, data_dir: str):
    """
    Load SHD dataset and convert to numpy arrays in event format (neuron_idx, value).

    Returns x_train, y_train, x_val, y_val, x_test, y_test, n_input_neurons
    where x shape: (N, T, 2) — same event format as MNIST loader.
    SHD has 700 input neurons and 20 classes.
    """
    from dataset_helpers.shd_helper import torch_SHD_loader

    (trainloader, _), (valloader, _), (testloader, _), max_data_length = torch_SHD_loader(
        batch_size=batch_size, shuffle=True, data_dir=data_dir,
    )

    def _collect(loader):
        all_x, all_y = [], []
        for batch in loader:
            data, labels = batch
            # data is jnp array (B, T, 2), labels is jnp array (B,)
            all_x.append(np.asarray(data, dtype=np.float32))
            all_y.append(np.asarray(labels, dtype=np.int64))
        return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)

    x_train, y_train = _collect(trainloader)
    x_val, y_val = _collect(valloader)
    x_test, y_test = _collect(testloader)

    n_input_neurons = 700
    return x_train, y_train, x_val, y_val, x_test, y_test, n_input_neurons


def load_mnist_arrays(batch_size: int, data_dir: str, dataset: str = "mnist"):
    """
    Load preprocessed MNIST/S-MNIST event arrays from the dataloader.

    Returns x_train, y_train, x_val, y_val, x_test, y_test where:
        x shape: (N, T, 2) — each timestep is (neuron_idx, value)
        MNIST:  T=351, neuron_idx ∈ [0..783], padded with (-2, -2)
        S-MNIST: T=784, neuron_idx always 0, value = (pixel/255 - 0.1307) / 0.3081

    Also returns n_input_neurons: 784 for MNIST, 1 for S-MNIST.
    """
    sequential = (dataset == "smnist")
    (train_loader, _), (val_loader, _), (test_loader, _), _ = mnist_loader_manual(
        batch_size=batch_size,
        shuffle=True,
        preprocess=True,
        CNN_preprocess=False,
        downsample=False,
        sequential=sequential,
        permuted=False,
        data_dir=data_dir,
    )
    # x shape: (N, T, 2) — (neuron_idx, value) per timestep
    x_train = np.asarray(train_loader.X[train_loader.indices], dtype=np.float32)
    y_train = np.asarray(train_loader.Y[train_loader.indices], dtype=np.int64)
    x_val   = np.asarray(val_loader.X[val_loader.indices],    dtype=np.float32)
    y_val   = np.asarray(val_loader.Y[val_loader.indices],    dtype=np.int64)
    x_test  = np.asarray(test_loader.X[test_loader.indices],  dtype=np.float32)
    y_test  = np.asarray(test_loader.Y[test_loader.indices],  dtype=np.int64)

    n_input_neurons = 1 if sequential else 784
    return x_train, y_train, x_val, y_val, x_test, y_test, n_input_neurons


# ---------------------------------------------------------------------------
#region FPTT parameter-state helpers  (direct port of the reference implementation)
# ---------------------------------------------------------------------------

def get_stats_named_params(model):
    """
    For each trainable parameter create (param, sm, lm) where:
        sm  = shadow copy (running average, initialised to param)
        lm  = momentum term (initialised to zero)
    """
    named_params = {}
    for name, param in model.named_parameters():
        sm = param.detach().clone()
        lm = torch.zeros_like(param.detach())
        named_params[name] = (param, sm, lm)
    return named_params


def post_optimizer_updates(named_params, alpha, beta):
    """
    Called after each optimizer.step().
        lm  +=  -alpha * (param - sm)
        sm   =  (1-beta)*sm + beta*param - (beta/alpha)*lm
    """
    for name in named_params:
        param, sm, lm = named_params[name]
        lm.data.add_(-alpha * (param.data - sm.data))
        sm.data.mul_(1.0 - beta)
        sm.data.add_(beta * param.data - (beta / alpha) * lm.data)


def reset_named_params(named_params):
    """Reset sm = param, lm = 0 at the start of each epoch."""
    for name in named_params:
        param, sm, lm = named_params[name]
        sm.data.copy_(param.data)
        lm.data.zero_()


def get_regularizer(named_params, alpha, lmbda, rho=0.0):
    """regularizer = (rho-1)*sum(param·lm) + lambda*0.5*alpha*sum(||param-sm||²)"""
    reg = torch.zeros([], device=next(iter(named_params.values()))[0].device)
    for name in named_params:
        param, sm, lm = named_params[name]
        reg = reg + (rho - 1.0) * torch.sum(param * lm)
        reg = reg + lmbda * 0.5 * alpha * torch.sum((param - sm) ** 2)
    return reg


# ---------------------------------------------------------------------------
#region PyTorch RNN model for FPTT  (multi-hidden-layer)
# ---------------------------------------------------------------------------

class FPTTRuleRNN(nn.Module):
    """
    Multi-layer async RNN with chunk-level forward for FPTT training.

    Uses the preprocessed event format from the dataloader:
        x_chunk: (B, chunk_len, 2) — each event is (neuron_idx, value)
        Padding events have neuron_idx < 0 and are skipped (drive = 0).

    Architecture (L hidden layers, hidden_sizes = [H0, H1, ..., H_{L-1}]):
        w_ih:    (n_input_neurons, H0)  — input weight per neuron (indexed by event idx)
        w_hh[l]: (Hl, Hl)              — recurrent weight for layer l
        w_ll[l]: (Hl, H_{l+1})        — inter-layer weight, l = 0..L-2
        w_out:   (H_{L-1}, C)          — last hidden → output
        bias[l]: (Hl,)                 — bias per layer

    Per timestep t with event (neuron_idx, value):
        Layer 0: inner = value * w_ih[neuron_idx] + z_prev_0 - o_prev_0 + o_prev_0 @ w_hh[0] + bias[0]
        Layer l: inner = o_{l-1} @ w_ll[l-1] + z_prev_l - o_prev_l + o_prev_l @ w_hh[l] + bias[l]
        z_l = tanh(inner) if use_tanh else inner
        o_l = topk(sync(relu(z_l)))
        logits += o_{L-1} @ w_out
    """
    def __init__(self, w_ih, w_hh_list, w_ll_list, w_out, bias_list,
                 sync_rate, firing_nb, use_tanh=False, no_reset=False,
                 vanilla=False):
        super().__init__()
        self.w_ih  = nn.Parameter(torch.tensor(np.asarray(w_ih), dtype=torch.float32))
        self.w_hh  = nn.ParameterList([
            nn.Parameter(torch.tensor(np.asarray(w), dtype=torch.float32))
            for w in w_hh_list
        ])
        self.w_ll  = nn.ParameterList([
            nn.Parameter(torch.tensor(np.asarray(w), dtype=torch.float32))
            for w in w_ll_list
        ])
        self.w_out = nn.Parameter(torch.tensor(np.asarray(w_out), dtype=torch.float32))
        self.bias  = nn.ParameterList([
            nn.Parameter(torch.tensor(np.asarray(b), dtype=torch.float32))
            for b in bias_list
        ])
        self.sync_rate = int(sync_rate)
        self.firing_nb = int(firing_nb)
        self.use_tanh  = bool(use_tanh)
        self.no_reset  = bool(no_reset)
        self.vanilla   = bool(vanilla)
        self.n_layers  = len(w_hh_list)

    def init_hidden(self, batch_size):
        """Returns list of (z, o) tuples, one per hidden layer."""
        device = self.w_ih.device
        hidden = []
        for l in range(self.n_layers):
            H = self.w_hh[l].shape[0]
            hidden.append((
                torch.zeros(batch_size, H, device=device),
                torch.zeros(batch_size, H, device=device),
            ))
        return hidden

    def forward_chunk(self, x_chunk, hidden, t_offset):
        """
        Run the multi-layer RNN over x_chunk.
            x_chunk: (B, chunk_len, 2) — each event is (neuron_idx, value)
        Returns accumulated logits (B, C) and updated hidden state.
        """
        batch_size, chunk_len, _ = x_chunk.shape
        C = self.w_out.shape[1]
        device = self.w_ih.device
        L = self.n_layers

        logits = torch.zeros(batch_size, C, device=device)

        for local_t in range(chunk_len):
            t = t_offset + local_t
            neuron_idx = x_chunk[:, local_t, 0].long()   # (B,)
            value      = x_chunk[:, local_t, 1]           # (B,)

            # Mask out padding events (neuron_idx < 0)
            valid = (neuron_idx >= 0).float()             # (B,)

            # Input projection: value * w_ih[neuron_idx] for each sample
            # Clamp idx to 0 for indexing safety (masked out anyway)
            safe_idx = neuron_idx.clamp(min=0)
            w_selected = self.w_ih[safe_idx]              # (B, H0)
            drive = value.unsqueeze(1) * w_selected * valid.unsqueeze(1)  # (B, H0)

            for l in range(L):
                z_prev, o_prev = hidden[l]

                if l > 0:
                    drive = o_prev_layer @ self.w_ll[l - 1]

                if self.vanilla:
                    # Vanilla RNN: h_t = act(drive + W_hh @ h_{t-1} + bias)
                    # use_tanh=True → tanh (standard vanilla RNN)
                    # use_tanh=False → relu (IRNN mode)
                    inner = (
                        drive
                        + o_prev @ self.w_hh[l]
                        + self.bias[l].unsqueeze(0)
                    )
                    h_t = torch.tanh(inner) if self.use_tanh else torch.relu(inner)
                    z_t = h_t
                    o_t = h_t
                else:
                    reset = 0.0 if self.no_reset else o_prev
                    inner = (
                        drive
                        + z_prev
                        - reset
                        + o_prev @ self.w_hh[l]
                        + self.bias[l].unsqueeze(0)
                    )
                    z_t = torch.tanh(inner) if self.use_tanh else inner
                    o_t = torch.relu(z_t)

                    sync_fire = 1.0 if ((t + 1) % self.sync_rate == 0) else 0.0
                    o_t = o_t * sync_fire
                    o_t = keep_top_k_batch_torch(o_t, self.firing_nb)

                hidden[l] = (z_t, o_t)
                o_prev_layer = o_t

            if not self.vanilla:
                # AED: accumulate logits from every timestep
                logits = logits + hidden[L - 1][1] @ self.w_out

        if self.vanilla:
            # Vanilla RNN: classify from last hidden state of the chunk only
            logits = hidden[L - 1][1] @ self.w_out

        return logits, hidden


class FPTTLstmRNN(nn.Module):
    """
    Standard LSTM baseline for FPTT comparison.
    API-compatible with FPTTRuleRNN: forward_chunk / init_hidden.

    Uses nn.LSTM under the hood. Input events (neuron_idx, value) are reduced
    to just the value channel — suitable for SMNIST where neuron_idx is always 0.
    """

    def __init__(self, input_size, hidden_size, n_classes, nlayers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=nlayers,
            batch_first=False,
            dropout=dropout if nlayers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def init_hidden(self, batch_size):
        """Returns [(h0, c0)] — single-element list wrapping LSTM state."""
        device = self.fc.weight.device
        h0 = torch.zeros(self.nlayers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.nlayers, batch_size, self.hidden_size, device=device)
        return [(h0, c0)]

    def forward_chunk(self, x_chunk, hidden, t_offset):
        """
        Args:
            x_chunk: (B, chunk_len, 2) — event format (neuron_idx, value)
            hidden:  [(h, c)] from init_hidden or previous chunk
            t_offset: ignored
        Returns:
            logits: (B, n_classes)
            hidden: [(h_new, c_new)]
        """
        values = x_chunk[:, :, 1]                           # (B, chunk_len)
        x_seq = values.unsqueeze(-1).permute(1, 0, 2)       # (chunk_len, B, 1)

        h_prev, c_prev = hidden[0]
        output, (h_new, c_new) = self.lstm(x_seq, (h_prev, c_prev))

        logits = self.fc(output[-1])                         # (B, n_classes)
        return logits, [(h_new, c_new)]


class FPTTVanillaRNN(nn.Module):
    """
    Minimal vanilla (Elman) RNN baseline for FPTT comparison.
    API-compatible with FPTTRuleRNN: forward_chunk / init_hidden.

    Uses nn.RNN (tanh nonlinearity) under the hood.
    """

    def __init__(self, input_size, hidden_size, n_classes, nlayers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=nlayers,
            batch_first=False,
            nonlinearity='tanh',
            dropout=dropout if nlayers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def init_hidden(self, batch_size):
        """Returns [(h0,)] — single-element list wrapping RNN hidden state."""
        device = self.fc.weight.device
        h0 = torch.zeros(self.nlayers, batch_size, self.hidden_size, device=device)
        return [(h0,)]

    def forward_chunk(self, x_chunk, hidden, t_offset):
        """
        Args:
            x_chunk: (B, chunk_len, 2) — event format (neuron_idx, value)
            hidden:  [(h,)] from init_hidden or previous chunk
            t_offset: ignored
        Returns:
            logits: (B, n_classes)
            hidden: [(h_new,)]
        """
        values = x_chunk[:, :, 1]                           # (B, chunk_len)
        x_seq = values.unsqueeze(-1).permute(1, 0, 2)       # (chunk_len, B, 1)

        h_prev = hidden[0][0]
        output, h_new = self.rnn(x_seq, h_prev)

        logits = self.fc(output[-1])                         # (B, n_classes)
        return logits, [(h_new,)]


class FPTTMinimalRNNAED(nn.Module):
    """
    MinimalRNN with AED (Async Event-Driven) inter-layer semantics, matching the
    reference implementation in async_RNN.py.

    Layer 0 (first hidden) — sparse-embedding input from raw events:
        z_t = tanh(value * W_phi[neuron_idx] + b_phi)
        u_t = sigmoid([h_prev, z_t] @ W_gate + b_gate)
        h_t = u_t * h_prev + (1 - u_t) * z_t
        o_t = top_k(ReLU(h_t))

    Layers 1..nlayers-1 — per-event integration from previous layer's events:
        For each non-zero event (idx_e, val_e) emitted by layer l-1 at this timestep:
            z   = tanh(val_e * W_phi_l[:, idx_e] + b_phi_l)
            u   = sigmoid([h_l, z] @ W_gate_l + b_gate_l)
            h_l = u * h_l + (1 - u) * z       # advances once PER event
            o_l = top_k(ReLU(h_l))            # fired into the next layer

    Output: logits += sum_k o_last^k @ W_out per timestep, equivalent to one logit
    add per event from the last hidden layer.
    """

    def __init__(self, n_input_neurons, hidden_size, n_classes,
                 sync_rate=1, firing_nb=10000, dropout=0.0, nlayers=1,
                 dense_output_firing=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.sync_rate   = sync_rate
        self.firing_nb   = firing_nb
        self.nlayers     = nlayers
        # If True, the LAST hidden layer's full dense ReLU(h) is fed to W_out
        # at every step (no top-k cap on the output projection). The per-event
        # top-k firing BETWEEN hidden layers is unaffected.
        self.dense_output_firing = dense_output_firing

        # Layer 0: AED input — embedding lookup by neuron_idx
        self.W_phi  = nn.Parameter(torch.empty(n_input_neurons, hidden_size))
        self.b_phi  = nn.Parameter(torch.zeros(hidden_size))
        self.W_gate = nn.Parameter(torch.empty(hidden_size * 2, hidden_size))
        self.b_gate = nn.Parameter(torch.zeros(hidden_size))

        # Layers 1..nlayers-1: dense input from previous layer's output
        self.W_phi_layers  = nn.ModuleList()
        self.W_gate_layers = nn.ModuleList()
        for _ in range(1, nlayers):
            phi_l  = nn.Linear(hidden_size, hidden_size, bias=True)
            gate_l = nn.Linear(hidden_size * 2, hidden_size, bias=True)
            nn.init.xavier_uniform_(phi_l.weight)
            nn.init.xavier_uniform_(gate_l.weight)
            self.W_phi_layers.append(phi_l)
            self.W_gate_layers.append(gate_l)

        # Output projection — from last layer only
        self.W_out = nn.Parameter(torch.empty(hidden_size, n_classes))

        self.drop = nn.Dropout(dropout) if dropout > 0 else None

        nn.init.xavier_uniform_(self.W_phi)
        nn.init.xavier_uniform_(self.W_gate)
        # logits are divided by n_real_events before the loss (see forward_chunk),
        # so the effective fan-in per step is hidden_size — xavier with fan_in=hidden_size.
        nn.init.normal_(self.W_out, mean=0.0, std=1.0 / hidden_size ** 0.5)

    def init_hidden(self, batch_size):
        device = self.W_phi.device
        return [(torch.zeros(batch_size, self.hidden_size, device=device),)
                for _ in range(self.nlayers)]

    def _event_step(self, l, idx, val, h_prev, valid_mask):
        """One MinimalRNN gate update at hidden layer l (l >= 1) for a single
        sparse event (idx, val). Used for inter-layer per-event integration:
        layer l's hidden state advances ONCE per incoming event from layer l-1.

        Args:
            l: layer index in [1, nlayers).
            idx: (B,) long — index of the firing neuron at layer l-1.
            val: (B,) float — its value.
            h_prev: (B, H).
            valid_mask: (B,) bool — True for real events (val != 0), False for masked slots.
        Returns:
            h_new: (B, H) — h_prev where !valid_mask, else updated h.
        """
        W_phi_l  = self.W_phi_layers[l - 1]   # nn.Linear(H, H)
        W_gate_l = self.W_gate_layers[l - 1]  # nn.Linear(2H, H)
        safe_idx = idx.clamp(min=0)
        valid_f  = valid_mask.float().unsqueeze(1)

        # Sparse embedding: phi = val * W_phi_l[:, idx] + b_phi_l
        # nn.Linear stores weight as (out, in); weight.T has rows indexed by `in`.
        phi = val.unsqueeze(1) * W_phi_l.weight.t()[safe_idx] + W_phi_l.bias.unsqueeze(0)
        phi = phi * valid_f
        z   = torch.tanh(phi)

        cat_hz = torch.cat([h_prev, z], dim=1)
        u      = torch.sigmoid(W_gate_l(cat_hz))
        h_new  = u * h_prev + (1.0 - u) * z
        return torch.where(valid_mask.unsqueeze(1), h_new, h_prev)

    def forward_chunk(self, x_chunk, hidden, t_offset, return_sparsity=False,
                      return_per_step=False):
        """
        Args:
            x_chunk: (B, chunk_len, 2) — event format (neuron_idx, value)
            hidden:  [(h,), ...]  — one tuple per layer
            t_offset: int — global timestep offset (for sync_rate)
            return_per_step: if True, return (B, chunk_len, C) per-timestep logits
                             instead of the single (B, C) accumulated logit.
            return_sparsity: if True, also return per-layer fired-event counts
        Returns:
            logits: (B, n_classes) — accumulated from last hidden layer's events
            hidden: updated [(h,), ...]
            layer_fired: (only if return_sparsity=True) list[float], per hidden layer —
                total fired events (post top-k) emitted per prediction, batch-averaged

        Per-event semantics (matches async_RNN.py): for nlayers >= 2, every event
        emitted by layer l-1 advances layer l's hidden state once, after which
        layer l fires its own events that may propagate to layer l+1.
        """
        B, chunk_len, _ = x_chunk.shape
        H = self.hidden_size
        C = self.W_out.shape[1]
        device = self.W_phi.device
        K = min(self.firing_nb, H)  # static inner-loop bound

        hs = [hidden[l][0] for l in range(self.nlayers)]   # list of (B, H)
        logits = torch.zeros(B, C, device=device)
        # Per-layer count of fired events (post top-k) emitted while processing one
        # input sequence, batch-averaged → total fired events per prediction per layer.
        layer_fired = [0.0] * self.nlayers
        n_real_events = 0  # count of non-padding input events for output normalisation
        step_logits = [] if return_per_step else None  # per-timestep logits (B, C)

        for local_t in range(chunk_len):
            t = t_offset + local_t
            neuron_idx = x_chunk[:, local_t, 0].long()
            value      = x_chunk[:, local_t, 1]

            valid    = (neuron_idx >= 0).float().unsqueeze(1)
            safe_idx = neuron_idx.clamp(min=0)
            n_real_events += (neuron_idx >= 0).float().mean().item()

            # --- Layer 0: AED sparse-embedding input from raw events ---
            phi_input = value.unsqueeze(1) * self.W_phi[safe_idx] + self.b_phi.unsqueeze(0)
            phi_input = phi_input * valid
            z_t       = torch.tanh(phi_input)
            cat_hz    = torch.cat([hs[0], z_t], dim=1)
            u_t       = torch.sigmoid(cat_hz @ self.W_gate + self.b_gate.unsqueeze(0))
            h_new     = u_t * hs[0] + (1.0 - u_t) * z_t
            hs[0]     = torch.where(valid.bool(), h_new, hs[0])

            sync_fire = 1.0 if ((t + 1) % self.sync_rate == 0) else 0.0

            # 1-layer: original behaviour (mathematically per-event already, since the
            # output matmul on the dense top-k vector equals Σ_e val_e · W_out[idx_e]).
            if self.nlayers == 1:
                o_t = torch.relu(hs[0]) * sync_fire
                if not self.dense_output_firing:
                    o_t = keep_top_k_batch_torch(o_t, self.firing_nb)
                if self.drop is not None and self.training:
                    o_t = self.drop(o_t)
                step_out = o_t @ self.W_out  # instantaneous projection from current h
                logits = logits + step_out
                if step_logits is not None:
                    # Per-step: current hidden state projection (not running sum).
                    # Each step predicts independently from h_t, like the SNN mem_out.
                    step_logits.append(step_out)
                if return_sparsity and sync_fire > 0:
                    # count only on real input events (skip padding steps)
                    layer_fired[0] += ((o_t != 0).float().sum(dim=1) * valid.squeeze(1)).mean().item()
                continue

            # --- Multi-layer per-event cascade (nlayers >= 2) ---
            o_0 = torch.relu(hs[0]) * sync_fire
            val_0, idx_0 = torch.topk(o_0, K, dim=1)  # (B, K), sorted desc
            if return_sparsity and sync_fire > 0:
                layer_fired[0] += ((val_0 != 0).float().sum(dim=1) * valid.squeeze(1)).mean().item()

            o_last_total = torch.zeros(B, H, device=device)

            for k in range(K):
                idx_k = idx_0[:, k]
                val_k = val_0[:, k]
                valid_k = (val_k != 0)
                if self.drop is not None and self.training:
                    val_k = self.drop(val_k)
                    valid_k = valid_k & (val_k != 0)

                # Layer 1 advances once per layer-0 event
                hs[1] = self._event_step(1, idx_k, val_k, hs[1], valid_k)
                o_1 = torch.relu(hs[1]) * sync_fire

                if self.nlayers == 2:
                    # Layer 1 (last hidden) → output. Either top-k fire (per-event
                    # protocol) or full dense ReLU(h) (when dense_output_firing=True).
                    o_1_fired = o_1 if self.dense_output_firing else keep_top_k_batch_torch(o_1, self.firing_nb)
                    o_last_total = o_last_total + o_1_fired
                    if return_sparsity and sync_fire > 0:
                        # gate by valid_k (real layer-0 event) and valid (real input step)
                        layer_fired[1] += ((o_1_fired != 0).float().sum(dim=1) * valid_k.float() * valid.squeeze(1)).mean().item()
                    continue

                # nlayers >= 3: layer 1 fires events that drive layer 2 per-event
                val_1, idx_1 = torch.topk(o_1, K, dim=1)
                if return_sparsity and sync_fire > 0:
                    layer_fired[1] += ((val_1 != 0).float().sum(dim=1) * valid_k.float() * valid.squeeze(1)).mean().item()
                for kk in range(K):
                    idx_kk = idx_1[:, kk]
                    val_kk = val_1[:, kk]
                    valid_kk = (val_kk != 0)
                    if self.drop is not None and self.training:
                        val_kk = self.drop(val_kk)
                        valid_kk = valid_kk & (val_kk != 0)

                    hs[2] = self._event_step(2, idx_kk, val_kk, hs[2], valid_kk)
                    o_2 = torch.relu(hs[2]) * sync_fire
                    # Layer 2 (last hidden) → output: dense or top-k per the flag.
                    o_2_fired = o_2 if self.dense_output_firing else keep_top_k_batch_torch(o_2, self.firing_nb)
                    o_last_total = o_last_total + o_2_fired
                    if return_sparsity and sync_fire > 0:
                        layer_fired[2] += ((o_2_fired != 0).float().sum(dim=1) * valid_kk.float() * valid.squeeze(1)).mean().item()

            # Output layer accumulates logits per event from last hidden layer.
            # Σ_e val_e · W_out[idx_e] equals (Σ_k o_last^k) @ W_out, so one matmul
            # at end-of-timestep is exact, not an approximation.
            step_out = o_last_total @ self.W_out
            logits = logits + step_out
            if step_logits is not None:
                step_logits.append(step_out)

        # Normalise by number of real (non-padding) events so W_out scale is
        # independent of sequence length and firing density across sessions.
        norm = max(n_real_events, 1.0)
        logits = logits / norm

        if return_per_step:
            # (B, T, C) — instantaneous h_t @ W_out at each event step.
            # Not normalised by n_real_events (each step is independent, not a running sum).
            per_step = torch.stack(step_logits, dim=1)
            if return_sparsity:
                return per_step, [(h,) for h in hs], layer_fired
            return per_step, [(h,) for h in hs]

        if return_sparsity:
            return logits, [(h,) for h in hs], layer_fired
        return logits, [(h,) for h in hs]


class MinimalRNNCell(nn.Module):
    """
    MinimalRNN cell (Chen, 2017 — "MinimalRNN: Toward More Interpretable
    and Trainable Recurrent Neural Networks").

    Equations:
        z_t = tanh(W_phi @ x_t + b_phi)      # candidate from input ONLY
        u_t = sigmoid(W_u @ [h_{t-1}, z_t] + b_u)  # update gate
        h_t = u_t * h_{t-1} + (1 - u_t) * z_t       # interpolation

    Key: the candidate z depends only on x_t (not on h_{t-1}), making it
    more interpretable. Only the update gate sees both h and z.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # phi: input → candidate (input-only transform)
        self.phi = nn.Linear(input_size, hidden_size)
        # update gate: [h, z] → u
        self.gate = nn.Linear(hidden_size + hidden_size, hidden_size)

    def forward(self, x, h):
        """
        Args:
            x: (B, input_size)
            h: (B, hidden_size)
        Returns:
            h_new: (B, hidden_size)
        """
        z = torch.tanh(self.phi(x))
        u = torch.sigmoid(self.gate(torch.cat([h, z], dim=1)))
        h_new = u * h + (1.0 - u) * z
        return h_new


class FPTTMinimalRNN(nn.Module):
    """
    MinimalRNN (Chen, 2017) wrapped for FPTT training.
    API-compatible with FPTTRuleRNN: forward_chunk / init_hidden.
    """

    def __init__(self, input_size, hidden_size, n_classes, nlayers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.cells = nn.ModuleList()
        for i in range(nlayers):
            in_sz = input_size if i == 0 else hidden_size
            self.cells.append(MinimalRNNCell(in_sz, hidden_size))
        self.drop = nn.Dropout(dropout) if nlayers > 1 and dropout > 0 else None
        self.fc = nn.Linear(hidden_size, n_classes)

    def init_hidden(self, batch_size):
        """Returns [(h0,), (h1,), ...] — one 1-tuple per layer."""
        device = self.fc.weight.device
        return [
            (torch.zeros(batch_size, self.hidden_size, device=device),)
            for _ in range(self.nlayers)
        ]

    def forward_chunk(self, x_chunk, hidden, t_offset):
        """
        Args:
            x_chunk: (B, chunk_len, 2) — event format (neuron_idx, value)
            hidden:  list of (h,) per layer
            t_offset: ignored
        Returns:
            logits: (B, n_classes)
            hidden: updated list of (h,) per layer
        """
        values = x_chunk[:, :, 1]                           # (B, chunk_len)
        x_seq = values.unsqueeze(-1)                         # (B, chunk_len, 1)
        B, T, _ = x_seq.shape

        new_hidden = [h_tup[0] for h_tup in hidden]  # unwrap to list of (B, H)

        for t in range(T):
            inp = x_seq[:, t, :]                             # (B, input_size)
            for i, cell in enumerate(self.cells):
                new_hidden[i] = cell(inp, new_hidden[i])
                inp = new_hidden[i]
                if self.drop is not None and i < self.nlayers - 1:
                    inp = self.drop(inp)

        logits = self.fc(new_hidden[-1])                     # (B, n_classes)
        return logits, [(h,) for h in new_hidden]


# ---------------------------------------------------------------------------
#region Manual gradient computation (no autograd) — multi-layer
# ---------------------------------------------------------------------------

def manual_forward_chunk(model, x_chunk, hidden, t_offset):
    """
    Multi-layer forward pass storing all intermediates for manual backprop.

    Args:
        x_chunk: (B, chunk_len, 2) — each event is (neuron_idx, value)

    Returns:
        logits:     (B, C)
        new_hidden: list of (z, o) per layer
        cache:      dict with per-layer, per-timestep intermediates
    """
    L = model.n_layers
    B, chunk_len, _ = x_chunk.shape
    C = model.w_out.data.shape[1]
    device = model.w_ih.device

    logits = torch.zeros(B, C, device=device)

    # Per-layer storage
    layer_cache = []
    for l in range(L):
        layer_cache.append(dict(
            all_z=[], all_o=[], all_o_prev=[], all_z_prev=[],
            all_fire_mask=[], all_drive=[],
        ))

    # Per-timestep input info for backward pass
    all_neuron_idx = []
    all_value = []
    all_valid = []

    for local_t in range(chunk_len):
        t = t_offset + local_t
        neuron_idx = x_chunk[:, local_t, 0].long()   # (B,)
        value      = x_chunk[:, local_t, 1]           # (B,)
        valid = (neuron_idx >= 0).float()             # (B,)
        safe_idx = neuron_idx.clamp(min=0)

        all_neuron_idx.append(safe_idx)
        all_value.append(value)
        all_valid.append(valid)

        w_selected = model.w_ih.data[safe_idx]        # (B, H0)
        drive = value.unsqueeze(1) * w_selected * valid.unsqueeze(1)

        for l in range(L):
            z_prev, o_prev = hidden[l]

            if l > 0:
                drive = o_prev_layer @ model.w_ll[l - 1].data

            if model.vanilla:
                # Vanilla: tanh or relu depending on use_tanh flag
                inner = (
                    drive
                    + o_prev @ model.w_hh[l].data
                    + model.bias[l].data.unsqueeze(0)
                )
                h_t = torch.tanh(inner) if model.use_tanh else torch.relu(inner)
                z_t = h_t
                o_t = h_t
                fire_mask = torch.ones_like(o_t)
            else:
                reset = 0.0 if model.no_reset else o_prev
                inner = (
                    drive
                    + z_prev
                    - reset
                    + o_prev @ model.w_hh[l].data
                    + model.bias[l].data.unsqueeze(0)
                )

                z_t = torch.tanh(inner) if model.use_tanh else inner.clone()
                o_t_raw = torch.relu(z_t)

                sync_fire = 1.0 if ((t + 1) % model.sync_rate == 0) else 0.0
                o_t_synced = o_t_raw * sync_fire
                o_t = keep_top_k_batch_torch(o_t_synced, model.firing_nb)

                fire_mask = (o_t != 0).float()

            lc = layer_cache[l]
            lc['all_z'].append(z_t)
            lc['all_o'].append(o_t)
            lc['all_o_prev'].append(o_prev)
            lc['all_z_prev'].append(z_prev)
            lc['all_fire_mask'].append(fire_mask)
            lc['all_drive'].append(drive)

            hidden[l] = (z_t, o_t)
            o_prev_layer = o_t

        if not model.vanilla:
            logits = logits + hidden[L - 1][1] @ model.w_out.data

    if model.vanilla:
        logits = hidden[L - 1][1] @ model.w_out.data

    cache = dict(
        layer_cache=layer_cache,
        all_neuron_idx=all_neuron_idx,  # list of (B,) long tensors
        all_value=all_value,            # list of (B,) float tensors
        all_valid=all_valid,            # list of (B,) float tensors
        t_offset=t_offset,
        chunk_len=chunk_len,
    )
    return logits, hidden, cache


def manual_backward_chunk(model, logits, y, oracle_prob, beta_p, cache,
                           logit_grad_scale=1.0):
    """
    Manual backpropagation through one FPTT chunk for multi-layer RNN.

    Args:
        logit_grad_scale: Chain-rule factor from logit accumulation.
            When avg_logits is used, this should be 1/(p+1).

    Returns:
        grads: dict with:
            'w_ih':      (n_input_neurons, H0) — full-size, scatter-added by neuron idx
            'w_hh':      list of L tensors, each (Hl, Hl)
            'w_ll':      list of (L-1) tensors, each (Hl, H_{l+1})
            'w_out':     (H_{L-1}, C)
            'bias':      list of L tensors, each (Hl,)
        loss_value: scalar
    """
    L = model.n_layers
    B = logits.shape[0]
    C = logits.shape[1]
    chunk_len = cache['chunk_len']
    layer_cache = cache['layer_cache']

    # --- Compute loss and dL/d(logits) ---
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)

    one_hot_y = F.one_hot(y, num_classes=C).float()
    d_ce = (probs - one_hot_y) / B
    d_oracle = (probs - oracle_prob) / B
    d_logits = beta_p * d_ce + (1.0 - beta_p) * d_oracle

    # Chain rule through logit accumulation (e.g. avg_logits: scale = 1/(p+1))
    d_logits = d_logits * logit_grad_scale

    ce_loss = F.nll_loss(log_probs, y)
    oracle_loss = torch.mean(-torch.sum(oracle_prob * log_probs, dim=1))
    loss_val = float((beta_p * ce_loss + (1.0 - beta_p) * oracle_loss).item())

    # --- Initialize gradient accumulators ---
    grad_w_out = torch.zeros_like(model.w_out.data)

    grad_w_hh = [torch.zeros_like(model.w_hh[l].data) for l in range(L)]
    grad_w_ll = [torch.zeros_like(model.w_ll[l].data) for l in range(L - 1)]
    grad_bias = [torch.zeros_like(model.bias[l].data) for l in range(L)]
    H0 = model.w_hh[0].shape[0]
    n_input = model.w_ih.shape[0]
    device = model.w_ih.device
    grad_w_ih = torch.zeros(n_input, H0, device=device)

    # Per-layer recurrence deltas (flowing from step t+1 to step t)
    delta_z_next = [torch.zeros(B, model.w_hh[l].shape[0], device=device) for l in range(L)]
    delta_o_next = [torch.zeros(B, model.w_hh[l].shape[0], device=device) for l in range(L)]

    for local_t in reversed(range(chunk_len)):
        lc_last = layer_cache[L - 1]
        o_last = lc_last['all_o'][local_t]

        if model.vanilla:
            # Vanilla: only last timestep contributes to logits
            if local_t == chunk_len - 1:
                grad_w_out += o_last.T @ d_logits
                dL_do_from_above = d_logits @ model.w_out.data.T
            else:
                dL_do_from_above = torch.zeros(B, model.w_hh[L - 1].shape[0], device=device)
        else:
            grad_w_out += o_last.T @ d_logits
            dL_do_from_above = d_logits @ model.w_out.data.T

        for l in reversed(range(L)):
            lc = layer_cache[l]
            z_t = lc['all_z'][local_t]
            o_prev = lc['all_o_prev'][local_t]
            fire_mask = lc['all_fire_mask'][local_t]

            dL_do = dL_do_from_above + delta_o_next[l]

            if model.vanilla:
                # Vanilla: tanh → dL/d(inner) = dL/dh * (1 - h^2)
                #          relu → dL/d(inner) = dL/dh * (inner > 0)
                if model.use_tanh:
                    act_deriv = 1.0 - z_t ** 2
                else:
                    act_deriv = (z_t > 0).float()
                dL_dinner = dL_do * act_deriv
            else:
                # Through topk + sync mask
                dL_do_raw = dL_do * fire_mask

                # Through ReLU
                relu_mask = (z_t > 0).float()
                dL_dz = dL_do_raw * relu_mask

                # Add recurrence contribution to z from step t+1
                dL_dz = dL_dz + delta_z_next[l]

                # Through tanh
                if model.use_tanh:
                    tanh_deriv = 1.0 - z_t ** 2
                    dL_dinner = dL_dz * tanh_deriv
                else:
                    dL_dinner = dL_dz

            # --- Gradient for W_hh[l] ---
            grad_w_hh[l] += o_prev.T @ dL_dinner

            # --- Gradient for bias[l] ---
            grad_bias[l] += dL_dinner.sum(dim=0)

            # --- Gradient for drive (input to this layer) ---
            if l == 0:
                # drive = value * w_ih[neuron_idx] * valid
                # dL/dW_ih[idx] += value * dL_dinner (summed over batch samples with that idx)
                safe_idx = cache['all_neuron_idx'][local_t]  # (B,)
                value    = cache['all_value'][local_t]        # (B,)
                valid    = cache['all_valid'][local_t]        # (B,)
                # Per-sample gradient contribution: (B, H0)
                per_sample = value.unsqueeze(1) * dL_dinner * valid.unsqueeze(1)
                # Scatter-add into grad_w_ih by neuron index
                grad_w_ih.scatter_add_(0, safe_idx.unsqueeze(1).expand_as(per_sample), per_sample)
            else:
                o_below = layer_cache[l - 1]['all_o'][local_t]
                grad_w_ll[l - 1] += o_below.T @ dL_dinner
                dL_do_from_above = dL_dinner @ model.w_ll[l - 1].data.T

            # --- Propagate recurrence deltas to step t-1 ---
            if model.vanilla:
                # Vanilla: only W_hh recurrence, no z_prev additive term
                delta_o_next[l] = dL_dinner @ model.w_hh[l].data.T
            else:
                delta_z_next[l] = dL_dinner
                delta_o_next[l] = dL_dinner @ model.w_hh[l].data.T
                if not model.no_reset:
                    delta_o_next[l] = delta_o_next[l] - dL_dinner

    grads = dict(
        w_ih=grad_w_ih,
        w_hh=grad_w_hh,
        w_ll=grad_w_ll,
        w_out=grad_w_out,
        bias=grad_bias,
    )
    return grads, loss_val


# ---------------------------------------------------------------------------
#region Manual gradient computation for MinimalRNN
# ---------------------------------------------------------------------------

def manual_forward_chunk_minimalrnn(model, x_chunk, hidden, t_offset):
    """
    Manual forward pass for FPTTMinimalRNN, storing intermediates for backprop.

    Args:
        model:   FPTTMinimalRNN instance
        x_chunk: (B, chunk_len, 2) — event format (neuron_idx, value)
        hidden:  list of (h,) per layer — from init_hidden or previous chunk
        t_offset: unused (kept for API compatibility)

    Returns:
        logits:     (B, C)
        new_hidden: list of (h,) per layer
        cache:      dict with per-layer, per-timestep intermediates
    """
    L = model.nlayers
    B, chunk_len, _ = x_chunk.shape
    device = model.fc.weight.device

    values = x_chunk[:, :, 1]       # (B, chunk_len)
    x_seq = values.unsqueeze(-1)     # (B, chunk_len, 1)

    # Unwrap hidden: list of (B, H)
    h_states = [h_tup[0].detach() for h_tup in hidden]

    # Per-layer cache
    layer_cache = []
    for l in range(L):
        layer_cache.append(dict(
            all_x_inp=[],     # input to this layer each timestep
            all_h_prev=[],    # h_{t-1}
            all_z=[],         # candidate z_t = tanh(phi(x))
            all_u=[],         # update gate u_t = sigmoid(gate([h, z]))
            all_h=[],         # output h_t
        ))

    for t in range(chunk_len):
        inp = x_seq[:, t, :]  # (B, input_size) for layer 0

        for l in range(L):
            cell = model.cells[l]
            h_prev = h_states[l]

            # Forward: z = tanh(phi(inp)), u = sigmoid(gate([h_prev, z])), h = u*h_prev + (1-u)*z
            z = torch.tanh(cell.phi(inp))
            cat_hz = torch.cat([h_prev, z], dim=1)
            u = torch.sigmoid(cell.gate(cat_hz))
            h_new = u * h_prev + (1.0 - u) * z

            lc = layer_cache[l]
            lc['all_x_inp'].append(inp)
            lc['all_h_prev'].append(h_prev)
            lc['all_z'].append(z)
            lc['all_u'].append(u)
            lc['all_h'].append(h_new)

            h_states[l] = h_new
            inp = h_new  # input to next layer

    logits = model.fc(h_states[-1])  # (B, C)

    cache = dict(
        layer_cache=layer_cache,
        chunk_len=chunk_len,
    )
    return logits, [(h,) for h in h_states], cache


def manual_backward_chunk_minimalrnn(model, logits, y, oracle_prob, beta_p, cache,
                                      logit_grad_scale=1.0):
    """
    Manual backpropagation through one FPTT chunk for FPTTMinimalRNN.

    MinimalRNN cell equations:
        z_t = tanh(W_phi @ x_t + b_phi)
        u_t = sigmoid(W_gate @ [h_{t-1}, z_t] + b_gate)
        h_t = u_t * h_{t-1} + (1 - u_t) * z_t

    Args:
        logit_grad_scale: Chain-rule factor from logit accumulation.
            When avg_logits is used, effective_logits = (sum + chunk) / (p+1),
            so d(effective)/d(chunk) = 1/(p+1) and this should be 1/(p+1).

    Returns:
        grads: dict mapping parameter names to gradient tensors
        loss_value: scalar
    """
    L = model.nlayers
    B = logits.shape[0]
    C = logits.shape[1]
    H = model.hidden_size
    chunk_len = cache['chunk_len']
    layer_cache = cache['layer_cache']
    device = model.fc.weight.device

    # --- Loss and dL/d(logits) ---
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)

    one_hot_y = F.one_hot(y, num_classes=C).float()
    d_ce = (probs - one_hot_y) / B
    d_oracle = (probs - oracle_prob) / B
    d_logits = beta_p * d_ce + (1.0 - beta_p) * d_oracle

    # Chain rule through logit accumulation (e.g. avg_logits: scale = 1/(p+1))
    d_logits = d_logits * logit_grad_scale

    ce_loss = F.nll_loss(log_probs, y)
    oracle_loss = torch.mean(-torch.sum(oracle_prob * log_probs, dim=1))
    loss_val = float((beta_p * ce_loss + (1.0 - beta_p) * oracle_loss).item())

    # --- Gradient accumulators ---
    # fc layer
    grad_fc_weight = torch.zeros_like(model.fc.weight.data)  # (C, H)
    grad_fc_bias = torch.zeros_like(model.fc.bias.data)       # (C,)

    # Per-cell gradients
    grad_phi_weight = [torch.zeros_like(model.cells[l].phi.weight.data) for l in range(L)]
    grad_phi_bias = [torch.zeros_like(model.cells[l].phi.bias.data) for l in range(L)]
    grad_gate_weight = [torch.zeros_like(model.cells[l].gate.weight.data) for l in range(L)]
    grad_gate_bias = [torch.zeros_like(model.cells[l].gate.bias.data) for l in range(L)]

    # --- dL/d(fc input) = dL/d(h_last_layer at final timestep) ---
    # logits = h @ W^T + b  where W is fc.weight (C, H)
    # dL/dW = d_logits^T @ h = (C, B)^T ... but d_logits is (B, C) so: (B,C)^T @ (B,H) = (C,H)
    h_final = layer_cache[L - 1]['all_h'][chunk_len - 1]
    grad_fc_weight += d_logits.T @ h_final
    grad_fc_bias += d_logits.sum(dim=0)

    # dL/dh for last layer at final timestep
    dL_dh_top = d_logits @ model.fc.weight.data  # (B, H)

    # --- Per-layer recurrence: dL/dh flowing backward through time ---
    # dL_dh[l] = gradient w.r.t. h_t at layer l
    dL_dh = [torch.zeros(B, H, device=device) for _ in range(L)]
    dL_dh[L - 1] = dL_dh_top

    for t in reversed(range(chunk_len)):
        # For layers top-down: propagate dL_dh through the cell
        for l in reversed(range(L)):
            lc = layer_cache[l]
            h_prev = lc['all_h_prev'][t]
            z = lc['all_z'][t]
            u = lc['all_u'][t]
            x_inp = lc['all_x_inp'][t]

            dL_dh_t = dL_dh[l]  # (B, H)

            # h_t = u * h_prev + (1 - u) * z
            # dL/du = dL/dh_t * (h_prev - z)
            dL_du = dL_dh_t * (h_prev - z)  # (B, H)

            # dL/dz = dL/dh_t * (1 - u)
            dL_dz = dL_dh_t * (1.0 - u)  # (B, H)

            # dL/dh_prev from the interpolation
            dL_dh_prev = dL_dh_t * u  # (B, H)

            # --- Through sigmoid gate ---
            # u = sigmoid(gate_pre), gate_pre = W_gate @ [h_prev, z] + b_gate
            # du/d(gate_pre) = u * (1 - u)
            sig_deriv = u * (1.0 - u)  # (B, H)
            dL_dgate_pre = dL_du * sig_deriv  # (B, H)

            # W_gate is (H, 2H), input is [h_prev, z]
            W_gate = model.cells[l].gate.weight.data  # (H, 2H)
            # dL/d[h_prev, z] from gate
            dL_dcat = dL_dgate_pre @ W_gate  # (B, 2H)
            dL_dh_prev_gate = dL_dcat[:, :H]
            dL_dz_gate = dL_dcat[:, H:]

            # Total dL/dh_prev and dL/dz
            dL_dh_prev = dL_dh_prev + dL_dh_prev_gate
            dL_dz_total = dL_dz + dL_dz_gate

            # --- Through tanh candidate ---
            # z = tanh(phi_pre), phi_pre = W_phi @ x_inp + b_phi
            tanh_deriv = 1.0 - z ** 2  # (B, H)
            dL_dphi_pre = dL_dz_total * tanh_deriv  # (B, H)

            # --- Accumulate gradients ---
            # gate: W_gate (H, 2H), input = [h_prev, z]
            cat_hz = torch.cat([h_prev, z], dim=1)  # (B, 2H)
            grad_gate_weight[l] += dL_dgate_pre.T @ cat_hz  # (H, 2H)
            grad_gate_bias[l] += dL_dgate_pre.sum(dim=0)    # (H,)

            # phi: W_phi (H, in_sz), input = x_inp
            grad_phi_weight[l] += dL_dphi_pre.T @ x_inp  # (H, in_sz)
            grad_phi_bias[l] += dL_dphi_pre.sum(dim=0)    # (H,)

            # --- Propagate to lower layer ---
            # dL/d(x_inp) for this layer = dL/d(phi_pre) @ W_phi  (from phi path)
            #                             + dL/d(gate_pre) contribution is already in dL_dz_gate
            # Actually x_inp only enters through phi, so:
            W_phi = model.cells[l].phi.weight.data  # (H, in_sz)
            dL_dx_inp = dL_dphi_pre @ W_phi  # (B, in_sz)

            if l > 0:
                # x_inp to layer l was h from layer l-1 at same timestep
                # So add this to dL_dh[l-1]
                dL_dh[l - 1] += dL_dx_inp

            # --- Propagate recurrence to previous timestep ---
            # dL_dh[l] for timestep t-1 gets dL_dh_prev
            dL_dh[l] = dL_dh_prev

    grads = {}
    for l in range(L):
        grads[f'cells.{l}.phi.weight'] = grad_phi_weight[l]
        grads[f'cells.{l}.phi.bias'] = grad_phi_bias[l]
        grads[f'cells.{l}.gate.weight'] = grad_gate_weight[l]
        grads[f'cells.{l}.gate.bias'] = grad_gate_bias[l]
    grads['fc.weight'] = grad_fc_weight
    grads['fc.bias'] = grad_fc_bias

    return grads, loss_val

#endregion


def train_one_epoch_manual_minimalrnn(
    x_train, y_train,
    model, optimizer, named_params,
    estimate_class_distribution,
    epoch,
    PARTS,
    alpha, beta, lmbda, rho,
    batch_size,
    clip,
    n_classes,
    warm_epochs=1,
    accumulate_logits=False,
    avg_logits=False,
):
    """
    FPTT training for one epoch using manual gradient computation for MinimalRNN.
    """
    model.train()
    L = model.nlayers
    n = x_train.shape[0]
    T = x_train.shape[1]
    step = T // PARTS
    _PARTS = PARTS if PARTS * step >= T else PARTS + 1

    rng = np.random.default_rng(epoch)
    perm = rng.permutation(n)

    total_loss = 0.0
    total_batches = 0

    for s in range(0, n, batch_size):
        idx = perm[s : s + batch_size]
        device = next(model.parameters()).device
        xb = torch.tensor(x_train[idx], dtype=torch.float32, device=device)
        yb = torch.tensor(y_train[idx], dtype=torch.long, device=device)
        B = yb.shape[0]

        hidden = model.init_hidden(B)
        cum_logits = torch.zeros(B, n_classes, device=device) if accumulate_logits else None

        for p in range(_PARTS):
            start = p * step
            end = min(start + step, T)
            if start >= T:
                break
            x_chunk = xb[:, start:end, :]

            # Detach hidden state
            h_detached = [(h.detach(),) for (h,) in hidden]

            # Oracle distribution
            if p < _PARTS - 1:
                if epoch <= warm_epochs:
                    oracle_prob = torch.full((B, n_classes), 1.0 / n_classes, device=device)
                else:
                    oracle_prob = estimate_class_distribution[yb.cpu(), p].to(device)
            else:
                oracle_prob = F.one_hot(yb, num_classes=n_classes).float()

            # Forward (manual)
            logits, hidden, cache = manual_forward_chunk_minimalrnn(
                model, x_chunk, h_detached, t_offset=start
            )

            # Use accumulated logits for loss if enabled
            if accumulate_logits:
                sum_logits = cum_logits + logits
                effective_logits = sum_logits / (p + 1) if avg_logits else sum_logits
            else:
                effective_logits = logits

            # Backward (manual) — pass effective_logits so loss is computed on accumulated signal
            beta_p = (p + 1) / _PARTS
            # Chain rule: d(effective_logits)/d(chunk_logits) = 1/(p+1) for avg, 1 for sum
            logit_scale = 1.0 / (p + 1) if (accumulate_logits and avg_logits) else 1.0
            grads, chunk_loss = manual_backward_chunk_minimalrnn(
                model, effective_logits, yb, oracle_prob, beta_p, cache,
                logit_grad_scale=logit_scale,
            )

            # Update cumulative logits (detached) — always store raw sum
            if accumulate_logits:
                cum_logits = sum_logits.detach()

            # Assign gradients to model parameters
            for name, param in model.named_parameters():
                _, sm, lm = named_params[name]
                reg_grad = (rho - 1.0) * lm + lmbda * alpha * (param.data - sm)

                if name in grads:
                    param.grad = grads[name] + reg_grad
                else:
                    param.grad = reg_grad

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            post_optimizer_updates(named_params, alpha=alpha, beta=beta)

            total_loss += chunk_loss
            total_batches += 1

            # Update oracle estimates
            if p < _PARTS - 1:
                with torch.no_grad():
                    probs = torch.softmax(effective_logits, dim=1)
                    filled = [False] * n_classes
                    for j in range(B):
                        c = int(yb[j].item())
                        if not filled[c] and torch.argmax(probs[j]).item() != c:
                            estimate_class_distribution[c, p] = probs[j].detach().cpu()
                            filled[c] = True
                        if all(filled):
                            break

    return total_loss / max(total_batches, 1)


def train_one_epoch_manual(
    x_train, y_train,
    model, optimizer, named_params,
    estimate_class_distribution,
    epoch,
    PARTS,
    alpha, beta, lmbda, rho,
    batch_size,
    clip,
    n_classes,
    warm_epochs=1,
    accumulate_logits=False,
    avg_logits=False,
):
    """
    FPTT training for one epoch using manual gradient computation (no autograd).
    Supports multi-layer RNN.
    """
    model.train()
    L = model.n_layers
    n = x_train.shape[0]
    T = x_train.shape[1]
    step = T // PARTS
    _PARTS = PARTS if PARTS * step >= T else PARTS + 1

    rng = np.random.default_rng(epoch)
    perm = rng.permutation(n)

    total_loss = 0.0
    total_batches = 0

    for s in range(0, n, batch_size):
        idx = perm[s : s + batch_size]
        device = next(model.parameters()).device
        xb  = torch.tensor(x_train[idx], dtype=torch.float32, device=device)  # (B, T, 2)
        yb  = torch.tensor(y_train[idx], dtype=torch.long, device=device)
        B   = yb.shape[0]

        hidden = model.init_hidden(B)
        cum_logits = torch.zeros(B, n_classes, device=device) if accumulate_logits else None

        for p in range(_PARTS):
            start = p * step
            end   = min(start + step, T)
            if start >= T:
                break
            x_chunk = xb[:, start:end, :]  # (B, chunk_len, 2)

            # Detach hidden state
            h_detached = [(z.detach(), o.detach()) for z, o in hidden]

            # Oracle distribution
            if p < _PARTS - 1:
                if epoch <= warm_epochs:
                    oracle_prob = torch.full((B, n_classes), 1.0 / n_classes, device=device)
                else:
                    oracle_prob = estimate_class_distribution[yb.cpu(), p].to(device)
            else:
                oracle_prob = F.one_hot(yb, num_classes=n_classes).float()

            # Forward (manual, no grad tracking)
            logits, hidden, cache = manual_forward_chunk(model, x_chunk, h_detached, t_offset=start)

            # Use accumulated logits for loss if enabled
            if accumulate_logits:
                sum_logits = cum_logits + logits
                effective_logits = sum_logits / (p + 1) if avg_logits else sum_logits
            else:
                effective_logits = logits

            # Backward (manual)
            beta_p = (p + 1) / _PARTS
            # Chain rule: d(effective_logits)/d(chunk_logits) = 1/(p+1) for avg, 1 for sum
            logit_scale = 1.0 / (p + 1) if (accumulate_logits and avg_logits) else 1.0
            grads, chunk_loss = manual_backward_chunk(
                model, effective_logits, yb, oracle_prob, beta_p, cache,
                logit_grad_scale=logit_scale,
            )

            # Update cumulative logits (detached) — always store raw sum
            if accumulate_logits:
                cum_logits = sum_logits.detach()

            # Assign gradients to model parameters
            for name, param in model.named_parameters():
                _, sm, lm = named_params[name]
                reg_grad = (rho - 1.0) * lm + lmbda * alpha * (param.data - sm)

                if name == 'w_ih':
                    param.grad = grads['w_ih'] + reg_grad
                elif name == 'w_out':
                    param.grad = grads['w_out'] + reg_grad
                elif name.startswith('w_hh.'):
                    l = int(name.split('.')[1])
                    param.grad = grads['w_hh'][l] + reg_grad
                elif name.startswith('w_ll.'):
                    l = int(name.split('.')[1])
                    param.grad = grads['w_ll'][l] + reg_grad
                elif name.startswith('bias.'):
                    l = int(name.split('.')[1])
                    param.grad = grads['bias'][l] + reg_grad

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            post_optimizer_updates(named_params, alpha=alpha, beta=beta)

            total_loss    += chunk_loss
            total_batches += 1

            # Update oracle estimates
            if p < _PARTS - 1:
                with torch.no_grad():
                    probs = torch.softmax(effective_logits, dim=1)
                    filled = [False] * n_classes
                    for j in range(B):
                        c = int(yb[j].item())
                        if not filled[c] and torch.argmax(probs[j]).item() != c:
                            estimate_class_distribution[c, p] = probs[j].detach().cpu()
                            filled[c] = True
                        if all(filled):
                            break

    return total_loss / max(total_batches, 1)


def compare_gradients(model, x_chunk, hidden, y, oracle_prob, beta_p, n_classes):
    """
    Compare autograd gradients with manual gradients on one chunk.
    x_chunk: (B, chunk_len, 2) — event format.
    Returns max absolute difference per parameter.
    """
    L = model.n_layers
    B = x_chunk.shape[0]
    t_offset = 0

    # --- Autograd ---
    model.zero_grad()
    hidden_auto = [(z.clone(), o.clone()) for z, o in hidden]
    logits_auto, _ = model.forward_chunk(x_chunk, hidden_auto, t_offset)
    log_probs = F.log_softmax(logits_auto, dim=1)
    clf_loss = beta_p * F.nll_loss(log_probs, y)
    oracle_loss = (1.0 - beta_p) * torch.mean(-torch.sum(oracle_prob * log_probs, dim=1))
    loss = clf_loss + oracle_loss
    loss.backward()

    auto_grads = {}
    for name, param in model.named_parameters():
        auto_grads[name] = param.grad.clone()

    # --- Manual ---
    h_detached = [(z.detach(), o.detach()) for z, o in hidden]
    logits_man, _, cache = manual_forward_chunk(model, x_chunk, h_detached, t_offset)
    man_grads, _ = manual_backward_chunk(model, logits_man, y, oracle_prob, beta_p, cache)

    # Compare
    results = {}

    # W_ih: both are full-size (n_input_neurons, H0)
    results['w_ih'] = float((auto_grads['w_ih'] - man_grads['w_ih']).abs().max())

    # W_out
    results['w_out'] = float((auto_grads['w_out'] - man_grads['w_out']).abs().max())

    # Per-layer W_hh, bias
    for l in range(L):
        key_hh = f'w_hh.{l}'
        results[key_hh] = float((auto_grads[key_hh] - man_grads['w_hh'][l]).abs().max())
        key_b = f'bias.{l}'
        results[key_b] = float((auto_grads[key_b] - man_grads['bias'][l]).abs().max())

    # Per inter-layer W_ll
    for l in range(L - 1):
        key_ll = f'w_ll.{l}'
        results[key_ll] = float((auto_grads[key_ll] - man_grads['w_ll'][l]).abs().max())

    # Logits match
    results['logits_diff'] = float((logits_auto.detach() - logits_man).abs().max())

    return results


# ---------------------------------------------------------------------------
#region FPTT training (autograd version)
# ---------------------------------------------------------------------------

def train_one_epoch(
    x_train, y_train,
    model, optimizer, named_params,
    estimate_class_distribution,
    epoch,
    PARTS,
    alpha, beta, lmbda, rho,
    batch_size,
    clip,
    n_classes,
    warm_epochs=1,
    accumulate_logits=False,
    avg_logits=False,
):
    """Train for one epoch using FPTT (autograd). Returns average surrogate loss."""
    model.train()
    device = next(model.parameters()).device
    n = x_train.shape[0]
    T = x_train.shape[1]
    step = T // PARTS
    _PARTS = PARTS if PARTS * step >= T else PARTS + 1

    rng = np.random.default_rng(epoch)
    perm = rng.permutation(n)

    total_loss = 0.0
    total_batches = 0

    for s in range(0, n, batch_size):
        idx = perm[s : s + batch_size]
        xb  = torch.tensor(x_train[idx], dtype=torch.float32, device=device)
        yb  = torch.tensor(y_train[idx], dtype=torch.long, device=device)
        B   = yb.shape[0]

        hidden = model.init_hidden(B)
        # Accumulated logits across chunks (detached — no cross-chunk gradient)
        cum_logits = torch.zeros(B, n_classes, device=device) if accumulate_logits else None

        for p in range(_PARTS):
            start = p * step
            end   = min(start + step, T)
            if start >= T:
                break
            x_chunk = xb[:, start:end, :]  # (B, chunk_len, 2)

            # Detach hidden state: no gradient flow between chunks
            h_detached = [tuple(t.detach() for t in tup) for tup in hidden]

            # Oracle distribution for this chunk
            if p < _PARTS - 1:
                if epoch <= warm_epochs:
                    oracle_prob = torch.full((B, n_classes), 1.0 / n_classes, device=device)
                else:
                    oracle_prob = estimate_class_distribution[yb.cpu(), p].to(device)
            else:
                oracle_prob = F.one_hot(yb, num_classes=n_classes).float()

            optimizer.zero_grad()
            chunk_logits, hidden = model.forward_chunk(x_chunk, h_detached, t_offset=start)

            # Use accumulated logits for loss if enabled
            if accumulate_logits:
                sum_logits = cum_logits + chunk_logits
                effective_logits = sum_logits / (p + 1) if avg_logits else sum_logits
            else:
                effective_logits = chunk_logits

            beta_p    = (p + 1) / _PARTS
            log_probs = F.log_softmax(effective_logits, dim=1)

            clf_loss    = beta_p         * F.nll_loss(log_probs, yb)
            oracle_loss = (1.0 - beta_p) * torch.mean(-torch.sum(oracle_prob * log_probs, dim=1))
            regularizer = get_regularizer(named_params, alpha=alpha, lmbda=lmbda, rho=rho)
            loss = clf_loss + oracle_loss + regularizer

            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            post_optimizer_updates(named_params, alpha=alpha, beta=beta)

            # Update cumulative logits (detached — gradient only flows through current chunk)
            # Always store the raw sum, not the averaged version
            if accumulate_logits:
                cum_logits = sum_logits.detach()

            total_loss    += float(loss.item())
            total_batches += 1

            if p < _PARTS - 1:
                with torch.no_grad():
                    probs = torch.softmax(effective_logits, dim=1)
                    filled = [False] * n_classes
                    for j in range(B):
                        c = int(yb[j].item())
                        if not filled[c] and torch.argmax(probs[j]).item() != c:
                            estimate_class_distribution[c, p] = probs[j].detach().cpu()
                            filled[c] = True
                        if all(filled):
                            break

    return total_loss / max(total_batches, 1)


def evaluate(x, y, model, batch_size, n_classes, accumulate_logits=False, PARTS=1,
             avg_logits=False, task="classification", stateful=False, per_step_last=False):
    """Full-sequence evaluation (no weight updates).
    x: (N, T, 2) — preprocessed event format.
    task='classification': returns accuracy.
    task='regression': returns R² averaged across output dims (matches r2_from_sums).
    stateful=True: process samples sequentially (batch=1), carry hidden state from
        one window to the next so the RNN sees long-range temporal context. Useful
        for neural decoding where consecutive samples are consecutive time bins.
    """
    model.eval()
    device = next(model.parameters()).device
    T = x.shape[1]
    step = T // PARTS
    _PARTS = PARTS if PARTS * step >= T else PARTS + 1
    all_preds = []
    with torch.no_grad():
        if stateful:
            # Batched-streams stateful eval: split x (in time order) into B parallel
            # streams, each of which carries its own hidden state forward sample-by-
            # sample. ~B× faster than batch=1 sequential, with B cold-starts instead
            # of 1 (negligible if chunk_size ≫ "warm-up time").
            N = x.shape[0]
            B = max(1, batch_size)
            chunk_size = N // B
            usable = B * chunk_size
            # x_streams: (B, chunk_size, T, 2) — stream i = x[i*chunk_size : (i+1)*chunk_size]
            x_streams = x[:usable].reshape(B, chunk_size, *x.shape[1:])
            hidden = model.init_hidden(B)
            all_chunk_preds = []
            for k in range(chunk_size):
                xb = torch.tensor(x_streams[:, k], dtype=torch.float32, device=device)
                logits, hidden = model.forward_chunk(xb, hidden, t_offset=0)
                hidden = [tuple(h.detach() for h in tup) for tup in hidden]
                all_chunk_preds.append(logits.cpu().numpy() if task == "regression"
                                       else logits.argmax(dim=1).cpu().numpy())
            # Stack to (chunk_size, B, ...) then transpose to (B, chunk_size, ...)
            stacked = np.stack(all_chunk_preds, axis=0).swapaxes(0, 1)
            # Flatten to (usable, ...) — preserves original time order
            all_preds.append(stacked.reshape(usable, *stacked.shape[2:]))
            # If there's a leftover tail, do it stateless (rare; only the last <B samples).
            if usable < N:
                tail = x[usable:]
                hidden = model.init_hidden(tail.shape[0])
                xb = torch.tensor(tail, dtype=torch.float32, device=device)
                logits, _ = model.forward_chunk(xb, hidden, t_offset=0)
                all_preds.append(logits.cpu().numpy() if task == "regression"
                                 else logits.argmax(dim=1).cpu().numpy())
        else:
            for s in range(0, x.shape[0], batch_size):
                xb = torch.tensor(x[s : s + batch_size], dtype=torch.float32, device=device)
                B  = xb.shape[0]
                hidden = model.init_hidden(B)
                if accumulate_logits and _PARTS > 1:
                    C = model.fc.weight.shape[0] if hasattr(model, 'fc') else model.w_out.data.shape[1]
                    cum_logits = torch.zeros(B, C, device=device)
                    for p in range(_PARTS):
                        start = p * step
                        end = min(start + step, T)
                        if start >= T:
                            break
                        x_chunk = xb[:, start:end, :]
                        h_detached = [tuple(t.detach() for t in tup) for tup in hidden]
                        chunk_logits, hidden = model.forward_chunk(x_chunk, h_detached, t_offset=start)
                        cum_logits = cum_logits + chunk_logits
                    logits = cum_logits
                elif per_step_last:
                    per_step, _ = model.forward_chunk(xb, hidden, t_offset=0,
                                                      return_per_step=True)
                    logits = per_step[:, -1, :]
                else:
                    logits, _ = model.forward_chunk(xb, hidden, t_offset=0)
                all_preds.append(logits.cpu().numpy() if task == "regression"
                                 else logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(all_preds)
    if task == "regression":
        # R² averaged across output dimensions, matching r2_from_sums semantics.
        y2 = y if y.ndim == 2 else y[:, None]
        ss_res = np.sum((y2 - preds) ** 2, axis=0)
        ss_tot = np.sum((y2 - y2.mean(axis=0, keepdims=True)) ** 2, axis=0)
        r2_per_dim = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
        return float(r2_per_dim.mean())
    return float(np.mean(preds == y))


# ---------------------------------------------------------------------------
#region Top-level training run
# ---------------------------------------------------------------------------

def _parse_hidden_sizes(hidden_size_arg):
    """Parse hidden sizes: int or comma-separated string → list of ints."""
    if isinstance(hidden_size_arg, (list, tuple)):
        return list(hidden_size_arg)
    if isinstance(hidden_size_arg, int):
        return [hidden_size_arg]
    return [int(x.strip()) for x in str(hidden_size_arg).split(',') if x.strip()]


def train_one_epoch_bptt(x_train, y_train, model, optimizer, batch_size, n_classes, epoch,
                          clip=1.0, task="classification", temporal_loss=False):
    """
    Standard BPTT training for FPTTMinimalRNNAED.

    When temporal_loss=True (regression only): uses temporally weighted MSE over
    all window timesteps (weights linearly from 0→1), matching SNN training.
    Otherwise uses plain MSE on the final accumulated logit.

    Returns (avg_loss, train_acc).
    """
    model.train()
    device = next(model.parameters()).device
    n = x_train.shape[0]
    rng = np.random.default_rng(epoch)
    perm = rng.permutation(n)
    total_loss = 0.0
    n_batches  = 0

    all_preds   = []
    all_targets = []

    use_per_step = (task == "regression" and temporal_loss)
    label_dtype = torch.float32 if task == "regression" else torch.long

    for s in range(0, n, batch_size):
        idx = perm[s : s + batch_size]
        xb = torch.tensor(x_train[idx], dtype=torch.float32, device=device)
        yb = torch.tensor(y_train[idx], dtype=label_dtype, device=device)
        B  = yb.shape[0]

        hidden = model.init_hidden(B)
        optimizer.zero_grad()

        if use_per_step:
            # per_step: (B, T, C); yb: (B, C) → expand to (B, T, C) for windowed loss
            per_step, _ = model.forward_chunk(xb, hidden, t_offset=0, return_per_step=True)
            T = per_step.shape[1]
            weights = torch.linspace(0, 1, steps=T, device=device)  # (T,)
            yb_exp = yb.unsqueeze(1).expand_as(per_step)             # (B, T, C)
            sq_err = (per_step - yb_exp) ** 2                        # (B, T, C)
            loss = (sq_err * weights.view(1, T, 1)).mean()
            # Use last-step prediction for train_acc tracking
            logits = per_step[:, -1, :]
        else:
            logits, _ = model.forward_chunk(xb, hidden, t_offset=0)
            if task == "regression":
                loss = F.mse_loss(logits, yb)
            else:
                loss = F.cross_entropy(logits, yb)

        loss.backward()

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches  += 1
        all_preds.append(logits.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    if task == "regression":
        ss_res = np.sum((targets - preds) ** 2, axis=0)
        ss_tot = np.sum((targets - targets.mean(axis=0, keepdims=True)) ** 2, axis=0)
        train_acc = float(np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0).mean())
    else:
        train_acc = float((preds.argmax(axis=1) == targets).mean())

    return avg_loss, train_acc


def measure_sparsity(model, x, batch_size=64):
    """
    Measure per-layer activations for FPTTMinimalRNNAED.
    Returns avg_act (list[float], one per hidden layer) — average total fired events
    (post top-k) emitted by each layer while processing one input sequence, i.e. the
    activation count per layer per prediction, and sparsity (list[float]) — avg_act /
    hidden_size per layer. Returns (None, None) for non-AED models.
    """
    if not isinstance(model, FPTTMinimalRNNAED):
        return None, None
    device = next(model.parameters()).device
    model.eval()
    nlayers = model.nlayers
    totals = [0.0] * nlayers
    n_batches = 0
    with torch.no_grad():
        for s in range(0, x.shape[0], batch_size):
            xb = torch.tensor(x[s:s + batch_size], dtype=torch.float32, device=device)
            hidden = model.init_hidden(xb.shape[0])
            _, _, layer_fired = model.forward_chunk(xb, hidden, t_offset=0, return_sparsity=True)
            for l in range(nlayers):
                totals[l] += layer_fired[l]
            n_batches += 1
    avg_act = [t / max(n_batches, 1) for t in totals]
    sparsity = [a / model.hidden_size for a in avg_act]
    return avg_act, sparsity


def train_fptt(
    x_train, y_train,
    hidden_size,
    n_classes,
    n_input_neurons,
    epochs,
    batch_size,
    lr,
    alpha,
    beta,
    lmbda,
    rho,
    PARTS,
    clip,
    seed,
    sync_rate,
    firing_nb,
    use_tanh,
    train_samples,
    warm_epochs,
    optim_name,
    no_reset=False,
    identity_hh=False,
    vanilla=False,
    model_type="rule",
    nlayers=1,
    dropout=0.0,
    device="cpu",
    manual_grad=False,
    accumulate_logits=False,
    avg_logits=False,
    bptt=False,
    x_val=None, y_val=None,
    x_test=None, y_test=None,
    save_path="",
    task="classification",
    dense_output_firing=False,
    stateful_eval=False,
    temporal_loss=False,
    weight_decay=0.0,
    load_checkpoint=None,
    eval_only=False,
):
    if train_samples > 0:
        # Subsample uniformly at random rather than taking the first N. Time-series
        # datasets (e.g., neural decoding) have ordered targets — the first N rows
        # of the train split can be a near-constant slice (y_std ≪ y_std_global),
        # which collapses the model to predicting that slice's mean.
        rng_sub = np.random.default_rng(seed)
        idx = rng_sub.choice(x_train.shape[0], train_samples, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]

    hidden_sizes = _parse_hidden_sizes(hidden_size)
    L = len(hidden_sizes)

    if model_type == "minimalrnn_aed":
        H = _parse_hidden_sizes(hidden_size)[0]
        model = FPTTMinimalRNNAED(
            n_input_neurons=n_input_neurons,
            hidden_size=H,
            n_classes=n_classes,
            sync_rate=sync_rate,
            firing_nb=firing_nb,
            dropout=dropout,
            nlayers=nlayers,
            dense_output_firing=dense_output_firing,
        )
    elif model_type in ("lstm", "rnn", "minimalrnn"):
        H = hidden_sizes[0]
        if len(hidden_sizes) > 1:
            print(f"Warning: {model_type.upper()} uses uniform hidden size. Using {H}. "
                  f"Use --nlayers to control depth.")
        if model_type == "lstm":
            model = FPTTLstmRNN(
                input_size=n_input_neurons,
                hidden_size=H,
                n_classes=n_classes,
                nlayers=nlayers,
                dropout=dropout,
            )
        elif model_type == "minimalrnn":
            model = FPTTMinimalRNN(
                input_size=n_input_neurons,
                hidden_size=H,
                n_classes=n_classes,
                nlayers=nlayers,
                dropout=dropout,
            )
        else:
            model = FPTTVanillaRNN(
                input_size=n_input_neurons,
                hidden_size=H,
                n_classes=n_classes,
                nlayers=nlayers,
                dropout=dropout,
            )
    else:
        # Build layer_sizes: (n_input_neurons, H0, H1, ..., H_{L-1}, C)
        layer_sizes = (n_input_neurons, *hidden_sizes, n_classes)
        weights = init_feedforward_weights(layer_sizes, seed)
        # weights[0] = (n_input_neurons, H0), ..., weights[L] = (H_{L-1}, C)
        w_ih = weights[0]
        w_out = weights[L]

        # Inter-layer weights: weights[1] .. weights[L-1]
        w_ll_list = [weights[i] for i in range(1, L)]

        # Recurrent weights: one per hidden layer
        if identity_hh:
            w_hh_list = [np.eye(hidden_sizes[l], dtype=np.float32) for l in range(L)]
        else:
            w_hh_list = [init_recurrent_weight(hidden_sizes[l], seed + l, gain=0.5) for l in range(L)]

        # Biases: one per hidden layer
        bias_list = [np.zeros(hidden_sizes[l], dtype=np.float32) for l in range(L)]

        model = FPTTRuleRNN(
            w_ih=w_ih, w_hh_list=w_hh_list, w_ll_list=w_ll_list,
            w_out=w_out, bias_list=bias_list,
            sync_rate=sync_rate, firing_nb=firing_nb, use_tanh=use_tanh,
            no_reset=no_reset, vanilla=vanilla,
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}  params={n_params:,}  device={device}")

    if load_checkpoint:
        ckpt = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint: {load_checkpoint}  (saved ep={ckpt.get('epoch','?')}  val={ckpt.get('val_acc','?')})")

    if eval_only:
        model.eval()
        val_acc = evaluate(x_val, y_val, model, batch_size=batch_size * 4, n_classes=n_classes,
                           accumulate_logits=accumulate_logits, PARTS=PARTS, avg_logits=avg_logits,
                           task=task, stateful=stateful_eval, per_step_last=temporal_loss) if x_val is not None else None
        test_acc = evaluate(x_test, y_test, model, batch_size=batch_size * 4, n_classes=n_classes,
                            accumulate_logits=accumulate_logits, PARTS=PARTS, avg_logits=avg_logits,
                            task=task, stateful=stateful_eval, per_step_last=temporal_loss) if x_test is not None else None
        avg_act, sparsity = measure_sparsity(model, x_test[:512] if x_test is not None else x_val[:512])
        parts = ["eval_only=True"]
        if val_acc is not None:
            parts.append(f"val_acc={val_acc:.4f}")
        if test_acc is not None:
            parts.append(f"test_acc={test_acc:.4f}")
        if avg_act is not None:
            parts.append("act=[" + ",".join(f"{a:.1f}" for a in avg_act) + "]"
                         "  sparsity=[" + ",".join(f"{s:.3f}" for s in sparsity) + "]")
        print("  ".join(parts))
        return model, []

    if optim_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)

    named_params = get_stats_named_params(model)

    estimate_class_distribution = torch.full(
        (n_classes, PARTS, n_classes), 1.0 / n_classes
    )

    logs = []
    best_val_acc = float("-inf")  # ensures the first epoch always sets a baseline
    for ep in range(1, epochs + 1):
        t0 = time.time()

        if bptt and model_type == "minimalrnn_aed":
            # Standard BPTT: no epoch reset, no oracle, full sequence in one pass.
            # train_acc comes from the training forward passes — no extra eval needed.
            avg_loss, train_acc = train_one_epoch_bptt(
                x_train, y_train,
                model, optimizer,
                batch_size=batch_size,
                n_classes=n_classes,
                epoch=ep,
                clip=clip,
                task=task,
                temporal_loss=temporal_loss,
            )
        elif model_type == "rule":
            reset_named_params(named_params)
            avg_loss = train_one_epoch_manual(
                x_train, y_train,
                model, optimizer, named_params,
                estimate_class_distribution,
                epoch=ep,
                PARTS=PARTS,
                alpha=alpha, beta=beta, lmbda=lmbda, rho=rho,
                batch_size=batch_size,
                clip=clip,
                n_classes=n_classes,
                warm_epochs=warm_epochs,
                accumulate_logits=accumulate_logits,
                avg_logits=avg_logits,
            )
        elif model_type == "minimalrnn" and manual_grad:
            reset_named_params(named_params)
            avg_loss = train_one_epoch_manual_minimalrnn(
                x_train, y_train,
                model, optimizer, named_params,
                estimate_class_distribution,
                epoch=ep,
                PARTS=PARTS,
                alpha=alpha, beta=beta, lmbda=lmbda, rho=rho,
                batch_size=batch_size,
                clip=clip,
                n_classes=n_classes,
                warm_epochs=warm_epochs,
                accumulate_logits=accumulate_logits,
                avg_logits=avg_logits,
            )
        else:
            # LSTM/RNN/MinimalRNN(autograd)/minimalrnn_aed(FPTT) use autograd training loop
            reset_named_params(named_params)
            avg_loss = train_one_epoch(
                x_train, y_train,
                model, optimizer, named_params,
                estimate_class_distribution,
                epoch=ep,
                PARTS=PARTS,
                alpha=alpha, beta=beta, lmbda=lmbda, rho=rho,
                batch_size=batch_size,
                clip=clip,
                n_classes=n_classes,
                warm_epochs=warm_epochs,
                accumulate_logits=accumulate_logits,
                avg_logits=avg_logits,
            )

        if not (bptt and model_type == "minimalrnn_aed"):
            # For non-BPTT paths train_acc isn't returned by the training loop,
            # so compute it here on the full training set.
            train_acc = evaluate(x_train, y_train, model,
                                 batch_size=batch_size * 4, n_classes=n_classes,
                                 accumulate_logits=accumulate_logits, PARTS=PARTS,
                                 avg_logits=avg_logits, task=task, stateful=stateful_eval,
                                 per_step_last=temporal_loss)
        val_acc = None
        if x_val is not None:
            val_acc = evaluate(x_val, y_val, model,
                               batch_size=batch_size * 4, n_classes=n_classes,
                               accumulate_logits=accumulate_logits, PARTS=PARTS,
                               avg_logits=avg_logits, task=task, stateful=stateful_eval,
                               per_step_last=temporal_loss)
        test_acc = None
        if x_test is not None:
            test_acc = evaluate(x_test, y_test, model,
                                batch_size=batch_size * 4, n_classes=n_classes,
                                accumulate_logits=accumulate_logits, PARTS=PARTS,
                                avg_logits=avg_logits, task=task, stateful=stateful_eval,
                                per_step_last=temporal_loss)
        # Measure sparsity for AED models
        avg_act, sparsity = measure_sparsity(
            model,
            x_test[:512] if x_test is not None else (x_val[:512] if x_val is not None else x_train[:512]),
        )

        dt = time.time() - t0
        logs.append((ep, avg_loss, train_acc, val_acc, test_acc, dt))
        parts = [f"epoch={ep}", f"loss={avg_loss:.6f}", f"train_acc={train_acc:.4f}"]
        if val_acc is not None:
            parts.append(f"val_acc={val_acc:.4f}")
        if test_acc is not None:
            parts.append(f"test_acc={test_acc:.4f}")
        if avg_act is not None:
            parts.append("act=[" + ",".join(f"{a:.1f}" for a in avg_act) + "]")
            parts.append("sparsity=[" + ",".join(f"{s:.3f}" for s in sparsity) + "]")
        parts.append(f"time={dt:.1f}s")
        print("  ".join(parts))

        if save_path and val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": ep, "val_acc": val_acc, "test_acc": test_acc,
                        "state_dict": model.state_dict()},
                       f"{save_path}_best.pt")

    if save_path:
        torch.save({"epoch": epochs, "state_dict": model.state_dict()},
                   f"{save_path}_final.pt")

    return model, logs


# ---------------------------------------------------------------------------
#region Smoke test  (uses synthetic T=40 data — fast, no MNIST loading needed)
# ---------------------------------------------------------------------------

def _make_synthetic_events(rng, N, T, n_input_neurons):
    """Create synthetic (neuron_idx, value) event data for testing."""
    idx = rng.integers(0, n_input_neurons, (N, T)).astype(np.float32)
    val = rng.normal(0, 1, (N, T)).astype(np.float32)
    return np.stack([idx, val], axis=-1)  # (N, T, 2)


def smoke_test(data_dir=""):
    """Quick correctness check using synthetic event data."""
    rng = np.random.default_rng(42)
    N, T, C = 400, 40, 4
    n_input_neurons = 20

    x_train = _make_synthetic_events(rng, N, T, n_input_neurons)
    y_train = rng.integers(0, C, N, dtype=np.int64)
    x_test  = _make_synthetic_events(rng, 80, T, n_input_neurons)
    y_test  = rng.integers(0, C, 80, dtype=np.int64)

    configs = [
        {"hidden_size": 32,      "label": "1 hidden layer (H=32)"},
        {"hidden_size": [32, 16], "label": "2 hidden layers (32, 16)"},
    ]

    for cfg in configs:
        print("=" * 60)
        print(f"Smoke test: 5 epochs, synthetic events — {cfg['label']}")
        print("=" * 60)

        _, logs = train_fptt(
            x_train=x_train, y_train=y_train,
            x_test=x_test,   y_test=y_test,
            hidden_size=cfg['hidden_size'],
            n_classes=C,
            n_input_neurons=n_input_neurons,
            epochs=5,
            batch_size=64,
            lr=1e-3,
            alpha=0.1,
            beta=0.5,
            lmbda=2.0,
            rho=0.0,
            PARTS=4,
            clip=1.0,
            seed=42,
            sync_rate=1,
            firing_nb=10000,
            use_tanh=False,
            train_samples=0,
            warm_epochs=1,
            optim_name="adam",
        )

        first_loss = logs[0][1]
        last_loss  = logs[-1][1]
        last_train = logs[-1][2]

        print(f"\nFirst epoch loss : {first_loss:.6f}")
        print(f"Last  epoch loss : {last_loss:.6f}")
        print(f"Last  train  acc : {last_train:.4f}")

        assert last_loss < first_loss, (
            f"Surrogate loss did not decrease: {first_loss:.6f} -> {last_loss:.6f}"
        )
        assert last_train > 0.10, f"Train accuracy below chance: {last_train:.4f}"
        print(f"Smoke test PASSED — {cfg['label']}.\n")

    return logs


def run_gradient_comparison():
    """
    Compare autograd vs manual gradients on synthetic event data with multiple
    configurations (with/without tanh, different firing_nb, 1 and 2 layers).
    """
    print("=" * 60)
    print("Gradient comparison: autograd vs manual")
    print("=" * 60)

    rng = np.random.default_rng(42)
    B, T, C = 8, 20, 4
    n_input_neurons = 10
    PARTS = 4
    chunk_len = T // PARTS

    configs = [
        {"use_tanh": False, "firing_nb": 10000, "sync_rate": 1, "hidden_sizes": [16],     "label": "1-layer, no tanh, dense"},
        {"use_tanh": True,  "firing_nb": 10000, "sync_rate": 1, "hidden_sizes": [16],     "label": "1-layer, tanh, dense"},
        {"use_tanh": True,  "firing_nb": 4,     "sync_rate": 1, "hidden_sizes": [16],     "label": "1-layer, tanh, fnb=4"},
        {"use_tanh": False, "firing_nb": 4,     "sync_rate": 2, "hidden_sizes": [16],     "label": "1-layer, no tanh, fnb=4, sr=2"},
        {"use_tanh": False, "firing_nb": 10000, "sync_rate": 1, "hidden_sizes": [16, 12], "label": "2-layer, no tanh, dense"},
        {"use_tanh": True,  "firing_nb": 10000, "sync_rate": 1, "hidden_sizes": [16, 12], "label": "2-layer, tanh, dense"},
        {"use_tanh": True,  "firing_nb": 4,     "sync_rate": 1, "hidden_sizes": [16, 12], "label": "2-layer, tanh, fnb=4"},
        {"use_tanh": False, "firing_nb": 10000, "sync_rate": 1, "hidden_sizes": [12, 10, 8], "label": "3-layer, no tanh, dense"},
    ]

    all_pass = True
    for cfg in configs:
        print(f"\n--- Config: {cfg['label']} ---")
        hidden_sizes = cfg['hidden_sizes']
        L = len(hidden_sizes)

        layer_sizes = (n_input_neurons, *hidden_sizes, C)
        weights = init_feedforward_weights(layer_sizes, seed=42)
        w_ih = weights[0]
        w_out = weights[L]
        w_ll_list = [weights[i] for i in range(1, L)]
        w_hh_list = [init_recurrent_weight(hidden_sizes[l], seed=42 + l, gain=0.5) for l in range(L)]
        bias_list = [rng.normal(0, 0.01, hidden_sizes[l]).astype(np.float32) for l in range(L)]

        model = FPTTRuleRNN(
            w_ih=w_ih, w_hh_list=w_hh_list, w_ll_list=w_ll_list,
            w_out=w_out, bias_list=bias_list,
            sync_rate=cfg['sync_rate'], firing_nb=cfg['firing_nb'],
            use_tanh=cfg['use_tanh'],
        )

        # Synthetic event data: (B, T, 2) with (neuron_idx, value)
        x_events = _make_synthetic_events(rng, B, T, n_input_neurons)
        x = torch.tensor(x_events, dtype=torch.float32)
        y = torch.tensor(rng.integers(0, C, B), dtype=torch.long)

        for p in range(PARTS):
            start = p * chunk_len
            x_chunk = x[:, start : start + chunk_len, :]
            hidden = model.init_hidden(B)
            with torch.no_grad():
                for pp in range(p):
                    s = pp * chunk_len
                    _, hidden = model.forward_chunk(x[:, s : s + chunk_len, :], hidden, t_offset=s)
                    hidden = [(z.detach(), o.detach()) for z, o in hidden]

            beta_p = (p + 1) / PARTS
            oracle_prob = torch.full((B, C), 1.0 / C)

            results = compare_gradients(
                model, x_chunk, hidden, y, oracle_prob, beta_p, C
            )

            ok = all(v < 1e-5 for v in results.values())
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            parts = [f"logits={results['logits_diff']:.2e}",
                     f"w_ih={results['w_ih']:.2e}",
                     f"w_out={results['w_out']:.2e}"]
            for l in range(L):
                parts.append(f"w_hh.{l}={results[f'w_hh.{l}']:.2e}")
                parts.append(f"bias.{l}={results[f'bias.{l}']:.2e}")
            for l in range(L - 1):
                parts.append(f"w_ll.{l}={results[f'w_ll.{l}']:.2e}")
            print(f"  chunk {p}: {status}  " + "  ".join(parts))

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL GRADIENT COMPARISONS PASSED.")
    else:
        print("SOME COMPARISONS FAILED — check output above.")
    print("=" * 60)


# ---------------------------------------------------------------------------
#region CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FPTT training for the custom async RNN on MNIST"
    )
    parser.add_argument("--data-dir",       type=str,   default="")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--hidden-size",    type=str,   default="128",
                        help="Hidden layer sizes, comma-separated for multi-layer (e.g. '128' or '128,64').")
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch-size",     type=int,   default=128)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--alpha",          type=float, default=0.1,
                        help="Consensus regularisation strength.")
    parser.add_argument("--beta",           type=float, default=0.5,
                        help="Running-average decay for sm update.")
    parser.add_argument("--lmbda",          type=float, default=2.0,
                        help="Scale for the quadratic consensus penalty.")
    parser.add_argument("--rho",            type=float, default=0.0,
                        help="Coefficient on the lm·param term.")
    parser.add_argument("--parts",          type=int,   default=10,
                        help="Number of chunks to split the sequence into.")
    parser.add_argument("--clip",           type=float, default=1.0,
                        help="Gradient clipping (0 = disabled).")
    parser.add_argument("--warm-epochs",    type=int,   default=1,
                        help="Epochs to use uniform oracle before learned distribution.")
    parser.add_argument("--sync-rate",      type=int,   default=1)
    parser.add_argument("--firing-nb",      type=int,   default=10000)
    parser.add_argument("--use-tanh",       action="store_true", default=False)
    parser.add_argument("--no-reset",       action="store_true", default=False,
                        help="Disable the -o_prev soft reset in the recurrence.")
    parser.add_argument("--identity-hh",   action="store_true", default=False,
                        help="Initialize recurrent weight matrices to identity.")
    parser.add_argument("--vanilla",       action="store_true", default=False,
                        help="Use vanilla RNN: h_t = tanh(W_ih x_t + W_hh h_{t-1} + b). No relu, no sync, no topk.")
    parser.add_argument("--train-samples",  type=int,   default=0,
                        help="Number of training samples; 0 = full dataset.")
    parser.add_argument("--optim",          type=str,   default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--dataset",        type=str,   default="mnist",
                        choices=["mnist", "smnist", "shd", "neural_decoding"],
                        help="Dataset: 'mnist', 'smnist', 'shd' (Spiking Heidelberg Digits, 20 classes), "
                             "or 'neural_decoding' (primate reaching, regression onto 2-D velocity).")
    parser.add_argument("--filename",       type=str, default="indy_20160622_01.mat",
                        help="Primate-reaching session filename (only for --dataset neural_decoding).")
    parser.add_argument("--window",         type=int, default=50,
                        help="Window length in timesteps for neural decoding.")
    parser.add_argument("--collapse-units", action="store_true", default=False,
                        help="Collapse spike unit slots per channel (neural decoding).")
    parser.add_argument("--preserve-exact-times", action="store_true", default=False,
                        help="Use raw (timestamp, channel) events instead of binned counts.")
    parser.add_argument("--val-samples",  type=int, default=0,
                        help="Cap val set to first N samples (0 = all). Used for eval speed.")
    parser.add_argument("--test-samples", type=int, default=0,
                        help="Cap test set to first N samples (0 = all). Used for eval speed.")
    parser.add_argument("--smoke-test",     action="store_true", default=False,
                        help="Run the built-in smoke test and exit.")
    parser.add_argument("--compare-grads",  action="store_true", default=False,
                        help="Run autograd vs manual gradient comparison and exit.")
    parser.add_argument("--model",          type=str,   default="rule",
                        choices=["rule", "lstm", "rnn", "minimalrnn", "minimalrnn_aed"],
                        help="Model type: 'rule' (custom AED RNN), 'lstm' (LSTM), 'rnn' (vanilla Elman), 'minimalrnn' (Chen 2017), or 'minimalrnn_aed' (MinimalRNN with AED event input + ReLU output + logit accumulation, matching async_RNN_fptt_mpi.py).")
    parser.add_argument("--device",         type=str,   default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto' (use cuda if available).")
    parser.add_argument("--nlayers",        type=int,   default=1,
                        help="Number of LSTM layers (only used with --model lstm).")
    parser.add_argument("--dropout",        type=float, default=0.0,
                        help="Dropout between LSTM layers (only used with --model lstm, nlayers>1).")
    parser.add_argument("--manual-grad",   action="store_true",
                        help="Use manual gradient computation instead of autograd (minimalrnn only).")
    parser.add_argument("--accumulate-logits", action="store_true", default=False,
                        help="Accumulate logits across FPTT chunks instead of resetting each chunk.")
    parser.add_argument("--avg-logits", action="store_true", default=False,
                        help="When accumulating logits, average by chunk count (prevents magnitude growth).")
    parser.add_argument("--bptt",       action="store_true", default=False,
                        help="Use standard BPTT instead of FPTT for minimalrnn_aed: full-sequence forward "
                             "pass, single CE loss, no oracle, no epoch reset.")
    parser.add_argument("--save-path",  type=str, default="",
                        help="If set, save best-val and final model weights to <save_path>_best.pt / _final.pt.")
    parser.add_argument("--dense-output-firing", action="store_true", default=False,
                        help="Send the FULL dense ReLU(h_last) into the logit projection at every step "
                             "instead of top-k firing. Per-event semantics BETWEEN hidden layers is unaffected.")
    parser.add_argument("--stateful-eval", action="store_true", default=False,
                        help="Eval with batch=1 and hidden state carried over between consecutive samples "
                             "(useful for time-series like neural decoding so the RNN sees long-range context).")
    parser.add_argument("--temporal-loss", action="store_true", default=False,
                        help="Use temporally weighted MSE over all window steps (weight linearly 0→1), "
                             "matching SNN training. Regression only.")
    parser.add_argument("--weight-decay",  type=float, default=0.0,
                        help="Weight decay (L2 regularisation) for Adam/SGD optimizer.")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to a .pt checkpoint to load before training/eval.")
    parser.add_argument("--eval-only", action="store_true", default=False,
                        help="Skip training; just evaluate the loaded checkpoint on val+test.")
    args = parser.parse_args()

    if args.compare_grads:
        run_gradient_comparison()
        return

    if args.smoke_test:
        smoke_test(data_dir=args.data_dir)
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.dataset == "shd":
        x_train, y_train, x_val, y_val, x_test, y_test, n_input_neurons = load_shd_arrays(
            args.batch_size, args.data_dir
        )
        n_classes = 20
        task = "classification"
    elif args.dataset == "neural_decoding":
        x_train, y_train, x_val, y_val, x_test, y_test, n_input_neurons = load_neural_decoding_arrays(
            args.batch_size, args.data_dir, filename=args.filename, window=args.window,
            collapse_units=args.collapse_units,
            preserve_exact_times=args.preserve_exact_times,
        )
        n_classes = y_train.shape[1]  # regression output dim, typically 2
        task = "regression"
        # Cap val/test for eval speed under per-event semantics. R² is stable on
        # ~2000 samples; full set inflates per-epoch eval cost without changing
        # the metric meaningfully.
        if args.val_samples > 0:
            x_val, y_val = x_val[:args.val_samples], y_val[:args.val_samples]
        if args.test_samples > 0:
            x_test, y_test = x_test[:args.test_samples], y_test[:args.test_samples]
    else:
        x_train, y_train, x_val, y_val, x_test, y_test, n_input_neurons = load_mnist_arrays(
            args.batch_size, args.data_dir, dataset=args.dataset
        )
        n_classes = 10
        task = "classification"

    hidden_sizes = _parse_hidden_sizes(args.hidden_size)

    print(
        f"FPTT training  dataset={args.dataset}  model={args.model}\n"
        f"  seed={args.seed}  hidden={hidden_sizes}  n_input_neurons={n_input_neurons}\n"
        f"  x_train={x_train.shape}  n_classes={n_classes}  epochs={args.epochs}\n"
        f"  batch={args.batch_size}  lr={args.lr}  alpha={args.alpha}  beta={args.beta}\n"
        f"  lmbda={args.lmbda}  rho={args.rho}  parts={args.parts}  clip={args.clip}\n"
        f"  sync_rate={args.sync_rate}  firing_nb={args.firing_nb}  use_tanh={args.use_tanh}\n"
        f"  no_reset={args.no_reset}  identity_hh={args.identity_hh}  vanilla={args.vanilla}\n"
        f"  train_samples={args.train_samples}  optim={args.optim}\n"
        f"  nlayers={args.nlayers}  dropout={args.dropout}\n"
        f"  accumulate_logits={args.accumulate_logits}  avg_logits={args.avg_logits}  bptt={args.bptt}"
    )

    train_fptt(
        x_train=x_train,  y_train=y_train,
        x_val=x_val,      y_val=y_val,
        x_test=x_test,    y_test=y_test,
        hidden_size=hidden_sizes,
        n_classes=n_classes,
        n_input_neurons=n_input_neurons,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        lmbda=args.lmbda,
        rho=args.rho,
        PARTS=args.parts,
        clip=args.clip,
        seed=args.seed,
        sync_rate=args.sync_rate,
        firing_nb=args.firing_nb,
        use_tanh=args.use_tanh,
        train_samples=args.train_samples,
        warm_epochs=args.warm_epochs,
        optim_name=args.optim,
        no_reset=args.no_reset,
        identity_hh=args.identity_hh,
        vanilla=args.vanilla,
        model_type=args.model,
        nlayers=args.nlayers,
        dropout=args.dropout,
        device=device,
        manual_grad=args.manual_grad,
        accumulate_logits=args.accumulate_logits,
        avg_logits=args.avg_logits,
        bptt=args.bptt,
        save_path=args.save_path,
        task=task,
        dense_output_firing=args.dense_output_firing,
        stateful_eval=args.stateful_eval,
        temporal_loss=args.temporal_loss,
        weight_decay=args.weight_decay,
        load_checkpoint=args.load_checkpoint,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()

# Quick commands:
# python async_RNN_fptt.py --smoke-test
# python async_RNN_fptt.py --compare-grads
# python async_RNN_fptt.py --dataset mnist --epochs 10 --hidden-size 128 --train-samples 3000 --parts 10
# python async_RNN_fptt.py --dataset mnist --epochs 10 --hidden-size 128,64 --train-samples 3000 --parts 10
# python async_RNN_fptt.py --dataset smnist --epochs 10 --hidden-size 128 --train-samples 3000 --parts 10
# python async_RNN_fptt.py --dataset smnist --epochs 20 --hidden-size 128 --use-tanh --firing-nb 1 --train-samples 0
