"""Evaluate the stateful-eval option on best.pt checkpoints from the full ND sweep."""
import torch
import numpy as np
import time
from async_RNN_fptt import load_neural_decoding_arrays, FPTTMinimalRNNAED, evaluate

SESSIONS = [
    # tag, fname, hidden, collapse, exact, SNN R²
    ('indy622', 'indy_20160622_01.mat', 256, False, True,  0.6914),
    ('loco301', 'loco_20170301_05.mat', 512, False, True,  0.6247),
    ('loco210', 'loco_20170210_03.mat', 256, False, True,  0.5453),
]


def main():
    print(f"{'Session':12s} {'ep':>3s} {'Stateless R²':>14s} {'Stateful R²':>13s}  "
          f"{'Δ stateful':>11s} {'SNN R²':>8s}  {'Δ vs SNN (stateful)':>20s}")
    print("-" * 100)

    for tag, fname, hidden, collapse, exact, snn_r2 in SESSIONS:
        ckpt = torch.load(f'results/weights/nd_{tag}_full_dense_30ep_best.pt',
                          map_location='cpu', weights_only=False)
        x_tr, y_tr, x_v, y_v, x_te, y_te, n_in = load_neural_decoding_arrays(
            batch_size=128, data_dir='', filename=fname, window=50,
            collapse_units=collapse, preserve_exact_times=exact,
        )
        m = FPTTMinimalRNNAED(n_input_neurons=n_in, hidden_size=hidden, n_classes=2,
                              firing_nb=10, dropout=0.0, nlayers=1,
                              dense_output_firing=True).cuda()
        m.load_state_dict(ckpt['state_dict'])

        # Stateless eval (full test)
        t0 = time.time()
        r2_stateless = evaluate(x_te, y_te, m, batch_size=256, n_classes=2,
                                task='regression', stateful=False)
        t_stateless = time.time() - t0

        # Stateful eval — batched streams
        t0 = time.time()
        r2_stateful = evaluate(x_te, y_te, m, batch_size=64, n_classes=2,
                               task='regression', stateful=True)
        t_stateful = time.time() - t0

        print(f"{tag:12s} {ckpt['epoch']:>3d} {r2_stateless:>14.4f} {r2_stateful:>13.4f}  "
              f"{r2_stateful - r2_stateless:>+11.4f} {snn_r2:>8.4f}  {r2_stateful - snn_r2:>+20.4f}")
        print(f"  (timings: stateless {t_stateless:.1f}s, stateful {t_stateful:.1f}s)")


if __name__ == "__main__":
    main()
