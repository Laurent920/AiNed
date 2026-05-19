import jax.numpy as jnp
import jax

# region REGRESSION HELPERS
def r2_accumulate(sums, y_true: jnp.ndarray, y_pred: jnp.ndarray):
    """
    Accumulate sufficient statistics for epoch-level R2.
    sums = (sum_y, sum_y2, sum_res, count)
    """
    sum_y, sum_y2, sum_res, count = sums

    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]

    valid_mask = jnp.all(y_true != -1, axis=-1)
    valid_mask_f = valid_mask.astype(y_true.dtype)
    valid_count = jnp.sum(valid_mask_f)

    sum_y = sum_y + jnp.sum(y_true * valid_mask_f[:, None], axis=0)
    sum_y2 = sum_y2 + jnp.sum((y_true ** 2) * valid_mask_f[:, None], axis=0)
    sum_res = sum_res + jnp.sum(((y_true - y_pred) ** 2) * valid_mask_f[:, None], axis=0)
    count = count + valid_count

    return sum_y, sum_y2, sum_res, count


def r2_from_sums(sum_y, sum_y2, sum_res, count):
    """
    Compute mean R2 across dimensions from accumulated sums.
    Mirrors literature-style computation over the full evaluation set.
    """
    count = jnp.maximum(count, 1.0)
    mean_y = sum_y / count
    ss_tot = sum_y2 - count * (mean_y ** 2)
    r2 = jnp.where(ss_tot > 0, 1.0 - (sum_res / ss_tot), 0.0)
    return jnp.mean(r2)


def mse_from_sum_res(sum_res, count):
    """
    Compute epoch-level mean squared error from accumulated squared residuals.
    """
    count = jnp.maximum(count, 1.0)
    mse_per_dim = sum_res / count
    return jnp.mean(mse_per_dim)


@jax.jit
def loss_func_regression(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Mean squared error over valid (non-padded) samples.
    Assumes padded labels are filled with -1.
    """
    if targets.ndim == 1:
        targets = targets[:, None]
        preds = preds[:, None]

    valid_mask = jnp.all(targets != -1, axis=-1)
    valid_mask_f = valid_mask.astype(preds.dtype)
    valid_count = jnp.sum(valid_mask_f)

    squared = (preds - targets) ** 2
    loss = jnp.sum(squared * valid_mask_f[:, None]) / (jnp.maximum(valid_count, 1.0) * preds.shape[-1])
    return loss
