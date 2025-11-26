"""Inference script for SVGP model using saved Shoji weights.

This script rebuilds the model architecture with the custom ARDRBFKernelLayer
and ShojiNaturalGradient optimizer, loads the trained weights, and performs
batched prediction to avoid OOM issues.
"""

import os
import math
import numpy as np
import pandas as pd
import scipy.io as sio

# Ensure legacy Keras APIs stay enabled (must be set before importing tf)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras

# Aliases
# -----------------------------------------------------------------------------
tfk = tfp.math.psd_kernels
tfpl = tfp.layers
ki = keras.initializers


# -----------------------------------------------------------------------------
# Optimizer: ShojiNaturalGradient (Theorem B' with Strict Acute Angle Constraint)
# -----------------------------------------------------------------------------
class ShojiNaturalGradient(keras.optimizers.legacy.Optimizer):
    """Shoji 等（2024）Theorem B' 的 O(D) 版本，并加入严格锐角约束。"""

    def __init__(
        self,
        learning_rate=0.001,
        momentum=0.9,
        angle_threshold=0.1,
        global_clipnorm=None,
        name="ShojiNaturalGradient",
        **kwargs,
    ):
        super().__init__(
            name=name,
            clipnorm=None,
            global_clipnorm=global_clipnorm,
            **kwargs,
        )
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("momentum", momentum)
        self.angle_threshold = angle_threshold

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "g", initializer="zeros")

    def _resource_apply_dense(self, grad, var):
        lr = self._decayed_lr(var.dtype)
        beta = tf.cast(self._get_hyper("momentum", var.dtype), var.dtype)
        angle_th = tf.cast(self.angle_threshold, var.dtype)
        eps_div = tf.cast(1e-12, var.dtype)

        g_prev = self.get_slot(var, "g")
        y = -grad
        g_cand = beta * g_prev + (1.0 - beta) * y

        cos_sim = tf.reduce_sum(y * g_cand) / (
            tf.norm(y) * tf.norm(g_cand) + eps_div
        )

        def reset_dir():
            return y

        def keep_dir():
            return g_cand

        # 硬重置：当夹角过钝（cos 过小）时回退到 y_t，避免 Shoji 度量奇异。
        g_new = tf.cond(cos_sim < angle_th, reset_dir, keep_dir)

        g_update = g_prev.assign(g_new, use_locking=self._use_locking)
        var_update = var.assign_add(lr * g_new, use_locking=self._use_locking)
        return tf.group(var_update, g_update)

    def _resource_apply_sparse(self, grad, var, indices=None):
        dense_grad = tf.convert_to_tensor(grad)
        return self._resource_apply_dense(dense_grad, var)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "angle_threshold": self.angle_threshold,
        }


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
USE_FLOAT64 = True
DTYPE = tf.float64 if USE_FLOAT64 else tf.float32
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.backend.set_floatx("float64" if USE_FLOAT64 else "float32")

LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 256
JITTER = 1e-3
NU = 0.30
WARP_DIM = 2
WARP_EPS = 1e-6
NOISE0 = 1e-1

PHYS_TARGET_LAM = 1000
PHYS_WARMUP_STEPS = 2000
PHYS_FORM = "logratio"
J_FLOOR_PERCENT = 1.0
J_ABS_FLOOR = 1e-30


# -----------------------------------------------------------------------------
# Utilities: composite Simpson weights & E_eq calculation
# -----------------------------------------------------------------------------
def _simpson_weights(m: int) -> np.ndarray:
    assert m % 2 == 0 and m > 0
    w = np.ones(m + 1, dtype=np.float64)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w


def preprocess_window_eq(X_np: np.ndarray, m=20, delta_ratio=0.2):
    """基于内侧裂尖对称窗口的 E_eq 复合 Simpson 平均。X: [Ro, R_n, crack_n, theta_deg, n_FGM]"""
    N = X_np.shape[0]
    Ro, Rn, crack_n, theta_deg, n_FGM = [X_np[:, i].astype(np.float64) for i in range(5)]

    Ri = Rn * Ro
    Rp = Ri
    Dp = Ro - Ri
    a = crack_n * Dp

    if np.any(Dp <= 0):
        bad = np.where(Dp <= 0)[0]
        raise ValueError(f"存在 Dp<=0（壁厚非正），样本行: {bad[:10]} ...")
    if np.any((a <= 0) | (a >= Dp)):
        bad = np.where((a <= 0) | (a >= Dp))[0]
        raise ValueError(f"裂纹深度 a 不在 (0, Dp) 内，样本行: {bad[:10]} ...")

    r_tip = Rp + a
    Delta = np.maximum(1e-12, delta_ratio * a)

    left_half = np.minimum(Delta, r_tip - Rp)
    right_half = np.minimum(Delta, Rp + Dp - r_tip)
    half = np.minimum(left_half, right_half)
    left, right = r_tip - half, r_tip + half
    Delta_eff = right - left

    if np.any(Delta_eff <= 0):
        bad = np.where(Delta_eff <= 0)[0]
        raise ValueError(f"窗口宽度为零，检查 delta_ratio/几何，样本行: {bad[:10]} ...")

    h = Delta_eff / m
    w = _simpson_weights(m)

    Eeq = np.zeros(N, dtype=np.float64)
    E0 = 214e9
    dE = 166e9

    for i in range(N):
        r_i = left[i] + h[i] * np.arange(m + 1, dtype=np.float64)
        r_i = np.clip(r_i, Rp[i], Rp[i] + Dp[i])

        xi = 1.0 - (r_i - Rp[i]) / Dp[i]
        Ei = E0 + dE * np.power(xi, n_FGM[i])

        integral = (h[i] / 3.0) * np.dot(w, Ei)
        Eeq[i] = integral / Delta_eff[i]

    aux = dict(
        Ro=Ro,
        Ri=Ri,
        Rp=Rp,
        Dp=Dp,
        a=a,
        r_tip=r_tip,
        left=left,
        right=right,
        h=h,
        m=m,
        delta_ratio=delta_ratio,
    )
    return Eeq.astype(np.float32), aux


# -----------------------------------------------------------------------------
# Data validation & preprocessing
# -----------------------------------------------------------------------------
def validate_and_prepare(
    X_raw: np.ndarray, Y_raw: np.ndarray | None, stats: dict | None = None
):
    """输入校验 + warp + 标准化。

    先对 crack_n 开方，并将角度拆解为正弦/余弦，再对展开后的特征做 z-score。
    当 ``stats`` 提供时，使用训练集的均值/方差进行归一化，避免推理时的数据泄漏。
    """
    if not isinstance(X_raw, np.ndarray):
        raise TypeError("X_raw 必须是 numpy.ndarray")
    if Y_raw is not None and not isinstance(Y_raw, np.ndarray):
        raise TypeError("Y_raw 必须是 numpy.ndarray 或 None")

    if X_raw.ndim != 2 or X_raw.shape[1] != 5:
        raise ValueError(f"X 形状应为 (N,5)，当前 {X_raw.shape}")

    if Y_raw is not None:
        if Y_raw.ndim == 1:
            Y_raw = Y_raw.reshape(-1, 1)
        elif Y_raw.ndim == 2 and Y_raw.shape[0] == 1 and Y_raw.shape[1] > 1:
            Y_raw = Y_raw.T
        if Y_raw.ndim != 2:
            raise ValueError(f"Y 形状应为二维 (N,C)，当前 {Y_raw.shape}")

    X_finite = np.all(np.isfinite(X_raw), axis=1)
    Y_finite = (
        np.ones_like(X_finite, dtype=bool)
        if Y_raw is None
        else np.all(np.isfinite(Y_raw), axis=1)
    )
    keep_mask = X_finite & Y_finite
    if not keep_mask.all():
        bad = np.where(~keep_mask)[0]
        print(f"警告：移除含 NaN/Inf 样本 {bad.shape[0]} 行，示例索引: {bad[:10]}")
    Xc = X_raw[keep_mask].astype(np.float64)
    Yc = None if Y_raw is None else Y_raw[keep_mask].astype(np.float64)

    Ro = Xc[:, 0]
    Rn = Xc[:, 1]
    crack_n = Xc[:, 2]
    if np.any(Ro <= 0):
        bad = np.where(Ro <= 0)[0]
        raise ValueError(f"存在 Ro<=0，样本行: {bad[:10]} ...")
    if np.any((Rn <= 0) | (Rn >= 1)):
        bad = np.where((Rn <= 0) | (Rn >= 1))[0]
        raise ValueError(f"R_n 必须在 (0,1)，样本行: {bad[:10]} ...")
    if np.any((crack_n <= 0) | (crack_n >= 1)):
        bad = np.where((crack_n <= 0) | (crack_n >= 1))[0]
        raise ValueError(f"crack_n 必须在 (0,1)，样本行: {bad[:10]} ...")

    Ri = Rn * Ro
    Dp = Ro - Ri
    a = crack_n * Dp
    if np.any(Dp <= 0):
        bad = np.where(Dp <= 0)[0]
        raise ValueError(f"存在 Dp<=0（壁厚非正），样本行: {bad[:10]} ...")
    if np.any((a <= 0) | (a >= Dp)):
        bad = np.where((a <= 0) | (a >= Dp))[0]
        raise ValueError(f"a 不在 (0,Dp)，样本行: {bad[:10]} ...")

    theta_deg = Xc[:, 3]
    theta_rad = np.deg2rad(theta_deg)

    crack_n_warp = np.sqrt(np.maximum(0.0, crack_n) + WARP_EPS)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)

    X_expanded = np.column_stack([Ro, Rn, crack_n_warp, sin_theta, cos_theta, Xc[:, 4]])

    if stats is None:
        X_mean = X_expanded.mean(axis=0)
        X_std = X_expanded.std(axis=0)
        X_std[X_std == 0] = 1.0
    else:
        X_mean = np.asarray(stats["X_mean"], dtype=np.float64)
        X_std = np.asarray(stats["X_std"], dtype=np.float64)
    X_norm = (X_expanded - X_mean) / X_std

    if Yc is not None:
        if stats is None:
            Y_mean = Yc.mean(axis=0)
            Y_std = Yc.std(axis=0)
            Y_std[Y_std == 0] = 1.0
        else:
            Y_mean = np.asarray(stats["Y_mean"], dtype=np.float64)
            Y_std = np.asarray(stats["Y_std"], dtype=np.float64)
        Y_norm = (Yc - Y_mean) / Y_std
    else:
        Y_mean = None if stats is None else np.asarray(stats.get("Y_mean"), dtype=np.float64)
        Y_std = None if stats is None else np.asarray(stats.get("Y_std"), dtype=np.float64)
        Y_norm = None

    if Y_norm is not None:
        if X_norm.shape[0] != Y_norm.shape[0]:
            raise ValueError(f"X/Y 样本数不一致：{X_norm.shape[0]} vs {Y_norm.shape[0]}")
        if not np.isfinite(Y_norm).all():
            bad = np.argwhere(~np.isfinite(Y_norm))[:5]
            raise ValueError(f"Y_norm 含非有限值，位置示例: {bad}")
    if not np.isfinite(X_norm).all():
        bad = np.argwhere(~np.isfinite(X_norm))[:5]
        raise ValueError(f"X_norm 含非有限值，位置示例: {bad}")

    return (
        X_expanded.astype(np.float32),
        X_norm.astype(np.float32),
        X_mean.astype(np.float32),
        X_std.astype(np.float32),
        None if Y_norm is None else Y_norm.astype(np.float32),
        None if Y_mean is None else Y_mean.astype(np.float32),
        None if Y_std is None else Y_std.astype(np.float32),
        keep_mask,
    )


def _self_check(
    X_np_raw: np.ndarray,
    keep_mask_np: np.ndarray,
    X_norm_tf: tf.Tensor,
    X_mean: np.ndarray,
    X_std: np.ndarray,
):
    assert _simpson_weights(4).tolist() == [1.0, 4.0, 2.0, 4.0, 1.0]

    Xtoy = np.array(
        [[2.0, 0.5, 0.2, 0.0, 1.0], [3.0, 0.6, 0.3, 45.0, 2.0]], dtype=np.float32
    )
    _e_toy, _ = preprocess_window_eq(Xtoy, m=20, delta_ratio=0.2)
    assert _e_toy.shape == (2,) and np.all(_e_toy > 0)

    _tmp = X_np_raw[keep_mask_np].astype(np.float64)
    _theta_rad = np.deg2rad(_tmp[:, 3])
    _tmp_warp = np.column_stack(
        [
            _tmp[:, 0],
            _tmp[:, 1],
            np.sqrt(np.maximum(0.0, _tmp[:, WARP_DIM]) + WARP_EPS),
            np.sin(_theta_rad),
            np.cos(_theta_rad),
            _tmp[:, 4],
        ]
    )
    assert np.allclose(_tmp_warp[:, :2], _tmp[:, :2])
    assert np.all((_tmp_warp[:, 3:5] >= -1.0) & (_tmp_warp[:, 3:5] <= 1.0)).all()

    _norm_mean = tf.reduce_mean(X_norm_tf, axis=0).numpy()
    assert np.allclose(_norm_mean, np.zeros_like(_norm_mean), atol=1e-6)


def load_data_and_prepare(mat_path: str = "matlab_input.mat"):
    """读取 .mat -> 几何验证 -> 物理量计算 -> Tensor 化。"""
    mat = sio.loadmat(mat_path)
    X_np_raw = mat["data_input"]
    Y_np_raw = mat["combinedData"]

    if X_np_raw.ndim == 2:
        if X_np_raw.shape[-1] == 5:
            pass
        elif X_np_raw.shape[0] == 5:
            X_np_raw = X_np_raw.T
        else:
            X_np_raw = np.reshape(X_np_raw, (-1, 5), order="F")
    else:
        raise ValueError("data_input 维度异常，期望2D矩阵")

    Y_np_raw = np.atleast_2d(Y_np_raw)
    if Y_np_raw.shape[0] == 1 and Y_np_raw.shape[1] > 1:
        Y_np_raw = Y_np_raw.T

    (
        X_warp_raw,
        X_norm,
        X_mean,
        X_std,
        Y_norm,
        Y_mean,
        Y_std,
        keep_mask_np,
    ) = validate_and_prepare(X_np_raw, Y_np_raw)

    Eeq_np, aux = preprocess_window_eq(X_np_raw[keep_mask_np], m=20, delta_ratio=0.2)

    X_norm_tf = tf.convert_to_tensor(X_norm, dtype=DTYPE)
    Y_norm_tf = tf.convert_to_tensor(Y_norm, dtype=DTYPE)
    Eeq_tf = tf.convert_to_tensor(Eeq_np, dtype=DTYPE)
    Y_mean_tf = tf.convert_to_tensor(Y_mean, dtype=DTYPE)
    Y_std_tf = tf.convert_to_tensor(Y_std, dtype=DTYPE)

    _self_check(X_np_raw, keep_mask_np, X_norm_tf, X_mean, X_std)

    return dict(
        X_raw=X_np_raw[keep_mask_np],
        X_warp_raw=X_warp_raw,
        X_norm=X_norm,
        Y_norm=Y_norm,
        X_norm_tf=X_norm_tf,
        Y_norm_tf=Y_norm_tf,
        Eeq_tf=Eeq_tf,
        Y_mean_tf=Y_mean_tf,
        Y_std_tf=Y_std_tf,
        keep_mask=keep_mask_np,
        aux=aux,
        X_mean=X_mean,
        X_std=X_std,
        Y_mean=Y_mean,
        Y_std=Y_std,
    )


# -----------------------------------------------------------------------------
# ARD RBF kernel layer
# -----------------------------------------------------------------------------
class ARDRBFKernelLayer(keras.layers.Layer):
    def __init__(self, amplitude=0.5, length_scale_diag=None, input_dim=5, **kwargs):
        super().__init__(**kwargs)
        if length_scale_diag is None:
            length_scale_diag = np.ones([input_dim], np.float32)
        length_scale_diag = np.asarray(length_scale_diag, dtype=np.float32)
        if length_scale_diag.shape[0] != input_dim:
            raise ValueError(
                f"length_scale_diag shape {length_scale_diag.shape} 与 input_dim={input_dim} 不一致"
            )
        self._amp_unconstrained = self.add_weight(
            name="amplitude",
            shape=[],
            initializer=ki.Constant(np.log(np.expm1(amplitude))),
            dtype=DTYPE,
            trainable=True,
        )
        self._ls_unconstrained = self.add_weight(
            name="length_scale_diag",
            shape=[input_dim],
            initializer=ki.Constant(np.log(np.expm1(length_scale_diag))),
            dtype=DTYPE,
            trainable=True,
        )

    @property
    def kernel(self):
        amp = tf.nn.softplus(self._amp_unconstrained)
        ls = tf.nn.softplus(self._ls_unconstrained) + tf.cast(1e-12, DTYPE)
        base = tfk.ExponentiatedQuadratic(amplitude=amp, length_scale=1.0)
        ard = tfk.FeatureScaled(base, scale_diag=ls)
        return ard

    def call(self, inputs, **kwargs):
        return inputs


# -----------------------------------------------------------------------------
# Model building
# -----------------------------------------------------------------------------
class SVGPModel(keras.Model):
    def __init__(self, vgp_layer: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        self.vgp_layer = vgp_layer

    def call(self, inputs, training=None):
        return self.vgp_layer(inputs, training=training)


def _identity_with_shape(dist: tfp.distributions.Distribution):
    """Return distribution itself while supplying a TensorShape for TFP layer checks.

    TFP 0.23 inspects `value.shape` inside DistributionLambda even when we return a
    Distribution via `convert_to_tensor_fn`. Attaching a TensorShape derived from
    batch/event shapes bypasses the attribute error without touching sampling or
    gradients.
    """
    if not hasattr(dist, "shape"):
        dist.shape = dist.batch_shape.concatenate(dist.event_shape)
    if not hasattr(dist, "get_shape"):
        dist.get_shape = lambda: dist.shape
    return dist


def build_model(
    X_norm: np.ndarray, Y_norm: np.ndarray, total_steps: int | None = None
):
    """构建 SVGP 模型（显式返回分布对象，避免依赖 Keras 副作用）。"""
    if total_steps is None:
        batches_per_epoch = math.ceil(X_norm.shape[0] / BATCH_SIZE)
        total_steps = EPOCHS * batches_per_epoch
    total_steps = max(1, int(total_steps))

    num_inducing = int(min(1000, X_norm.shape[0]))
    D_out = int(Y_norm.shape[1])

    inducing_init = X_norm[
        np.random.choice(X_norm.shape[0], size=num_inducing, replace=False)
    ]
    inducing_init_multi = np.stack([inducing_init] * D_out, axis=0).astype(
        np.float64 if USE_FLOAT64 else np.float32
    )

    initial_scale = (
        np.eye(num_inducing, dtype=np.float64 if USE_FLOAT64 else np.float32)[None, ...]
        * 0.1
    )
    initial_scale = np.tile(initial_scale, (D_out, 1, 1))

    vgp_layer = tfpl.VariationalGaussianProcess(
        num_inducing_points=num_inducing,
        kernel_provider=ARDRBFKernelLayer(
            amplitude=0.5,
            length_scale_diag=np.array([2.0, 0.5, 0.5, 0.1, 0.1, 2.0], np.float32),
            input_dim=6,
        ),
        event_shape=(D_out,),
        inducing_index_points_initializer=ki.Constant(inducing_init_multi),
        unconstrained_observation_noise_variance_initializer=ki.Constant(
            np.log(np.expm1(NOISE0))
        ),
        variational_inducing_observations_scale_initializer=ki.Constant(initial_scale),
        jitter=JITTER,
        convert_to_tensor_fn=_identity_with_shape,
        name="SVGPLayer",
    )

    model = SVGPModel(vgp_layer)
    _ = model(tf.zeros([1, 6], dtype=DTYPE))  # 预构建变量，便于 summary/save_weights

    def negloglik(y_true, rv_pred):
        return -rv_pred.log_prob(y_true)

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=total_steps,
        alpha=0.01,
    )

    optimizer = ShojiNaturalGradient(
        learning_rate=lr_schedule,
        momentum=0.9,
        angle_threshold=0.1,
        global_clipnorm=1.0,
    )
    model.compile(optimizer=optimizer, loss=negloglik)
    return model


# -----------------------------------------------------------------------------
# Prediction helper
# -----------------------------------------------------------------------------
def predict_mean_batched(model, X_tf: tf.Tensor, batch_size: int = 8192):
    means = []
    N = X_tf.shape[0]
    for i in range(0, N, batch_size):
        rv = model(X_tf[i : i + batch_size], training=False)
        means.append(rv.mean().numpy())
    return np.vstack(means)


# -----------------------------------------------------------------------------
# Inference logic
# -----------------------------------------------------------------------------
def load_and_predict(
    weights_paths: tuple[str, str],
    X_new_raw: np.ndarray,
    train_stats: dict,
    template_data: dict | None = None,
):
    """Rebuild two single-output SVGP models and predict on ``X_new_raw``.

    The raw features (5 columns) are warped/normalized using training statistics.
    Each column-specific model is loaded separately, predictions are denormalized
    with the matching mean/std, and concatenated into a ``[N, 2]`` array.
    """
    if template_data is None:
        template_data = load_data_and_prepare()

    (
        _,
        X_norm_new,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = validate_and_prepare(X_new_raw, None, stats=train_stats)

    total_steps = max(
        1, EPOCHS * math.ceil(template_data["X_norm"].shape[0] / BATCH_SIZE)
    )
    X_tf = tf.convert_to_tensor(X_norm_new, dtype=DTYPE)

    preds_phys = []
    for idx, weight_path in enumerate(weights_paths):
        y_norm_template = template_data["Y_norm"][:, idx : idx + 1]
        model = build_model(template_data["X_norm"], y_norm_template, total_steps)
        model.load_weights(weight_path)

        preds_norm_col = predict_mean_batched(model, X_tf)
        mean_col = np.asarray(train_stats["Y_mean"], dtype=np.float64)[idx : idx + 1]
        std_col = np.asarray(train_stats["Y_std"], dtype=np.float64)[idx : idx + 1]
        preds_phys.append(preds_norm_col * std_col + mean_col)

    return np.hstack(preds_phys)


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data_and_prepare()
    weight_files = ("svgp_col0.h5", "svgp_col1.h5")

    train_stats = dict(
        X_mean=data["X_mean"],
        X_std=data["X_std"],
        Y_mean=data["Y_mean"],
        Y_std=data["Y_std"],
    )

    preds_physical = load_and_predict(
        weight_files, data["X_raw"], train_stats=train_stats, template_data=data
    )

    print("Predictive mean (first 10 rows, physical scale):")
    print(preds_physical[:10])

    df_out = pd.DataFrame(preds_physical, columns=["Predicted_KI", "Predicted_KII"])
    # 如果你想把输入 X 和预测 Y 拼在一起保存：
    # df_in = pd.DataFrame(data["X_norm"] * data["X_std"] + data["X_mean"], columns=["Ro", "Rn", "crack_n", "theta", "FGM_n"])
    # df_all = pd.concat([df_in, df_out], axis=1)
    # df_all.to_csv("all_predictions.csv", index=False)

    df_out.to_csv("all_predictions.csv", index=False)
    print("全部预测结果已保存至: all_predictions.csv")
