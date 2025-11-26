# =============================================================================
# NaN-safe SVGP + 物理正则（log-ratio + warmup + 批量预测）
# -----------------------------------------------------------------------------
# 目的：
# 1) 彻底区分“数据问题导致的 NaN”和“训练数值不稳定导致的 NaN”。
# 2) 解决你提到的“归一化能避免 NaN，但物理约束被削弱”的矛盾：
#    - 物理正则改为 **log-ratio** 形式（对比例数值稳定、且对尺度不敏感）。
#    - 在物理项内部使用 **反标准化的 K** 与 **E'**（保持物理量），再做 log-ratio。
#    - 为防 batch 极小 J→0 引发数值问题，加入 **分位数自适应地板**（J_floor）。
# 3) 训练策略：
#    - **warmup**：物理正则从 0 → λ_target 逐步爬坡，避免一上来就把模型拉崩。
#    - **梯度裁剪**：clipnorm 稳住早期噪声。
# 4) OOM 修复：
#    - 将 VGP 层的 convert_to_tensor_fn 改为 mean（非 sample）。
#    - 推理阶段提供 **批量预测** 工具，避免对 88,920 条样本一次性构造 O(N^2) 的核矩阵。
# 5) 仍保持：数据导入后**全面校验**（物理与有限性）、crack_n **pre-warp 再 z-score**。
# =============================================================================

import argparse
import json
import os

# 强制使用遗留版 Keras (双重保险)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import math
from dataclasses import dataclass, asdict, field

import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras

# 简写
tfk = tfp.math.psd_kernels
tfpl = tfp.layers
ki = keras.initializers


# ------------------------------------------------------------
# 优化器：ShojiNaturalGradient（Theorem B' with Strict Acute Angle Constraint）
# ------------------------------------------------------------


class ShojiNaturalGradient(keras.optimizers.legacy.Optimizer):
    """Shoji 等（2024）Theorem B' 的 O(D) 版本，并加入严格锐角约束。"""

    def __init__(
            self,
            learning_rate=0.001,
            momentum=0.9,
            angle_threshold=0.5,
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
        eps_div = tf.cast(1e-7, var.dtype)

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


# ------------------------------------------------------------
# 可调参数
# ------------------------------------------------------------
USE_FLOAT64 = True  # 使用 float64 保持数值稳定
DTYPE = tf.float64 if USE_FLOAT64 else tf.float32
np.random.seed(42);
tf.random.set_seed(42)
tf.keras.backend.set_floatx('float64' if USE_FLOAT64 else 'float32')

# 训练超参（可被超参优化覆盖）
LEARNING_RATE = 1e-3
EPOCHS = 150
BATCH_SIZE = 256
JITTER = 1e-3
NU = 0.30  # 泊松比
WARP_DIM = 2  # 对 crack_n 开方 warp（0-based 第2列）
WARP_EPS = 1e-6
NOISE0 = 0.05  # 观测噪声方差初值（softplus 逆会处理）
KL_SCALE = 0.01
DEFAULT_AMPLITUDE = 0.5
DEFAULT_LENGTH_SCALES = np.array([2.0, 0.5, 0.5, 0.1, 0.1, 2.0], np.float32)

# 物理正则控制
PHYS_TARGET_LAM = 1000  # 目标 λ
PHYS_WARMUP_STEPS = 2000  # 前多少 step 从 0 → λ（线性爬坡）
PHYS_FORM = 'logratio'  # 'logratio' 或 'ratio'
J_FLOOR_PERCENT = 1.0  # 对每个 batch 的 J_ref，下分位数百分位（1.0 表示第 1 百分位）
J_ABS_FLOOR = 1e-30  # 绝对地板，避免 log(0)

VERBOSE = 1  # 逐 batch 打印
PRINT_EVERY = 1
DEBUG_PHYS_FLOOR = False  # 是否打印 batch 分位数地板，默认关闭避免日志污染


@dataclass
class SVGPHyperParams:
    learning_rate: float = LEARNING_RATE
    noise0: float = NOISE0
    kl_scale: float = KL_SCALE
    amplitude: float = DEFAULT_AMPLITUDE
    length_scale_diag: np.ndarray = field(default_factory=lambda: DEFAULT_LENGTH_SCALES.copy())
    num_inducing: int | None = None

    def __post_init__(self):
        if self.length_scale_diag is None:
            object.__setattr__(self, "length_scale_diag", DEFAULT_LENGTH_SCALES.copy())


def _hp_to_dict(hp: SVGPHyperParams) -> dict:
    payload = asdict(hp)
    payload["length_scale_diag"] = [float(x) for x in hp.length_scale_diag]
    return payload


# ------------------------------------------------------------
# 工具：复合 Simpson 权重 & E_eq 计算
# ------------------------------------------------------------

def _simpson_weights(m: int) -> np.ndarray:
    assert m % 2 == 0 and m > 0
    w = np.ones(m + 1, dtype=np.float64)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w


def preprocess_window_eq(X_np: np.ndarray, m=20, delta_ratio=0.2):
    """基于内侧裂尖对称窗口的 E_eq 复合 Simpson 平均。X: [Ro, R_n, crack_n, theta_deg, n_FGM]

    参数
    ----
    m : int
        Simpson 复合分段数（必须为偶数）；越大积分越平滑但计算量线性增加。
    delta_ratio : float
        窗口宽度相对裂纹深度 a 的比例，决定积分区间大小。
    """

    # 拆分特征并强制使用 float64 保持积分精度
    N = X_np.shape[0]
    Ro, Rn, crack_n, theta_deg, n_FGM = [X_np[:, i].astype(np.float64) for i in range(5)]

    # 由几何关系得到内半径/裂尖位置/壁厚等中间量
    Ri = Rn * Ro  # 内半径 = 比例 * 外半径
    Rp = Ri  # 裂尖位置（假设在内壁）
    Dp = Ro - Ri  # 壁厚
    a = crack_n * Dp  # 裂纹深度

    # 基础几何合法性校验，提前拦截不合理样本
    if np.any(Dp <= 0):
        bad = np.where(Dp <= 0)[0]
        raise ValueError(f"存在 Dp<=0（壁厚非正），样本行: {bad[:10]} ...")
    if np.any((a <= 0) | (a >= Dp)):
        bad = np.where((a <= 0) | (a >= Dp))[0]
        raise ValueError(f"裂纹深度 a 不在 (0, Dp) 内，样本行: {bad[:10]} ...")

    # 根据裂尖中心点构造对称窗口，并限制窗口不越出物理边界
    r_tip = Rp + a
    Delta = np.maximum(1e-12, delta_ratio * a)  # 窗口半宽的基准值

    left_half = np.minimum(Delta, r_tip - Rp)  # 向内的最大可行半宽
    right_half = np.minimum(Delta, Rp + Dp - r_tip)  # 向外的最大可行半宽
    half = np.minimum(left_half, right_half)  # 取两侧最小，确保对称
    left, right = r_tip - half, r_tip + half
    Delta_eff = right - left  # 实际窗口宽度

    if np.any(Delta_eff <= 0):
        bad = np.where(Delta_eff <= 0)[0]
        raise ValueError(f"窗口宽度为零，检查 delta_ratio/几何，样本行: {bad[:10]} ...")

    # Simpson 积分步长与权重
    h = Delta_eff / m
    w = _simpson_weights(m)

    # 构造等效模量积分：预先定义梯度材料的 E(r)
    Eeq = np.zeros(N, dtype=np.float64)
    E0 = 214e9
    dE = 166e9

    for i in range(N):
        # r_i: 对称窗口内的积分网格（截断到合法区间）
        r_i = left[i] + h[i] * np.arange(m + 1, dtype=np.float64)
        r_i = np.clip(r_i, Rp[i], Rp[i] + Dp[i])

        # xi: 归一化径向坐标，Ei: 对应梯度模量分布
        xi = 1.0 - (r_i - Rp[i]) / Dp[i]
        Ei = E0 + dE * np.power(xi, n_FGM[i])

        # Simpson 复合积分并除以窗口宽度，得到平均等效模量
        integral = (h[i] / 3.0) * np.dot(w, Ei)
        Eeq[i] = integral / Delta_eff[i]

    # 返回辅助中间量，便于调试/验证几何
    aux = dict(
        Ro=Ro, Ri=Ri, Rp=Rp, Dp=Dp, a=a,
        r_tip=r_tip, left=left, right=right, h=h,
        m=m, delta_ratio=delta_ratio,
    )
    return Eeq.astype(np.float32), aux


# ------------------------------------------------------------
# 数据校验 & 预处理：先 warp(crack_n) 再 z-score
# ------------------------------------------------------------

def validate_and_prepare(X_raw: np.ndarray, Y_raw: np.ndarray):
    """输入校验 + warp + 标准化。

    该函数确保：
    1. X/Y 形状合理且均为有限值；
    2. 几何/物理约束满足常识（半径与裂纹比例均在 (0,1)）；
    3. 对 crack_n 开方，角度展开为正弦/余弦后再做 z-score 标准化。
    返回展开后的 X（未标准化）、标准化后的 X/Y 及对应均值、方差。
    """
    if not isinstance(X_raw, np.ndarray) or not isinstance(Y_raw, np.ndarray):
        raise TypeError("X_raw/Y_raw 必须是 numpy.ndarray")

    # X 基础形状：必须是 (N,5)
    if X_raw.ndim != 2 or X_raw.shape[1] != 5:
        raise ValueError(f"X 形状应为 (N,5)，当前 {X_raw.shape}")

    # Y 统一二维：避免 (C,) / (1,C) 这类特例
    if Y_raw.ndim == 1:
        Y_raw = Y_raw.reshape(-1, 1)
    elif Y_raw.ndim == 2 and Y_raw.shape[0] == 1 and Y_raw.shape[1] > 1:
        Y_raw = Y_raw.T
    if Y_raw.ndim != 2:
        raise ValueError(f"Y 形状应为二维 (N,C)，当前 {Y_raw.shape}")

    # 有限性过滤：丢弃含 NaN/Inf 的行，并提示被移除的行数
    X_finite = np.all(np.isfinite(X_raw), axis=1)
    Y_finite = np.all(np.isfinite(Y_raw), axis=1)
    keep_mask = X_finite & Y_finite
    if not keep_mask.all():
        bad = np.where(~keep_mask)[0]
        print(f"警告：移除含 NaN/Inf 样本 {bad.shape[0]} 行，示例索引: {bad[:10]}")
    Xc = X_raw[keep_mask].astype(np.float64)
    Yc = Y_raw[keep_mask].astype(np.float64)

    # 物理约束：半径比例与裂纹比例均应在 (0,1)
    Ro = Xc[:, 0];
    Rn = Xc[:, 1];
    crack_n = Xc[:, 2]
    if np.any(Ro <= 0):
        bad = np.where(Ro <= 0)[0];
        raise ValueError(f"存在 Ro<=0，样本行: {bad[:10]} ...")
    if np.any((Rn <= 0) | (Rn >= 1)):
        bad = np.where((Rn <= 0) | (Rn >= 1))[0];
        raise ValueError(f"R_n 必须在 (0,1)，样本行: {bad[:10]} ...")
    if np.any((crack_n <= 0) | (crack_n >= 1)):
        bad = np.where((crack_n <= 0) | (crack_n >= 1))[0];
        raise ValueError(f"crack_n 必须在 (0,1)，样本行: {bad[:10]} ...")

    # 几何一致性：裂纹深度 a 必须小于壁厚 Dp
    Ri = Rn * Ro
    Dp = Ro - Ri
    a = crack_n * Dp
    if np.any(Dp <= 0):
        bad = np.where(Dp <= 0)[0];
        raise ValueError(f"存在 Dp<=0（壁厚非正），样本行: {bad[:10]} ...")
    if np.any((a <= 0) | (a >= Dp)):
        bad = np.where((a <= 0) | (a >= Dp))[0];
        raise ValueError(f"a 不在 (0,Dp)，样本行: {bad[:10]} ...")

    # 按列拆分特征，显式转换角度
    theta_deg = Xc[:, 3]
    theta_rad = np.deg2rad(theta_deg)

    # pre-warp crack_n：对裂纹比例开方，缓解极小值导致的梯度尖锐
    crack_n_warp = np.sqrt(np.maximum(0.0, crack_n) + WARP_EPS)

    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)

    # 重新组装 6 维输入：[Ro, Rn, sqrt(crack_n), sin_theta, cos_theta, FGM_n]
    X_expanded = np.column_stack(
        [Ro, Rn, crack_n_warp, sin_theta, cos_theta, Xc[:, 4]]
    )

    # z-score（针对展开后数据）：避免零方差造成除零
    X_mean = X_expanded.mean(axis=0)
    X_std = X_expanded.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X_expanded - X_mean) / X_std

    # Y 标准化
    Y_mean = Yc.mean(axis=0)
    Y_std = Yc.std(axis=0)
    Y_std[Y_std == 0] = 1.0
    Y_norm = (Yc - Y_mean) / Y_std

    # 最终检查：尺寸/有限性安全
    if X_norm.shape[0] != Y_norm.shape[0]:
        raise ValueError(f"X/Y 样本数不一致：{X_norm.shape[0]} vs {Y_norm.shape[0]}")
    if not np.isfinite(X_norm).all():
        bad = np.argwhere(~np.isfinite(X_norm))[:5]
        raise ValueError(f"X_norm 含非有限值，位置示例: {bad}")
    if not np.isfinite(Y_norm).all():
        bad = np.argwhere(~np.isfinite(Y_norm))[:5]
        raise ValueError(f"Y_norm 含非有限值，位置示例: {bad}")

    return (
        X_expanded.astype(np.float32),
        X_norm.astype(np.float32), X_mean.astype(np.float32), X_std.astype(np.float32),
        Y_norm.astype(np.float32), Y_mean.astype(np.float32), Y_std.astype(np.float32),
        keep_mask,
    )


# ------------------------------------------------------------
# ARD RBF 核（不含 FeatureTransformed）
# ------------------------------------------------------------
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
            name="amplitude", shape=[],
            initializer=ki.Constant(np.log(np.expm1(amplitude))),
            dtype=DTYPE, trainable=True)
        self._ls_unconstrained = self.add_weight(
            name="length_scale_diag", shape=[input_dim],
            initializer=ki.Constant(np.log(np.expm1(length_scale_diag))),
            dtype=DTYPE, trainable=True)

    @property
    def kernel(self):
        amp = tf.nn.softplus(self._amp_unconstrained)
        ls = tf.nn.softplus(self._ls_unconstrained) + tf.cast(1e-12, DTYPE)
        base = tfk.ExponentiatedQuadratic(amplitude=amp, length_scale=1.0)
        ard = tfk.FeatureScaled(base, scale_diag=ls)
        return ard

    def call(self, inputs, **kwargs):
        return inputs


# ------------------------------------------------------------
# 日志辅助
# ------------------------------------------------------------

def _format_log(epoch: int, step: int, steps_per_epoch: int, metrics: dict) -> str:
    """格式化单步训练日志，方便在训练循环中复用。"""
    return (
        f"Epoch {epoch + 1}/{EPOCHS} Step {step}/{steps_per_epoch} "
        f"- loss: {metrics['loss']:.6e} - nll: {metrics['nll']:.6e} - kl: {metrics['kl']:.6e} "
        f"- phys: {metrics['phys']:.6e} - lam:{metrics['lam']:.2e} "
        f"- r_mean:{metrics['r_mean']:.2e} - J_floor:{metrics['J_floor']:.3e}"
    )


def _self_check(X_np_raw: np.ndarray, keep_mask_np: np.ndarray, X_norm_tf: tf.Tensor, X_mean: np.ndarray,
                X_std: np.ndarray):
    """轻量自测，确保关键计算与 warp 一致，便于导入模块时快速发现问题。"""
    assert _simpson_weights(4).tolist() == [1.0, 4.0, 2.0, 4.0, 1.0]
    Xtoy = np.array([[2.0, 0.5, 0.2, 0.0, 1.0], [3.0, 0.6, 0.3, 45.0, 2.0]], dtype=np.float32)
    _e_toy, _ = preprocess_window_eq(Xtoy, m=20, delta_ratio=0.2)
    assert _e_toy.shape == (2,) and np.all(_e_toy > 0)

    # warp 与展开一致性检查
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

    # 标准化后应为零均值（数值误差允许 1e-6）
    _norm_mean = tf.reduce_mean(X_norm_tf, axis=0).numpy()
    assert np.allclose(_norm_mean, np.zeros_like(_norm_mean), atol=1e-6)


def load_data_and_prepare(mat_path: str = "matlab_input.mat"):
    """读取 .mat -> 几何验证 -> 物理量计算 -> Tensor 化。

    返回一个包含 numpy 与 Tensor 版本数据的字典，用于后续模型构建和训练。
    """
    mat = sio.loadmat(mat_path)
    X_np_raw = mat["data_input"]
    Y_np_raw = mat["combinedData"]

    # 统一 X 形状
    if X_np_raw.ndim == 2:
        if X_np_raw.shape[-1] == 5:
            pass
        elif X_np_raw.shape[0] == 5:
            X_np_raw = X_np_raw.T
        else:
            X_np_raw = np.reshape(X_np_raw, (-1, 5), order="F")
    else:
        raise ValueError("data_input 维度异常，期望2D矩阵")

    # 统一 Y 维度
    Y_np_raw = np.atleast_2d(Y_np_raw)
    if Y_np_raw.shape[0] == 1 and Y_np_raw.shape[1] > 1:
        Y_np_raw = Y_np_raw.T

    # 校验 + 预处理（warp->zscore）
    (
        X_warp_raw,
        X_norm,
        X_mean, X_std,
        Y_norm,
        Y_mean, Y_std,
        keep_mask_np,
    ) = validate_and_prepare(X_np_raw, Y_np_raw)

    # E_eq（使用原始几何计算）
    Eeq_np, aux = preprocess_window_eq(X_np_raw[keep_mask_np], m=20, delta_ratio=0.2)

    # 转 Tensor
    X_norm_tf = tf.convert_to_tensor(X_norm, dtype=DTYPE)
    Y_norm_tf = tf.convert_to_tensor(Y_norm, dtype=DTYPE)
    Eeq_tf = tf.convert_to_tensor(Eeq_np, dtype=DTYPE)
    Y_mean_tf = tf.convert_to_tensor(Y_mean, dtype=DTYPE)
    Y_std_tf = tf.convert_to_tensor(Y_std, dtype=DTYPE)

    _self_check(X_np_raw, keep_mask_np, X_norm_tf, X_mean, X_std)

    return dict(
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
        X_raw=X_np_raw[keep_mask_np],
        Y_raw=Y_np_raw[keep_mask_np],
    )


class SVGPModel(keras.Model):
    def __init__(self, vgp_layer: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        self.vgp_layer = vgp_layer

    def call(self, inputs, training=None):
        return self.vgp_layer(inputs, training=training)


def _identity_with_shape(dist: tfp.distributions.Distribution):
    """Return distribution itself while supplying a TensorShape for TFP layer checks.

    TFP 0.23's DistributionLambda/VariationalGaussianProcess inspects `value.shape`
    even when `convert_to_tensor_fn` returns a Distribution. A bare Distribution
    lacks `.shape`, so we attach one derived from batch+event shapes to bypass the
    check without sampling or altering gradients.
    """

    if not hasattr(dist, "shape"):
        # Best-effort TensorShape; can include None for unknown dims but satisfies
        # the attribute access used by distribution_layer.py. This is pure-Python
        # metadata and does not touch graph ops or gradients.
        dist.shape = dist.batch_shape.concatenate(dist.event_shape)

    if not hasattr(dist, "get_shape"):
        dist.get_shape = lambda: dist.shape


    return dist


def build_model(
        X_norm: np.ndarray,
        Y_norm: np.ndarray,
        total_steps: int | None = None,
        hyperparams: SVGPHyperParams | None = None,
):
    """构建 SVGP 模型（显式返回分布对象，避免依赖 Keras 副作用）。"""
    if total_steps is None:
        batches_per_epoch = math.ceil(X_norm.shape[0] / BATCH_SIZE)
        total_steps = EPOCHS * batches_per_epoch
    total_steps = max(1, int(total_steps))

    hp = hyperparams or SVGPHyperParams()
    num_inducing = int(hp.num_inducing or min(1000, X_norm.shape[0]))
    D_out = int(Y_norm.shape[1])

    inducing_init = X_norm[np.random.choice(X_norm.shape[0], size=num_inducing, replace=False)]
    inducing_init_multi = np.stack([inducing_init] * D_out, axis=0).astype(np.float64 if USE_FLOAT64 else np.float32)

    initial_scale = np.eye(num_inducing, dtype=np.float64 if USE_FLOAT64 else np.float32)[None, ...] * 0.1
    initial_scale = np.tile(initial_scale, (D_out, 1, 1))

    vgp_layer = tfpl.VariationalGaussianProcess(
        num_inducing_points=num_inducing,
        kernel_provider=ARDRBFKernelLayer(
            amplitude=hp.amplitude,
            length_scale_diag=np.array(hp.length_scale_diag, np.float32),
            input_dim=6,
        ),
        event_shape=(D_out,),
        inducing_index_points_initializer=ki.Constant(inducing_init_multi),
        unconstrained_observation_noise_variance_initializer=ki.Constant(np.log(np.expm1(hp.noise0))),
        variational_inducing_observations_scale_initializer=ki.Constant(initial_scale),
        jitter=JITTER,
        convert_to_tensor_fn=_identity_with_shape,  # 返回分布对象并附加 shape 元数据
        name="SVGPLayer",
    )



    model = SVGPModel(vgp_layer)
    _ = model(tf.zeros([1, 6], dtype=DTYPE))  # 预构建变量，便于 summary/save_weights

    def negloglik(y_true, rv_pred):
        return -rv_pred.log_prob(y_true)

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=hp.learning_rate,
        decay_steps=total_steps,
        alpha=0.01,
    )

    optimizer = ShojiNaturalGradient(
        learning_rate=lr_schedule,
        momentum=0.9,
        angle_threshold=0.5,
        global_clipnorm=1.0,
    )
    model.compile(optimizer=optimizer, loss=negloglik)
    return model, hp


def _dry_run(model: keras.Model, X_norm_tf: tf.Tensor, Y_norm_tf: tf.Tensor):
    """快速前向测试，确保 log_prob 有限。"""
    M = int(min(32, X_norm_tf.shape[0]))
    rv0 = model(X_norm_tf[:M], training=False)
    ll0 = rv0.log_prob(Y_norm_tf[:M]).numpy()
    if not np.isfinite(ll0).all():
        raise RuntimeError("Dry-run log_prob 非有限：请提高 jitter/噪声或检查数据尺度。")


def make_train_step(
        model: keras.Model, Y_std_tf: tf.Tensor, Y_mean_tf: tf.Tensor, kl_scale: float
):
    """返回带物理正则的单步训练函数，封装 warmup/地板/ratio 选择。"""
    kl_scale_tf = tf.cast(kl_scale, DTYPE)

    @tf.function
    def train_step_phys(x_batch, y_batch, eeq_phys_batch, lam_var):
        lam = tf.cast(lam_var, DTYPE)

        with tf.GradientTape() as tape:
            rv = model(x_batch, training=True)
            nll = -tf.reduce_mean(rv.log_prob(y_batch))
            kl_raw = tf.reduce_sum(rv.surrogate_posterior_kl_divergence_prior())
            kl = kl_scale_tf * kl_raw

            # 反标准化到物理量（确保正则作用在真实尺度上）
            y_pred_mean = rv.mean()
            K_pred = y_pred_mean * Y_std_tf + Y_mean_tf
            K_true = y_batch * Y_std_tf + Y_mean_tf

            # E'：平面应变（或应力）。若要切换到平面应力，可直接使用 eeq_phys_batch。
            Eprime = eeq_phys_batch / (1.0 - NU ** 2)

            # 物理 J（带绝对地板，避免除零）
            eps = tf.cast(J_ABS_FLOOR, DTYPE)
            J_hat = tf.reduce_sum(tf.square(K_pred), axis=-1) / (Eprime + eps)
            J_ref = tf.reduce_sum(tf.square(K_true), axis=-1) / (Eprime + eps)

            # 分位数自适应地板：避免极小 J 造成 log/ratio 爆炸
            j_floor = tfp.stats.percentile(J_ref, q=tf.cast(J_FLOOR_PERCENT, tf.float32))
            j_floor = tf.maximum(j_floor, eps)
            J_hat_c = tf.maximum(J_hat, j_floor)
            J_ref_c = tf.maximum(J_ref, j_floor)

            if DEBUG_PHYS_FLOOR:
                tf.print("[debug] J_floor:", j_floor)

            # 可切换的物理形式：log-ratio 更稳健，ratio 更直观
            if PHYS_FORM == 'logratio':
                r = tf.math.log(J_hat_c) - tf.math.log(J_ref_c)
            else:  # 'ratio'
                r = J_hat_c / J_ref_c - 1.0
            phys = tf.reduce_mean(tf.square(r)) * lam

            # 汇总损失
            loss = nll + kl + phys

        # 反向传播 + 参数更新
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return {
            "loss": loss, "nll": nll, "kl": kl, "phys": phys, "lam": lam,
            "r_mean": tf.reduce_mean(r), "J_floor": j_floor,
        }

    return train_step_phys


def _slice_data_for_column(data: dict, col_idx: int) -> dict:
    """Prepare a per-column view of the dataset for single-output training."""

    y_norm_slice = data["Y_norm"][:, col_idx : col_idx + 1]
    y_mean_slice = data["Y_mean"][col_idx : col_idx + 1]
    y_std_slice = data["Y_std"][col_idx : col_idx + 1]

    return {
        **data,
        "Y_norm": y_norm_slice,
        "Y_norm_tf": tf.convert_to_tensor(y_norm_slice, dtype=DTYPE),
        "Y_mean_tf": tf.convert_to_tensor(y_mean_slice, dtype=DTYPE),
        "Y_std_tf": tf.convert_to_tensor(y_std_slice, dtype=DTYPE),
    }


def _train_val_split(N: int, val_ratio: float = 0.15):
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(N * (1.0 - val_ratio))
    return idx[:split], idx[split:]


def _materialize_subset(data: dict, train_idx: np.ndarray) -> dict:
    return {
        **data,
        "X_norm": data["X_norm"][train_idx],
        "Y_norm": data["Y_norm"][train_idx],
        "Eeq": data["Eeq"][train_idx],
        "X_norm_tf": tf.convert_to_tensor(data["X_norm"][train_idx], dtype=DTYPE),
        "Y_norm_tf": tf.convert_to_tensor(data["Y_norm"][train_idx], dtype=DTYPE),
        "Eeq_tf": tf.convert_to_tensor(data["Eeq"][train_idx], dtype=DTYPE),
    }


def _evaluate_config(hp: SVGPHyperParams, data: dict, val_data: dict | None = None, steps: int = 40) -> tuple[float, float | None]:
    model, hp_used = build_model(data["X_norm"], data["Y_norm"], total_steps=steps, hyperparams=hp)
    train_step_phys = make_train_step(model, data["Y_std_tf"], data["Y_mean_tf"], hp_used.kl_scale)

    ds = _make_dataset(data["X_norm_tf"], data["Y_norm_tf"], data["Eeq_tf"], shuffle_buffer=1024)
    iterator = iter(ds)
    lam_var = tf.Variable(0.0, dtype=DTYPE, trainable=False)
    step_counter = tf.Variable(0, dtype=tf.int64, trainable=False)
    for _ in range(steps):
        try:
            xb, yb, eb = next(iterator)
        except StopIteration:
            iterator = iter(ds)
            xb, yb, eb = next(iterator)
        step_counter.assign_add(1)
        lam_value = tf.cast(PHYS_TARGET_LAM, DTYPE) * tf.minimum(
            tf.cast(1.0, DTYPE),
            tf.cast(step_counter, DTYPE) / tf.cast(PHYS_WARMUP_STEPS, DTYPE),
        )
        lam_var.assign(lam_value)
        _ = train_step_phys(xb, yb, eb, lam_var)

    # Validation NLL 作为评分，数值越低越好
    rv_train = model(data["X_norm_tf"], training=False)
    nll_train = -tf.reduce_mean(rv_train.log_prob(data["Y_norm_tf"]))

    val_nll = None
    if val_data is not None:
        rv_val = model(val_data["X_norm_tf"], training=False)
        val_nll = float(-tf.reduce_mean(rv_val.log_prob(val_data["Y_norm_tf"])) .numpy())

    return float(nll_train.numpy()), val_nll


def hyperparameter_search(data: dict, col_idx: int, max_trials: int = 5) -> SVGPHyperParams:
    train_idx, val_idx = _train_val_split(data["X_norm"].shape[0])
    train_data = _materialize_subset(_slice_data_for_column(data, col_idx), train_idx)
    val_data = _materialize_subset(_slice_data_for_column(data, col_idx), val_idx)

    search_space = [
        SVGPHyperParams(learning_rate=lr, noise0=n0, kl_scale=kl, amplitude=amp)
        for lr in (5e-4, 1e-3)
        for n0 in (0.02, 0.05)
        for kl in (0.005, 0.01)
        for amp in (0.4, 0.6)
    ]

    best_hp = SVGPHyperParams()
    best_score = float("inf")

    for trial, hp in enumerate(search_space[:max_trials], start=1):
        train_nll, val_nll = _evaluate_config(hp, train_data, val_data)
        total_score = 0.5 * train_nll + 0.5 * (val_nll if val_nll is not None else train_nll)
        if total_score < best_score:
            best_score = total_score
            best_hp = hp
        print(f"[HyperOpt] Trial {trial}: lr={hp.learning_rate}, noise0={hp.noise0}, kl={hp.kl_scale}, amp={hp.amplitude}, score={total_score:.4f}")

    return best_hp


def _make_dataset(x_tf, y_tf, e_tf, shuffle_buffer=4096):
    ds = tf.data.Dataset.from_tensor_slices((x_tf, y_tf, e_tf))
    if shuffle_buffer:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def run_training(model: keras.Model, data: dict, save_name: str, hyperparams: SVGPHyperParams, target_col: int):
    """训练主循环：构建 Dataset、迭代 epochs，并打印可控日志。"""
    train_step_phys = make_train_step(model, data["Y_std_tf"], data["Y_mean_tf"], hyperparams.kl_scale)
    steps_per_epoch = math.ceil(data["X_norm_tf"].shape[0] / BATCH_SIZE)
    step_counter = tf.Variable(0, dtype=tf.int64, trainable=False)
    lam_var = tf.Variable(0.0, dtype=DTYPE, trainable=False)

    base_dataset = _make_dataset(data["X_norm_tf"], data["Y_norm_tf"], data["Eeq_tf"])

    for epoch in range(EPOCHS):
        for step, (xb, yb, eb) in enumerate(base_dataset, start=1):
            step_counter.assign_add(1)
            lam_value = tf.cast(PHYS_TARGET_LAM, DTYPE) * tf.minimum(
                tf.cast(1.0, DTYPE),
                tf.cast(step_counter, DTYPE) / tf.cast(PHYS_WARMUP_STEPS, DTYPE),
            )
            lam_var.assign(lam_value)
            out = train_step_phys(xb, yb, eb, lam_var)
            metrics = {
                "loss": float(out['loss'].numpy()),
                "nll": float(out['nll'].numpy()),
                "kl": float(out['kl'].numpy()),
                "phys": float(out['phys'].numpy()),
                "lam": float(out['lam'].numpy()),
                "r_mean": float(out['r_mean'].numpy()),
                "J_floor": float(out['J_floor'].numpy()),
            }

            if VERBOSE and (step % PRINT_EVERY == 0 or step == 1 or step == steps_per_epoch):
                print(_format_log(epoch, step, steps_per_epoch, metrics))

            if not np.isfinite(metrics["loss"]) or not np.isfinite(metrics["nll"]) or not np.isfinite(metrics["phys"]):
                print(f"\n[NaN DETECTED] at epoch {epoch + 1}, step {step}")
                # 打印该 batch 的安全统计，便于定位
                Kp = (model(xb, training=False).mean() * data["Y_std_tf"] + data["Y_mean_tf"]).numpy()
                Kt = (yb * data["Y_std_tf"] + data["Y_mean_tf"]).numpy()
                Eb = eb.numpy() / (1 - NU ** 2)
                print("K_pred stats -> min/median/max:", np.nanmin(Kp), np.nanmedian(Kp), np.nanmax(Kp))
                print("K_true stats -> min/median/max:", np.nanmin(Kt), np.nanmedian(Kt), np.nanmax(Kt))
                print("E' stats -> min/median/max:", np.nanmin(Eb), np.nanmedian(Eb), np.nanmax(Eb))
                raise RuntimeError("Loss became NaN/Inf; see stats above.")

        print(f"epoch {epoch + 1} done")

    model.save_weights(save_name)
    print(f"\n[System] Model weights saved to: {save_name}")

    stats_payload = {
        "X_mean": data["X_mean"].tolist(),
        "X_std": data["X_std"].tolist(),
        "Y_mean": data["Y_mean"].tolist(),
        "Y_std": data["Y_std"].tolist(),
    }
    with h5py.File(save_name, "a") as f:
        f.attrs["hyperparams"] = json.dumps(_hp_to_dict(hyperparams))
        f.attrs["target_column"] = int(target_col)
        f.attrs["train_stats"] = json.dumps(stats_payload)



def main():
    parser = argparse.ArgumentParser(description="Train SVGP for a single target column")
    parser.add_argument("--target-col", type=int, default=0, choices=[0, 1], help="目标列索引 (0 或 1)")
    parser.add_argument("--output", type=str, default=None, help="权重文件名，默认 svgp_col{target}.h5")
    parser.add_argument("--skip-hyperopt", action="store_true", help="跳过超参数搜索，直接用默认值")
    args = parser.parse_args()

    data = load_data_and_prepare()
    total_steps = EPOCHS * math.ceil(data["X_norm"].shape[0] / BATCH_SIZE)

    col_idx = int(args.target_col)
    save_name = args.output or f"svgp_col{col_idx}.h5"
    col_data = _slice_data_for_column(data, col_idx)

    hyperparams = SVGPHyperParams()
    if not args.skip_hyperopt:
        hyperparams = hyperparameter_search(data, col_idx)
        print(f"[HyperOpt] Selected hp: {_hp_to_dict(hyperparams)}")

    model, hyperparams = build_model(col_data["X_norm"], col_data["Y_norm"], total_steps, hyperparams)
    _dry_run(model, col_data["X_norm_tf"], col_data["Y_norm_tf"])
    run_training(model, col_data, save_name, hyperparams, col_idx)


if __name__ == "__main__":
    main()


# ------------------------------------------------------------
# 批量预测工具（避免一次性对全部样本构造 K(X,X) 导致 OOM）
# ------------------------------------------------------------

def predict_mean_batched(model, X_tf: tf.Tensor, batch_size: int = 8192):
    means = []
    N = X_tf.shape[0]
    for i in range(0, N, batch_size):
        rv = model(X_tf[i:i + batch_size], training=False)
        means.append(rv.mean().numpy())
    return np.vstack(means)

# 示例：仅在需要时调用
# pred_norm = predict_mean_batched(model, X_norm_tf, batch_size=4096)
# print("Predictive mean (first 3 rows, normalized):\n", pred_norm[:3])
