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

import os
import math
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

# 简写
tfk  = tfp.math.psd_kernels
tfpl = tfp.layers
ki   = keras.initializers

# ------------------------------------------------------------
# 可调参数
# ------------------------------------------------------------
USE_FLOAT64    = False             # 如需更稳可设 True（并把 JITTER 降到 1e-5 左右）
DTYPE          = tf.float64 if USE_FLOAT64 else tf.float32
np.random.seed(42); tf.random.set_seed(42)
tf.keras.backend.set_floatx('float64' if USE_FLOAT64 else 'float32')

# 训练超参
LEARNING_RATE  = 1e-3
EPOCHS         = 10
BATCH_SIZE     = 256
JITTER         = 1e-4
NU             = 0.30               # 泊松比
WARP_DIM       = 2                  # 对 crack_n 开方 warp（0-based 第2列）
WARP_EPS       = 1e-6
NOISE0         = 1e-1               # 观测噪声方差初值（softplus 逆会处理）

# 物理正则控制
PHYS_TARGET_LAM   = 1e-2            # 目标 λ
PHYS_WARMUP_STEPS = 2000            # 前多少 step 从 0 → λ（线性爬坡）
PHYS_FORM         = 'logratio'      # 'logratio' 或 'ratio'
J_FLOOR_PERCENT   = 1.0             # 对每个 batch 的 J_ref，下分位数百分位（1.0 表示第 1 百分位）
J_ABS_FLOOR       = 1e-30           # 绝对地板，避免 log(0)

# 调试开关：默认关闭，避免训练阶段的无关打印噪声
DEBUG_PHYS_FLOOR  = False           # True 时打印每个 batch 的分位数自适应地板，定位物理项异常

VERBOSE           = 1                # 逐 batch 打印
PRINT_EVERY       = 1

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
    """基于内侧裂尖对称窗口的 E_eq 复合 Simpson 平均。X: [Ro, R_n, crack_n, theta_deg, n_FGM]"""
    N = X_np.shape[0]
    Ro, Rn, crack_n, theta_deg, n_FGM = [X_np[:, i].astype(np.float64) for i in range(5)]

    Ri = Rn * Ro
    Rp = Ri
    Dp = Ro - Ri
    a  = crack_n * Dp

    if np.any(Dp <= 0):
        bad = np.where(Dp <= 0)[0]
        raise ValueError(f"存在 Dp<=0（壁厚非正），样本行: {bad[:10]} ...")
    if np.any((a <= 0) | (a >= Dp)):
        bad = np.where((a <= 0) | (a >= Dp))[0]
        raise ValueError(f"裂纹深度 a 不在 (0, Dp) 内，样本行: {bad[:10]} ...")

    r_tip = Rp + a
    Delta = np.maximum(1e-12, delta_ratio * a)

    left_half  = np.minimum(Delta, r_tip - Rp)
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
        xi  = 1.0 - (r_i - Rp[i]) / Dp[i]
        Ei  = E0 + dE * np.power(xi, n_FGM[i])
        integral = (h[i] / 3.0) * np.dot(w, Ei)
        Eeq[i]   = integral / Delta_eff[i]

    aux = dict(Ro=Ro, Ri=Ri, Rp=Rp, Dp=Dp, a=a,
               r_tip=r_tip, left=left, right=right, h=h, m=m, delta_ratio=delta_ratio)
    return Eeq.astype(np.float32), aux

# ------------------------------------------------------------
# 数据校验 & 预处理：先 warp(crack_n) 再 z-score
# ------------------------------------------------------------

def validate_and_prepare(X_raw: np.ndarray, Y_raw: np.ndarray):
    if not isinstance(X_raw, np.ndarray) or not isinstance(Y_raw, np.ndarray):
        raise TypeError("X_raw/Y_raw 必须是 numpy.ndarray")

    # X 基础形状
    if X_raw.ndim != 2 or X_raw.shape[1] != 5:
        raise ValueError(f"X 形状应为 (N,5)，当前 {X_raw.shape}")

    # Y 统一二维
    if Y_raw.ndim == 1:
        Y_raw = Y_raw.reshape(-1, 1)
    elif Y_raw.ndim == 2 and Y_raw.shape[0] == 1 and Y_raw.shape[1] > 1:
        Y_raw = Y_raw.T
    if Y_raw.ndim != 2:
        raise ValueError(f"Y 形状应为二维 (N,C)，当前 {Y_raw.shape}")

    # 有限性过滤
    X_finite = np.all(np.isfinite(X_raw), axis=1)
    Y_finite = np.all(np.isfinite(Y_raw), axis=1)
    keep_mask = X_finite & Y_finite
    if not keep_mask.all():
        bad = np.where(~keep_mask)[0]
        print(f"警告：移除含 NaN/Inf 样本 {bad.shape[0]} 行，示例索引: {bad[:10]}")
    Xc = X_raw[keep_mask].astype(np.float64)
    Yc = Y_raw[keep_mask].astype(np.float64)

    # 物理约束
    Ro  = Xc[:,0]; Rn = Xc[:,1]; crack_n = Xc[:,2]
    if np.any(Ro <= 0):
        bad = np.where(Ro <= 0)[0]; raise ValueError(f"存在 Ro<=0，样本行: {bad[:10]} ...")
    if np.any((Rn <= 0) | (Rn >= 1)):
        bad = np.where((Rn <= 0) | (Rn >= 1))[0]; raise ValueError(f"R_n 必须在 (0,1)，样本行: {bad[:10]} ...")
    if np.any((crack_n <= 0) | (crack_n >= 1)):
        bad = np.where((crack_n <= 0) | (crack_n >= 1))[0]; raise ValueError(f"crack_n 必须在 (0,1)，样本行: {bad[:10]} ...")

    # 几何一致性
    Ri = Rn * Ro
    Dp = Ro - Ri
    a  = crack_n * Dp
    if np.any(Dp <= 0):
        bad = np.where(Dp <= 0)[0]; raise ValueError(f"存在 Dp<=0（壁厚非正），样本行: {bad[:10]} ...")
    if np.any((a <= 0) | (a >= Dp)):
        bad = np.where((a <= 0) | (a >= Dp))[0]; raise ValueError(f"a 不在 (0,Dp)，样本行: {bad[:10]} ...")

    # pre-warp crack_n
    X_warp = Xc.copy()
    X_warp[:, WARP_DIM] = np.sqrt(np.maximum(0.0, X_warp[:, WARP_DIM]) + WARP_EPS)

    # z-score（针对 warp 后数据）
    X_mean = X_warp.mean(axis=0)
    X_std  = X_warp.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X_warp - X_mean) / X_std

    # Y 标准化
    Y_mean = Yc.mean(axis=0)
    Y_std  = Yc.std(axis=0)
    Y_std[Y_std == 0] = 1.0
    Y_norm = (Yc - Y_mean) / Y_std

    # 最终检查
    if X_norm.shape[0] != Y_norm.shape[0]:
        raise ValueError(f"X/Y 样本数不一致：{X_norm.shape[0]} vs {Y_norm.shape[0]}")
    if not np.isfinite(X_norm).all():
        bad = np.argwhere(~np.isfinite(X_norm))[:5]
        raise ValueError(f"X_norm 含非有限值，位置示例: {bad}")
    if not np.isfinite(Y_norm).all():
        bad = np.argwhere(~np.isfinite(Y_norm))[:5]
        raise ValueError(f"Y_norm 含非有限值，位置示例: {bad}")

    return (
        X_warp.astype(np.float32),
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
        ls  = tf.nn.softplus(self._ls_unconstrained) + tf.cast(1e-12, DTYPE)
        base = tfk.ExponentiatedQuadratic(amplitude=amp, length_scale=1.0)
        ard  = tfk.FeatureScaled(base, scale_diag=ls)
        return ard
    def call(self, inputs, **kwargs):
        return inputs

# ------------------------------------------------------------
# 读取 .mat & 数据准备
# ------------------------------------------------------------
mat = sio.loadmat("matlab_input.mat")
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
(X_warp_raw,
 X_norm,
 X_mean, X_std,
 Y_norm,
 Y_mean, Y_std,
 keep_mask_np) = validate_and_prepare(X_np_raw, Y_np_raw)

# E_eq（使用原始几何计算）
Eeq_np, aux = preprocess_window_eq(X_np_raw[keep_mask_np], m=20, delta_ratio=0.2)

# 转 Tensor
X_norm_tf = tf.convert_to_tensor(X_norm, dtype=DTYPE)
Y_norm_tf = tf.convert_to_tensor(Y_norm, dtype=DTYPE)
Eeq_tf    = tf.convert_to_tensor(Eeq_np, dtype=DTYPE)
Y_mean_tf = tf.convert_to_tensor(Y_mean, dtype=DTYPE)
Y_std_tf  = tf.convert_to_tensor(Y_std,  dtype=DTYPE)

# ------------------------------------------------------------
# 自测（不会改变训练）
# ------------------------------------------------------------
assert _simpson_weights(4).tolist() == [1.0, 4.0, 2.0, 4.0, 1.0]
Xtoy = np.array([[2.0,0.5,0.2,0.0,1.0],[3.0,0.6,0.3,45.0,2.0]], dtype=np.float32)
_e_toy,_ = preprocess_window_eq(Xtoy, m=20, delta_ratio=0.2)
assert _e_toy.shape==(2,) and np.all(_e_toy>0)
_tmp = X_np_raw[keep_mask_np].astype(np.float64)
_tmp_w = _tmp.copy(); _tmp_w[:,WARP_DIM] = np.sqrt(np.maximum(0.0,_tmp_w[:,WARP_DIM])+WARP_EPS)
_mask_other = np.ones(5, dtype=bool); _mask_other[WARP_DIM]=False
assert np.allclose(_tmp[:,_mask_other], _tmp_w[:,_mask_other])

# ------------------------------------------------------------
# 构建 SVGP（convert_to_tensor_fn=mean 以减轻内存）
# ------------------------------------------------------------
num_inducing = int(min(100, X_norm.shape[0]))
D_out        = int(Y_norm.shape[1])

inducing_init = X_norm[np.random.choice(X_norm.shape[0], size=num_inducing, replace=False)]
inducing_init_multi = np.stack([inducing_init] * D_out, axis=0).astype(np.float64 if USE_FLOAT64 else np.float32)

initial_scale = np.eye(num_inducing, dtype=np.float64 if USE_FLOAT64 else np.float32)[None, ...] * 0.1
initial_scale = np.tile(initial_scale, (D_out, 1, 1))

vgp_layer = tfpl.VariationalGaussianProcess(
    num_inducing_points=num_inducing,
    kernel_provider=ARDRBFKernelLayer(amplitude=0.5, length_scale_diag=np.ones(5, np.float32), input_dim=5),
    event_shape=(D_out,),
    inducing_index_points_initializer=ki.Constant(inducing_init_multi),
    unconstrained_observation_noise_variance_initializer=ki.Constant(np.log(np.expm1(NOISE0))),
    variational_inducing_observations_scale_initializer=ki.Constant(initial_scale),
    jitter=JITTER,
    convert_to_tensor_fn=lambda d: d.mean(),  # 避免默认 sample 带来的大矩阵 & 随机噪声
    name="SVGPLayer",
)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=[5], dtype=DTYPE),
    vgp_layer,
])

# ELBO 损失

def negloglik(y_true, rv_pred):
    return -rv_pred.log_prob(y_true)

optimizer = keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0)
model.compile(optimizer=optimizer, loss=negloglik)

# dry-run：确保前向有限
M = int(min(32, X_norm_tf.shape[0]))
rv0 = model(X_norm_tf[:M], training=False)
ll0 = rv0.log_prob(Y_norm_tf[:M]).numpy()
if not np.isfinite(ll0).all():
    raise RuntimeError("Dry-run log_prob 非有限：请提高 jitter/噪声或检查数据尺度。")

# ------------------------------------------------------------
# 物理正则：log-ratio + 分位数地板 + warmup
# ------------------------------------------------------------
@tf.function
def train_step_phys(x_batch, y_batch, eeq_phys_batch, step_counter):
    # 线性 warmup：从 0 → PHYS_TARGET_LAM
    lam = tf.cast(PHYS_TARGET_LAM, DTYPE) * tf.minimum(1.0, tf.cast(step_counter, DTYPE)/tf.cast(PHYS_WARMUP_STEPS, DTYPE))

    with tf.GradientTape() as tape:
        rv = model(x_batch, training=True)
        nll = -tf.reduce_mean(rv.log_prob(y_batch))
        kl  = tf.add_n(model.losses) if model.losses else tf.cast(0.0, DTYPE)

        # 反标准化到物理量
        K_pred = rv.mean()*Y_std_tf + Y_mean_tf
        K_true = y_batch*Y_std_tf + Y_mean_tf

        # E'：平面应变（或应力）
        Eprime = eeq_phys_batch/(1.0-NU**2)  # 若是平面应力，替换为 eeq_phys_batch

        # 物理 J（带地板）
        eps = tf.cast(J_ABS_FLOOR, DTYPE)
        J_hat = tf.reduce_sum(tf.square(K_pred), axis=-1) / (Eprime + eps)
        J_ref = tf.reduce_sum(tf.square(K_true), axis=-1) / (Eprime + eps)

        # 分位数自适应地板（按 batch）
        # 说明：小数据/极小 J 会导致 log/比值不稳定，这里对两者同时加地板，保障数值稳定且不破坏相对尺度。
        j_floor = tfp.stats.percentile(J_ref, q=tf.cast(J_FLOOR_PERCENT, tf.float32))
        j_floor = tf.maximum(j_floor, eps)
        J_hat_c = tf.maximum(J_hat, j_floor)
        J_ref_c = tf.maximum(J_ref, j_floor)

        # 仅在调试时打印地板，避免默认训练被刷屏
        if DEBUG_PHYS_FLOOR:
            tf.print("[DEBUG] J_floor (batch):", j_floor)

        if PHYS_FORM == 'logratio':
            r = tf.math.log(J_hat_c) - tf.math.log(J_ref_c)
        else:  # 'ratio'
            r = J_hat_c / J_ref_c - 1.0
        phys = tf.reduce_mean(tf.square(r)) * lam

        loss = nll + kl + phys

    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return {"loss": loss, "nll": nll, "kl": kl, "phys": phys, "lam": lam,
            "r_mean": tf.reduce_mean(r), "J_floor": j_floor}

# ------------------------------------------------------------
# 训练循环（逐 batch 打印，定位 NaN）
# ------------------------------------------------------------
# 日志格式化：集中控制字段，方便增删指标
def _format_log(epoch, step, steps_per_epoch, out_dict):
    return (
        f"Epoch {epoch+1}/{EPOCHS} Step {step}/{steps_per_epoch} - "
        f"loss: {float(out_dict['loss'].numpy()):.6e} - "
        f"nll: {float(out_dict['nll'].numpy()):.6e} - "
        f"kl: {float(out_dict['kl'].numpy()):.6e} - "
        f"phys: {float(out_dict['phys'].numpy()):.6e} - "
        f"lam:{float(out_dict['lam'].numpy()):.2e} - "
        f"r_mean:{float(out_dict['r_mean'].numpy()):.2e} - "
        f"J_floor:{float(out_dict['J_floor'].numpy()):.3e}"
    )

steps_per_epoch = math.ceil(X_norm_tf.shape[0] / BATCH_SIZE)
step_counter = tf.Variable(0, dtype=tf.int64, trainable=False)

for epoch in range(EPOCHS):
    ds = tf.data.Dataset.from_tensor_slices((X_norm_tf, Y_norm_tf, Eeq_tf)).shuffle(4096).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    for step, (xb, yb, eb) in enumerate(ds, start=1):
        step_counter.assign_add(1)
        out = train_step_phys(xb, yb, eb, step_counter)
        loss_v = float(out['loss'].numpy()); nll_v = float(out['nll'].numpy()); kl_v = float(out['kl'].numpy()); phys_v = float(out['phys'].numpy())

        if VERBOSE and (step % PRINT_EVERY == 0 or step==1 or step==steps_per_epoch):
            # 条件打印：遵循 VERBOSE/PRINT_EVERY，便于拓展到更多指标
            print(_format_log(epoch, step, steps_per_epoch, out))

        if not np.isfinite(loss_v) or not np.isfinite(nll_v) or not np.isfinite(phys_v):
            print(f"\n[NaN DETECTED] at epoch {epoch+1}, step {step}")
            # 打印该 batch 的安全统计，便于定位
            Kp = (model(xb, training=False).mean()*Y_std_tf + Y_mean_tf).numpy()
            Kt = (yb*Y_std_tf + Y_mean_tf).numpy()
            print("K_pred stats -> min/median/max:", np.nanmin(Kp), np.nanmedian(Kp), np.nanmax(Kp))
            print("K_true stats -> min/median/max:", np.nanmin(Kt), np.nanmedian(Kt), np.nanmax(Kt))
            Eb = eb.numpy(); print("E' stats -> min/median/max:", np.nanmin(Eb/(1-NU**2)), np.nanmedian(Eb/(1-NU**2)), np.nanmax(Eb/(1-NU**2)))
            raise RuntimeError("Loss became NaN/Inf; see stats above.")

    print(f"epoch {epoch+1} done")

# ------------------------------------------------------------
# 批量预测工具（避免一次性对全部样本构造 K(X,X) 导致 OOM）
# ------------------------------------------------------------

def predict_mean_batched(model, X_tf: tf.Tensor, batch_size: int = 8192):
    means = []
    N = X_tf.shape[0]
    for i in range(0, N, batch_size):
        rv = model(X_tf[i:i+batch_size], training=False)
        means.append(rv.mean().numpy())
    return np.vstack(means)

# 示例：仅在需要时调用
# pred_norm = predict_mean_batched(model, X_norm_tf, batch_size=4096)
# print("Predictive mean (first 3 rows, normalized):\n", pred_norm[:3])
