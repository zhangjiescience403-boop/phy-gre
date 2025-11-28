"""轻量形状回归：用 torch 生成假数据，跑通模型 1 step 并校验输入输出维度。"""
from importlib import util
from pathlib import Path

import pytest

tf = pytest.importorskip("tensorflow")
torch = pytest.importorskip("torch")

_MODULE_PATH = Path(__file__).resolve().parents[1] / "123_fix1.py"
_SPEC = util.spec_from_file_location("svgp_fix_module", _MODULE_PATH)
svgp_fix_module = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(svgp_fix_module)  # type: ignore


def _make_mock_data(num_samples: int = 8, out_dim: int = 2):
    torch.manual_seed(0)

    # 构造满足几何约束的随机特征：Ro>0，0<Rn/crack_n<1
    Ro = torch.rand(num_samples, 1) + 1.0
    Rn = torch.rand(num_samples, 1) * 0.8 + 0.1
    crack_n = torch.rand(num_samples, 1) * 0.8 + 0.1
    theta_deg = torch.rand(num_samples, 1) * 180.0
    n_fgm = torch.rand(num_samples, 1) * 2.0

    X_raw = torch.cat([Ro, Rn, crack_n, theta_deg, n_fgm], dim=1).numpy()
    Y_raw = torch.randn(num_samples, out_dim).numpy()
    return X_raw, Y_raw


def test_single_step_shapes():
    X_raw, Y_raw = _make_mock_data()

    # 复用原始校验与预处理逻辑
    (
        X_warp_raw,
        X_norm,
        X_mean,
        X_std,
        Y_norm,
        Y_mean,
        Y_std,
        keep_mask,
    ) = svgp_fix_module.validate_and_prepare(X_raw, Y_raw)

    # 使用较小的 Simpson 段数即可满足形状测试
    Eeq_np, _ = svgp_fix_module.preprocess_window_eq(X_raw[keep_mask], m=4, delta_ratio=0.1)

    X_norm_tf = tf.convert_to_tensor(X_norm, dtype=svgp_fix_module.DTYPE)
    Y_norm_tf = tf.convert_to_tensor(Y_norm[:, :1], dtype=svgp_fix_module.DTYPE)
    Y_other_norm_tf = tf.convert_to_tensor(Y_norm[:, 1:], dtype=svgp_fix_module.DTYPE)
    Eeq_tf = tf.convert_to_tensor(Eeq_np, dtype=svgp_fix_module.DTYPE)
    Y_mean_tf = tf.convert_to_tensor(Y_mean[:1], dtype=svgp_fix_module.DTYPE)
    Y_std_tf = tf.convert_to_tensor(Y_std[:1], dtype=svgp_fix_module.DTYPE)
    Y_other_mean_tf = tf.convert_to_tensor(Y_mean[1:], dtype=svgp_fix_module.DTYPE)
    Y_other_std_tf = tf.convert_to_tensor(Y_std[1:], dtype=svgp_fix_module.DTYPE)
    mask_zero_tf = tf.zeros(Y_norm_tf.shape[0], dtype=tf.bool)

    # 构建模型并做一次前向
    model = svgp_fix_module.build_model(X_norm, Y_norm[:, :1], target_col=0)
    rv = model(X_norm_tf, training=False)
    assert rv.mean().shape == Y_norm_tf.shape

    # 单步训练，检查返回标量是否有限
    train_step = svgp_fix_module.make_train_step(
        model,
        Y_std_tf,
        Y_mean_tf,
        svgp_fix_module.KL_SCALE,
        0,
        Y_other_std_tf,
        Y_other_mean_tf,
    )
    lam_var = tf.Variable(1.0, dtype=svgp_fix_module.DTYPE)
    out = train_step(X_norm_tf, Y_norm_tf, Y_other_norm_tf, Eeq_tf, mask_zero_tf, lam_var)

    for key in ["loss", "nll", "kl", "phys", "lam", "r_mean", "J_floor", "loss_sym"]:
        tensor = out[key]
        assert tensor.shape == ()
        assert tf.reduce_all(tf.math.is_finite(tensor))
