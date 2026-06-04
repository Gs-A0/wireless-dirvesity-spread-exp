"""
Part 1: diversity combining experiment.

Students complete SC, MRC and a BER simulation over independent Rayleigh
flat fading branches.
"""

import numpy as np

from utils import (
    bpsk_demodulate,
    bpsk_modulate,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
    plot_diversity_snapshot,
    rayleigh_fading_branches,
)


def _validate_branch_arrays(received, channel):
    received = np.asarray(received, dtype=complex)
    channel = np.asarray(channel, dtype=complex)
    if received.ndim != 2 or channel.ndim != 2:
        raise ValueError('received and channel must be 2-D arrays: branches x symbols')
    if received.shape != channel.shape:
        raise ValueError('received and channel must have the same shape')
    if received.shape[0] < 1 or received.shape[1] < 1:
        raise ValueError('received and channel must not be empty')
    if np.any(np.abs(channel) < 1e-12):
        raise ValueError('channel contains near-zero coefficients')
    return received, channel


def selection_combining(received, channel):
    """
    Selection combining for flat fading branches.

    Parameters:
        received: complex array with shape (num_branches, num_symbols).
        channel: complex channel coefficients with the same shape.

    Returns:
        combined: one-dimensional equalized symbol estimates.

    Requirement:
        For each symbol, select the branch with the largest |h|^2 and divide
        the selected received sample by its channel coefficient.
    """
    received, channel = _validate_branch_arrays(received, channel)

    # --- 选择合并 SC ---
    # 对每个符号，选择 |h|^2 最大的分支，然后用 r/h 均衡
    num_branches, num_symbols = received.shape
    combined = np.zeros(num_symbols, dtype=complex)

    # 计算每个分支每个符号的 |h|^2
    channel_power = np.abs(channel) ** 2

    # 对每个符号找出最强分支的索引
    best_branch_indices = np.argmax(channel_power, axis=0)

    for sym_idx in range(num_symbols):
        best = best_branch_indices[sym_idx]
        combined[sym_idx] = received[best, sym_idx] / channel[best, sym_idx]

    return combined


def maximal_ratio_combining(received, channel):
    """
    Maximal ratio combining for flat fading branches.

    Returns:
        combined = sum(conj(h_i) * r_i) / sum(|h_i|^2)
    """
    received, channel = _validate_branch_arrays(received, channel)

    # --- 最大比合并 MRC ---
    # combined = sum(conj(h_i) * r_i) / sum(|h_i|^2)
    numerator = np.sum(np.conj(channel) * received, axis=0)
    denominator = np.sum(np.abs(channel) ** 2, axis=0)
    combined = numerator / denominator

    return combined


def simulate_diversity_ber(snr_db_values, num_bits=4000, num_branches=2, seed=2026):
    """
    Simulate BER for no diversity, SC and MRC.

    Returns:
        dict with keys: '单分支', 'SC', 'MRC'. Each value is a list of BERs.
    """
    snr_db_values = np.asarray(snr_db_values, dtype=float)
    if snr_db_values.ndim != 1 or len(snr_db_values) == 0:
        raise ValueError('snr_db_values must be a non-empty one-dimensional array')
    if num_bits <= 0 or num_branches < 2:
        raise ValueError('num_bits must be positive and num_branches must be at least 2')

    # --- 分集 BER 仿真 ---
    # 1. 生成随机比特并做 BPSK 调制
    bits = generate_bits(num_bits, seed=seed)
    symbols = bpsk_modulate(bits)

    ber_single = []
    ber_sc = []
    ber_mrc = []

    for snr_db in snr_db_values:
        # 2. 生成多分支瑞利衰落接收信号
        received, channel = rayleigh_fading_branches(
            symbols, num_branches, snr_db, seed=seed
        )

        # 3. 单分支均衡（仅使用第 0 分支）
        single_eq = received[0] / channel[0]
        bits_single = bpsk_demodulate(single_eq)
        ber_single.append(calculate_ber(bits, bits_single))

        # 4. 选择合并 SC
        sc_eq = selection_combining(received, channel)
        bits_sc = bpsk_demodulate(sc_eq)
        ber_sc.append(calculate_ber(bits, bits_sc))

        # 5. 最大比合并 MRC
        mrc_eq = maximal_ratio_combining(received, channel)
        bits_mrc = bpsk_demodulate(mrc_eq)
        ber_mrc.append(calculate_ber(bits, bits_mrc))

        # 更新 seed，使每个 SNR 点的噪声独立
        seed = seed + 1 if seed is not None else None

    return {'单分支': ber_single, 'SC': ber_sc, 'MRC': ber_mrc}


def equal_gain_combining(received, channel):
    """Optional: equal-gain combining with phase-only correction."""
    received, channel = _validate_branch_arrays(received, channel)

    # --- 等增益合并 EGC ---
    # 仅做相位校正，各分支等权重叠加
    # combined = mean(r_i * conj(h_i) / |h_i|)
    num_branches = received.shape[0]

    # 提取相位校正因子：conj(h) / |h|，即 e^{-j * angle(h)}
    phase_correction = np.conj(channel) / (np.abs(channel) + 1e-15)

    # 各分支等权重相干叠加
    combined = np.mean(received * phase_correction, axis=0)

    return combined


def run_diversity_demo():
    """Run Part 1 demo and generate figures."""
    print('=' * 60)
    print('Part 1: 分集合并实验')
    print('=' * 60)
    snr_db_values = np.array([0, 3, 6, 9, 12, 15], dtype=float)

    try:
        ber_curves = simulate_diversity_ber(snr_db_values, num_bits=6000, num_branches=2, seed=2026)
        plot_ber_curve(snr_db_values, ber_curves, '瑞利衰落信道下分集合并 BER 对比', 'diversity_ber_curve.png')

        bits = generate_bits(120, seed=7)
        symbols = bpsk_modulate(bits)
        received, channel = rayleigh_fading_branches(symbols, 2, snr_db=8, seed=17)
        branch_equalized = received[0] / channel[0]
        mrc_output = maximal_ratio_combining(received, channel)
        plot_diversity_snapshot(symbols, branch_equalized, mrc_output, 'diversity_waveform_snapshot.png')

        print('[OK] 已生成 results/diversity_ber_curve.png')
        print('[OK] 已生成 results/diversity_waveform_snapshot.png')
    except NotImplementedError as error:
        print(f'[WAIT] 尚未完成核心函数: {error}')
    except Exception as error:
        print(f'[FAIL] Part 1 运行失败: {error}')


if __name__ == '__main__':
    run_diversity_demo()
