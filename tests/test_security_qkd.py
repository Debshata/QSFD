"""Tests for the BB84 simulation (src/security/qkd/bb84.py)."""
from src.security.qkd.bb84 import run_bb84


def test_bb84_deterministic_given_seed():
    r1 = run_bb84(1000, seed=42)
    r2 = run_bb84(1000, seed=42)
    assert r1.sifted_key_alice == r2.sifted_key_alice
    assert r1.qber == r2.qber


def test_bb84_clean_channel_zero_qber():
    r = run_bb84(2000, seed=1)
    assert r.qber == 0.0
    assert r.key_accepted is True


def test_bb84_sifted_length_roughly_half_raw():
    r = run_bb84(4000, seed=1)
    assert 1600 < r.sifted_length < 2400  # ~50% +/- generous margin for one seed


def test_bb84_noisy_channel_qber_tracks_injected_noise():
    r = run_bb84(4000, seed=1, channel_noise=0.05)
    assert 0.02 < r.qber < 0.09  # roughly matches the 5% injected noise, one seed


def test_bb84_eavesdropper_produces_high_qber():
    r = run_bb84(4000, seed=1, eve_intercept_resend=True)
    assert r.qber > 0.15  # intercept-resend attack: theoretical ~25% QBER
    assert r.key_accepted is False


def test_bb84_high_qber_rejects_key():
    r = run_bb84(4000, seed=1, eve_intercept_resend=True, qber_accept_threshold=0.11)
    assert r.key_accepted is False
    assert r.final_key_length == 0
