import numpy as np
from bucso.utils import coalesce_bins, lin_to_db

def test_rbw_coalescing_power_sum():
    # Two tones inside one bin should sum linearly (~ +3 dB when equal)
    bw = 10_000.0
    f0 = 29_000_000_000.0
    freqs = np.array([f0 + 100.0, f0 + 500.0])  # both within 10 kHz
    levels_db = np.array([-65.0, -65.0])

    cf, cL = coalesce_bins(freqs, levels_db, bin_width_hz=bw)
    assert len(cf) == 1
    # -65 dBc + -65 dBc power-sum ≈ -62 dBc (within ~0.2 dB tolerance)
    assert abs(cL[0] - (-62.0)) < 0.25

def test_rbw_coalescing_separate_bins():
    bw = 10_000.0
    f0 = 29e9
    freqs = np.array([f0 + 100.0, f0 + 20_000.0])  # second outside the bin
    levels_db = np.array([-70.0, -70.0])
    cf, cL = coalesce_bins(freqs, levels_db, bin_width_hz=bw)
    assert len(cf) == 2
    # No summation; levels preserved closely
    assert all(abs(v + 70.0) < 1e-6 for v in cL)
