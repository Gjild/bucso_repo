from bucso.mixer import Mixer
from bucso.synth import Synth

def test_drive_derate_linear_slope(mixer1_model: Mixer):
    # Under-drive by 3 dB -> derate 3 dB (slope 1.0, capped at 6 dB)
    der = mixer1_model.drive_derate_db(delivered_dbm=8.0)  # min is 11
    assert abs(der - 3.0) < 1e-6
    # Large underdrive capped at 6 dB
    der = mixer1_model.drive_derate_db(delivered_dbm=0.0)
    assert abs(der - 6.0) < 1e-6

def test_lo_family_scaling_cap(mixer1_model: Mixer):
    # |m| * ΔLO with cap 12 dB. ΔLO = -10 dBc, |m|=3 => -30 dB but capped at -12 dB
    val = mixer1_model.family_scale_db(order_abs=3, lo_rel_dBc=-10.0)
    assert val >= -12.0  # not more negative than the cap
    assert abs(val - (-12.0)) < 1e-6

def test_divider_spectrum_affects_harmonics(lo1_synth: Synth):
    mode = lo1_synth.mdl.modes[0]
    # Compare carriers for divider /1 vs /4
    c1 = lo1_synth.equivalent_carriers(10e9, mode=mode, divider="/1", pfd_hz=100e6)
    c4 = lo1_synth.equivalent_carriers(10e9, mode=mode, divider="/4", pfd_hz=100e6)
    # Find k=2 harmonic in each set
    h2_1 = next(c for c in c1 if c.tag == "harm2")
    h2_4 = next(c for c in c4 if c.tag == "harm2")
    # Divider spectrum has +12 dB for /4 in this fixture
    assert abs((h2_4.rel_dBc - h2_1.rel_dBc) - 12.0) < 1e-6
