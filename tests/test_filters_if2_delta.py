from bucso.filters import IF2Parametric

def test_if2_delta_attn_worse_outside_passband():
    win = IF2Parametric(center_hz=4.5e9, bw_hz=0.5e9, passband_il_db=1.0,
                        stop_floor_db=-80.0, rolloff_db_per_dec=40.0)
    # Desired at center (≈ -1 dB); then farther from center should have *more negative* attenuation
    a_des = win.attn_at(4.5e9)
    a_edge = win.attn_at(4.5e9 + 0.25e9)          # still in passband
    a_out  = win.attn_at(4.5e9 + 0.5e9 + 0.5e9)   # 0.5 GHz beyond edge
    # Numbers are negative (dB); “worse” attenuation ⇒ more negative ⇒ comparisons flip.
    assert a_des >= a_edge >= a_out
