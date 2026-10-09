"""E2: GPS (NMEA) import, lap split, track reconstruction and trace comparison."""

import math

import numpy as np
import pytest

from fsk_lapsim.config import default_data_dir
from fsk_lapsim.gps import (
    compare_trace,
    nmea_checksum,
    parse_nmea,
    read_nmea,
    split_laps,
    synth_nmea,
    to_local_xy,
    track_from_lap,
)
from fsk_lapsim.solver import simulate
from fsk_lapsim.trackgen import load_track

SAMPLE = default_data_dir() / "gps" / "sample_DUMMY.nmea"


def test_checksum_and_sentence_parsing():
    body = "GPRMC,123519.00,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W"
    good = f"${body}*{nmea_checksum(body)}"
    gga = "GPGGA,123519.00,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"
    lines = [
        good,
        f"${body}*00",
        "garbage",
        f"${gga}*{nmea_checksum(gga)}",
        "$GPRMC,123520.00,V,,,,,,,230394,,,N*"
        + nmea_checksum("GPRMC,123520.00,V,,,,,,,230394,,,N"),
    ]
    df = parse_nmea(lines)
    assert len(df) == 1  # bad checksum, garbage and void fix skipped
    r = df.iloc[0]
    assert r["lat_deg"] == pytest.approx(48 + 7.038 / 60)
    assert r["lon_deg"] == pytest.approx(11 + 31.0 / 60)
    assert r["speed_mps"] == pytest.approx(22.4 * 0.514444)
    assert r["sats"] == 8 and r["hdop"] == 0.9 and r["alt_m"] == 545.4
    with pytest.raises(ValueError):
        parse_nmea(["$GPGSV,1,1,00*79"])


def test_southern_western_and_vtg_speed_and_midnight():
    rmc1 = "GNRMC,235959.80,A,3345.000,S,07030.000,W,,,091026,,,A"
    vtg = "GNVTG,10.0,T,,M,,N,36.0,K,A"
    rmc2 = "GNRMC,000000.20,A,3345.001,S,07030.000,W,,,101026,,,A"
    df = parse_nmea([f"${b}*{nmea_checksum(b)}" for b in (rmc1, vtg, rmc2)])
    assert df.iloc[0]["lat_deg"] < 0 and df.iloc[0]["lon_deg"] < 0
    assert df.iloc[0]["speed_mps"] == pytest.approx(10.0)  # from VTG km/h
    assert df.iloc[1]["t_s"] == pytest.approx(0.4)  # across midnight


def test_local_projection():
    x, y = to_local_xy([35.0, 35.001], [126.0, 126.0])
    assert x[1] == pytest.approx(0.0) and y[1] == pytest.approx(111.2, rel=1e-3)
    x, y = to_local_xy([35.0], [126.001], lat0=35.0, lon0=126.0)
    assert x[0] == pytest.approx(111.2 * math.cos(math.radians(35)), rel=1e-3)


@pytest.mark.parametrize(
    "track_name", ["tracks/template_endurance_B.yaml", "tracks/fsk2026_endurance.yaml"]
)
def test_roundtrip_course_method_recovers_track_and_lap(baseline, track_name):
    tr = load_track(track_name)
    lap = simulate(baseline, tr)
    df = parse_nmea(
        synth_nmea(
            tr,
            lap,
            laps=3,
            rate_hz=5.0,
            pos_sigma_m=0.7,
            speed_sigma_mps=0.1,
            course_sigma_deg=0.5,
            seed=2,
        )
    )
    laps = split_laps(df)
    assert len(laps) == 2
    # crossing-time error ≈ position noise / speed at the line, two crossings per lap
    tol = 3.0 * 0.7 * math.sqrt(2.0) / float(lap.v[0])
    assert laps[0].lap_time_s == pytest.approx(lap.lap_time_s, abs=tol)
    rec = track_from_lap(laps[0])
    assert rec.closed and rec.length_m == pytest.approx(tr.length_m, rel=0.01)
    sim = simulate(baseline, rec)
    assert sim.lap_time_s == pytest.approx(lap.lap_time_s, rel=0.02)
    cmp = compare_trace(laps[0], rec, sim)
    s = cmp.summary()
    assert s["speed_rmse_kph"] < 6.0 and abs(s["mean_speed_diff_kph"]) < 2.0


def test_position_method_and_errors(baseline):
    tr = load_track("tracks/endurance_generated.yaml")
    lap = simulate(baseline, tr)
    df = parse_nmea(synth_nmea(tr, lap, laps=2, rate_hz=10.0, pos_sigma_m=0.0, seed=0))
    laps = split_laps(df)
    pos = track_from_lap(laps[0], method="position", smoothing_sigma_m=0.05)
    assert pos.length_m == pytest.approx(tr.length_m, rel=0.01)
    with pytest.raises(ValueError):
        track_from_lap(laps[0], method="magic")
    no_cog = laps[0]
    no_cog.course_deg = None
    with pytest.raises(ValueError):
        track_from_lap(no_cog)
    assert split_laps(df, start_xy=(1e4, 1e4)) == []


def test_bundled_dummy_log():
    df = read_nmea(SAMPLE)
    assert np.median(np.diff(df["t_s"])) == pytest.approx(0.2)
    laps = split_laps(df)
    assert len(laps) == 2
    assert "DUMMY" in SAMPLE.read_text(encoding="ascii").splitlines()[0]
