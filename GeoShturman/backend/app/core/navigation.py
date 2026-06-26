"""High-level navigation solve pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correlation import CandidateResult, coarse_search, refine_search_around_candidates
from .dem import DEMData, dem_xy_to_geodetic
from .kalman import kalman_smooth_1d
from .metrics import build_quality_report
from .nmea import parse_nmea_text
from .profile import align_profile_to_reference, clean_profile, radio_agl_to_terrain_msl


@dataclass
class NavigationSolution:
    estimated: dict
    quality: dict
    measured_profile: np.ndarray
    reference_profile: np.ndarray
    trajectory: dict
    candidates: list
    heatmap: np.ndarray
    metadata: dict


def solve_navigation(
    dem: DEMData,
    nmea_text: str,
    barometric_altitude_msl: float = 1500.0,
    sample_rate_hz: float = 5.0,
    search_center_x_m: float | None = None,
    search_center_y_m: float | None = None,
    search_radius_m: float = 2000.0,
    coarse_step_m: float = 250.0,
    fine_step_m: float = 50.0,
    azimuth_coarse_step_deg: float = 5.0,
    azimuth_fine_step_deg: float = 1.0,
    speed_min_mps: float = 20.0,
    speed_max_mps: float = 80.0,
    speed_coarse_step_mps: float = 5.0,
    speed_fine_step_mps: float = 1.0,
    enable_kalman: bool = True,
    parallel_jobs: int | None = 1,
    compensate_baro_drift: bool = True,
) -> NavigationSolution:
    """Estimate location, speed, and track direction from NMEA radio altitude."""

    parsed = parse_nmea_text(nmea_text)
    valid_altitudes = [item.altitude_m for item in parsed if item.parsed_ok and item.altitude_m is not None]
    if not valid_altitudes:
        raise ValueError("NMEA text contains no valid altitude samples")

    radio_profile = np.asarray(valid_altitudes, dtype=float)
    measured_profile = radio_agl_to_terrain_msl(radio_profile, barometric_altitude_msl)
    measured_profile = clean_profile(
        measured_profile,
        median_window=3,
        max_jump_m=120.0,
        hampel_window=7,
        outlier_sigma=3.5,
    )
    if enable_kalman:
        measured_profile = kalman_smooth_1d(measured_profile, process_variance=2.0, measurement_variance=6.0)

    center_x = float(search_center_x_m) if search_center_x_m is not None else dem.origin_x_m + dem.width_m / 2.0
    center_y = float(search_center_y_m) if search_center_y_m is not None else dem.origin_y_m + dem.height_m / 2.0

    coarse = coarse_search(
        dem=dem,
        measured_terrain_profile=measured_profile,
        sample_rate_hz=sample_rate_hz,
        search_center_x_m=center_x,
        search_center_y_m=center_y,
        search_radius_m=search_radius_m,
        search_step_m=coarse_step_m,
        azimuth_step_deg=azimuth_coarse_step_deg,
        speed_min_mps=speed_min_mps,
        speed_max_mps=speed_max_mps,
        speed_step_mps=speed_coarse_step_mps,
        top_k=12,
        n_jobs=parallel_jobs,
        compensate_drift=compensate_baro_drift,
    )
    refined = refine_search_around_candidates(
        dem=dem,
        measured_terrain_profile=measured_profile,
        sample_rate_hz=sample_rate_hz,
        coarse_result=coarse,
        search_radius_m=max(coarse_step_m, fine_step_m),
        search_step_m=fine_step_m,
        azimuth_window_deg=max(azimuth_coarse_step_deg, azimuth_fine_step_deg * 3.0),
        azimuth_step_deg=azimuth_fine_step_deg,
        speed_window_mps=max(speed_coarse_step_mps, speed_fine_step_mps * 3.0),
        speed_step_mps=speed_fine_step_mps,
        top_n=5,
        top_k=12,
        n_jobs=parallel_jobs,
        compensate_drift=compensate_baro_drift,
    )

    best = refined.best
    quality = build_quality_report(refined.candidates, measured_profile, best)
    corrected_measured, drift_report = align_profile_to_reference(measured_profile, best.reference_profile, degree=1)
    estimated = {
        "start_x_m": best.start_x_m,
        "start_y_m": best.start_y_m,
        "end_x_m": best.end_x_m,
        "end_y_m": best.end_y_m,
        "azimuth_deg": best.azimuth_deg,
        "speed_mps": best.speed_mps,
        "correlation": best.correlation,
        "rmse_m": best.rmse_m,
        "mae_m": best.mae_m,
        "combined_score": best.combined_score,
        "confidence": quality["confidence"],
        "baro_drift_offset_m": best.drift_offset_m,
        "baro_drift_slope_m_per_sample": best.drift_slope_m_per_sample,
    }
    start_geo = dem_xy_to_geodetic(dem, best.start_x_m, best.start_y_m)
    end_geo = dem_xy_to_geodetic(dem, best.end_x_m, best.end_y_m)
    if start_geo is not None and end_geo is not None:
        estimated.update(
            {
                "start_lat_deg": start_geo.lat_deg,
                "start_lon_deg": start_geo.lon_deg,
                "end_lat_deg": end_geo.lat_deg,
                "end_lon_deg": end_geo.lon_deg,
            }
        )
    trajectory = {
        "start": _point_to_dict(dem, best.start_x_m, best.start_y_m),
        "end": _point_to_dict(dem, best.end_x_m, best.end_y_m),
        "duration_s": max(0.0, (measured_profile.size - 1) / float(sample_rate_hz)),
        "sample_count": int(measured_profile.size),
    }

    return NavigationSolution(
        estimated=estimated,
        quality=quality,
        measured_profile=measured_profile,
        reference_profile=best.reference_profile,
        trajectory=trajectory,
        candidates=[_candidate_to_dict(candidate) for candidate in refined.candidates],
        heatmap=refined.heatmap,
        metadata={
            "parsed_samples": len(parsed),
            "valid_samples": int(radio_profile.size),
            "sample_rate_hz": float(sample_rate_hz),
            "barometric_altitude_msl": float(barometric_altitude_msl),
            "enable_kalman": bool(enable_kalman),
            "parallel_jobs": parallel_jobs,
            "compensate_baro_drift": bool(compensate_baro_drift),
            "baro_drift": drift_report,
            "corrected_measured_profile": corrected_measured,
            "coarse": coarse.metadata,
            "coarse_azimuth_values": coarse.azimuth_values,
            "refined": refined.metadata,
            "refined_azimuth_values": refined.azimuth_values,
        },
    )


def _candidate_to_dict(candidate: CandidateResult) -> dict:
    return {
        "start_x_m": candidate.start_x_m,
        "start_y_m": candidate.start_y_m,
        "end_x_m": candidate.end_x_m,
        "end_y_m": candidate.end_y_m,
        "azimuth_deg": candidate.azimuth_deg,
        "speed_mps": candidate.speed_mps,
        "correlation": candidate.correlation,
        "rmse_m": candidate.rmse_m,
        "mae_m": candidate.mae_m,
        "combined_score": candidate.combined_score,
        "baro_drift_offset_m": candidate.drift_offset_m,
        "baro_drift_slope_m_per_sample": candidate.drift_slope_m_per_sample,
    }


def _point_to_dict(dem: DEMData, x_m: float, y_m: float) -> dict:
    point = {"x_m": float(x_m), "y_m": float(y_m)}
    geo = dem_xy_to_geodetic(dem, x_m, y_m)
    if geo is not None:
        point.update({"lat_deg": geo.lat_deg, "lon_deg": geo.lon_deg})
    return point
