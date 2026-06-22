import numpy as np
import pandas as pd
import xarray as xr

from driftnet.utils import meters_to_degrees

# Re-using the meters_to_degrees and rk4_step logic from earlier...


def generate_target_bbox(
    lat: float, lon: float, lat_degree_radius: float = 0.05, lon_degree_radius: float = 0.05
) -> dict[str, float]:
    """
    Creates a bounding box of a specified degree radius around a specific coordinate point.
    """
    return {
        "min_lat": lat - lat_degree_radius,
        "max_lat": lat + lat_degree_radius,
        "min_lon": lon - lon_degree_radius,
        "max_lon": lon + lon_degree_radius,
    }


def get_velocity_at_point(
    ds: xr.Dataset, time: np.datetime64, lat: float, lon: float, u_var: str = "u", v_var: str = "v"
) -> tuple[float, float]:
    """
    Interpolates the velocity field at a specific continuous space-time point.
    Assumes land points have 0 velocity.

    Args:
        ds: xarray Dataset containing the Zarr velocity fields.
        time: Target timestamp.
        lat: Target latitude coordinate.
        lon: Target longitude coordinate.
    """
    try:
        # Use bilinear interpolation in space and nearest/linear in time
        point = ds.interp(time=time, lat=lat, lon=lon, method="linear")
        u = float(point[u_var].values)
        v = float(point[v_var].values)

        # Handle cases where interpolation yields NaN (e.g., edge of land mask)
        if np.isnan(u) or np.isnan(v):
            return 0.0, 0.0
        return u, v
    except KeyError:
        # Out of bounds temporally or spatially
        return 0.0, 0.0


def rk4_step(
    ds: xr.Dataset, time: np.datetime64, lat: float, lon: float, dt: float
) -> tuple[float, float]:
    """
    Advances a particle position using a 4th-order Runge-Kutta scheme.

    Args:
        ds: xarray Dataset containing velocity components.
        time: Current time.
        lat: Current latitude.
        lon: Current longitude.
        dt: Time step size in seconds.

    Returns:
        Tuple[float, float]: Next (lat, lon) coordinates.
    """
    dt_half = np.timedelta64(int(dt / 2), "s")
    dt_full = np.timedelta64(int(dt), "s")

    # K1
    u1, v1 = get_velocity_at_point(ds, time, lat, lon)
    dlon1, dlat1 = meters_to_degrees(u1, v1, lat)

    # K2
    lat2 = lat + dlat1 * (dt / 2)
    lon2 = lon + dlon1 * (dt / 2)
    u2, v2 = get_velocity_at_point(ds, time + dt_half, lat2, lon2)
    dlon2, dlat2 = meters_to_degrees(u2, v2, lat2)

    # K3
    lat3 = lat + dlat2 * (dt / 2)
    lon3 = lon + dlon2 * (dt / 2)
    u3, v3 = get_velocity_at_point(ds, time + dt_half, lat3, lon3)
    dlon3, dlat3 = meters_to_degrees(u3, v3, lat3)

    # K4
    lat4 = lat + dlat3 * dt
    lon4 = lon + dlon3 * dt
    u4, v4 = get_velocity_at_point(ds, time + dt_full, lat4, lon4)
    dlon4, dlat4 = meters_to_degrees(u4, v4, lat4)

    # Weighted average update
    new_lat = lat + (dt / 6.0) * (dlat1 + 2 * dlat2 + 2 * dlat3 + dlat4)
    new_lon = lon + (dt / 6.0) * (dlon1 + 2 * dlon2 + 2 * dlon3 + dlon4)

    return new_lat, new_lon


def is_in_target_cell(lat: float, lon: float, target_bbox: dict[str, float]) -> bool:
    """
    Checks if a position falls within a defined target bounding box bounding a grid cell.
    Expects target_bbox to contain keys: 'min_lat', 'max_lat', 'min_lon', 'max_lon'.
    """
    return (
        target_bbox["min_lat"] <= lat <= target_bbox["max_lat"]
        and target_bbox["min_lon"] <= lon <= target_bbox["max_lon"]
    )


def track_particle_with_history(
    ds: xr.Dataset,
    start_lat: float,
    start_lon: float,
    start_time: np.datetime64,
    dt: float,
    max_steps: int,
    target_bbox: dict[str, float] | None = None,
) -> tuple[str, np.ndarray, np.ndarray, float]:
    """
    Tracks a particle and records its full spatial trajectory over time.

    Returns:
        Tuple[str, np.ndarray, np.ndarray, float]:
        (status, lats_history, lons_history, transit_time_seconds)
    """
    lat, lon = start_lat, start_lon
    current_time = start_time
    transit_time = 0.0

    # Initialize history lists with the seeding location
    lats_history = [lat]
    lons_history = [lon]

    lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
    lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())

    for _step in range(max_steps):
        # 1. Evaluate target interception (only for ML/Phase 2)
        if target_bbox is not None:  # noqa SIM102
            if (
                target_bbox["min_lat"] <= lat <= target_bbox["max_lat"]
                and target_bbox["min_lon"] <= lon <= target_bbox["max_lon"]
            ):
                return "hit_target", np.array(lats_history), np.array(lons_history), transit_time

        # 2. Out of bounds check
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            return "out_of_bounds", np.array(lats_history), np.array(lons_history), transit_time

        # 3. Calculate next position using RK4
        next_lat, next_lon = rk4_step(ds, current_time, lat, lon, dt)

        # 4. Stuck on land check
        if np.isclose(next_lat, lat) and np.isclose(next_lon, lon):
            return "stuck_on_land", np.array(lats_history), np.array(lons_history), transit_time

        # Update states and append to history logs
        lat, lon = next_lat, next_lon
        lats_history.append(lat)
        lons_history.append(lon)

        current_time += np.timedelta64(int(dt), "s")
        transit_time += dt

    status = "missed_target" if target_bbox is not None else "completed_duration"
    return status, np.array(lats_history), np.array(lons_history), transit_time


def evaluate_and_extract_paths(
    ds_orig: xr.Dataset,
    ds_pred: xr.Dataset,
    seed_particles: list[tuple[float, float]],
    start_time: np.datetime64,
    baseline_duration_hours: float,
    safety_margin_hours: float,
    dt: float = 1800.0,
    tolerance_degrees: float = 0.05,
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    """
    Runs the evaluation and extracts trajectories for plotting.

    Returns:
        df_metrics: Pandas DataFrame to be exported to CSV.
        trajectory_map: Dictionary mapping particle IDs to their GT/ML coordinate arrays.
    """
    gt_steps = int((baseline_duration_hours * 3600) / dt)
    ml_max_steps = int(((baseline_duration_hours + safety_margin_hours) * 3600) / dt)

    records = []
    trajectory_map = {}

    for p_id, (s_lat, s_lon) in enumerate(seed_particles):
        # --- PHASE 1: GROUND TRUTH TRACKING ---
        _, lats_gt, lons_gt, _ = track_particle_with_history(
            ds=ds_orig,
            start_lat=s_lat,
            start_lon=s_lon,
            start_time=start_time,
            dt=dt,
            max_steps=gt_steps,
            target_bbox=None,
        )

        # Define target zone centered around where the true particle finished
        f_lat_gt, f_lon_gt = lats_gt[-1], lons_gt[-1]
        target_box = {
            "min_lat": f_lat_gt - tolerance_degrees,
            "max_lat": f_lat_gt + tolerance_degrees,
            "min_lon": f_lon_gt - tolerance_degrees,
            "max_lon": f_lon_gt + tolerance_degrees,
        }

        # --- PHASE 2: ML PREDICTED TRACKING ---
        status_ml, lats_ml, lons_ml, time_ml = track_particle_with_history(
            ds=ds_pred,
            start_lat=s_lat,
            start_lon=s_lon,
            start_time=start_time,
            dt=dt,
            max_steps=ml_max_steps,
            target_bbox=target_box,
        )

        # Compile summary row
        records.append(
            {
                "particle_id": p_id,
                "start_lat": s_lat,
                "start_lon": s_lon,
                "target_center_lat": f_lat_gt,
                "target_center_lon": f_lon_gt,
                "ml_outcome": status_ml,
                "ml_final_lat": lats_ml[-1],
                "ml_final_lon": lons_ml[-1],
                "gt_required_time_hours": baseline_duration_hours,
                "ml_transit_time_hours": time_ml / 3600.0,
                "time_anomaly_hours": (
                    (time_ml / 3600.0) - baseline_duration_hours
                    if status_ml == "hit_target"
                    else np.nan
                ),
                "final_skid_distance_deg": np.sqrt(
                    (f_lat_gt - lats_ml[-1]) ** 2 + (f_lon_gt - lons_ml[-1]) ** 2
                ),
            }
        )

        # Store full coordinate paths for plotting
        trajectory_map[p_id] = {
            "lons_gt": lons_gt,
            "lats_gt": lats_gt,
            "lons_pred": lons_ml,
            "lats_pred": lats_ml,
        }

    df_metrics = pd.DataFrame(records)
    return df_metrics, trajectory_map

    return pd.DataFrame(records)
