from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

import xarray as xr

from driftnet.utils import meters_to_degrees

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
    ds: xr.Dataset,
    time: np.datetime64,
    lat: float,
    lon: float,
    tree_u: KDTree,
    shape_u: tuple[int, ...],
    tree_v: KDTree,
    shape_v: tuple[int, ...]
) -> tuple[float, float]:
    """
    Finds nearest spatial indices on a curvilinear C-grid and interpolates in time.
    Assumes land points have 0 velocity.
    """

    # 1. Find the nearest 1D index for U and V coordinates
    _, idx_u = tree_u.query([lat, lon])
    _, idx_v = tree_v.query([lat, lon])

    # 2. Convert the 1D indices back to 2D (y, x) logical grid indices
    y_u, x_u = np.unravel_index(idx_u, shape_u)
    y_v, x_v = np.unravel_index(idx_v, shape_v)

    # 3. Select the exact spatial points, then interpolate temporally
    u_point = ds.isel(y=y_u, x=x_u).interp(time_counter=time, method="linear")
    v_point = ds.isel(y=y_v, x=x_v).interp(time_counter=time, method="linear")

    # 4. Extract U from the U-lookup (component 0) and V from V-lookup (component 1)
    u = float(u_point['velocity'].isel(component=0).values)
    v = float(v_point['velocity'].isel(component=1).values)

    # Handle cases where interpolation yields NaN (e.g., land mask)
    if np.isnan(u) or np.isnan(v):
        return 0.0, 0.0

    return u, v

    # except (KeyError, IndexError, ValueError):
    #     # Out of bounds temporally or spatially
    #     print('either key')
    #     return 0.0, 0.0


def rk4_step(
    ds: xr.Dataset,
    time: np.datetime64,
    lat: float,
    lon: float,
    dt: float,
    tree_u: KDTree,
    shape_u: tuple,
    tree_v: KDTree,
    shape_v: tuple
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
    u1, v1 = get_velocity_at_point(ds, time, lat, lon, tree_u, shape_u, tree_v, shape_v)

    dlon1, dlat1 = meters_to_degrees(u1, v1, lat)

    # K2
    lat2 = lat + dlat1 * (dt / 2)
    lon2 = lon + dlon1 * (dt / 2)
    u2, v2 = get_velocity_at_point(ds, time + dt_half, lat2, lon2, tree_u, shape_u, tree_v, shape_v)
    dlon2, dlat2 = meters_to_degrees(u2, v2, lat2)

    # K3
    lat3 = lat + dlat2 * (dt / 2)
    lon3 = lon + dlon2 * (dt / 2)
    u3, v3 = get_velocity_at_point(ds, time + dt_half, lat3, lon3, tree_u, shape_u, tree_v, shape_v)
    dlon3, dlat3 = meters_to_degrees(u3, v3, lat3)

    # K4
    lat4 = lat + dlat3 * dt
    lon4 = lon + dlon3 * dt
    u4, v4 = get_velocity_at_point(ds, time + dt_full, lat4, lon4, tree_u, shape_u, tree_v, shape_v)
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
    grid_data_path: Path,
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

    # 1. Load the 2D coordinate arrays
    grid_data = np.load(grid_data_path)

    # 2. Flatten them to build the spatial trees
    u_coords = np.column_stack((grid_data['u_lat'].ravel(), grid_data['u_lon'].ravel()))
    v_coords = np.column_stack((grid_data['v_lat'].ravel(), grid_data['v_lon'].ravel()))

    # 3. Build the KD-Trees (do this ONCE before the time loop)
    tree_u = KDTree(u_coords)
    tree_v = KDTree(v_coords)

    # 4. Save the original 2D shapes so we can convert the 1D KDTree index back to 2D (y,x)
    shape_u = grid_data['u_lat'].shape
    shape_v = grid_data['v_lat'].shape

    lat_min, lat_max = float(grid_data['v_lat'].min()), float(grid_data['v_lat'].max())
    lon_min, lon_max = float(grid_data['u_lon'].min()), float(grid_data['u_lon'].max())

    for _step in range(max_steps - 1):
        # 1. Evaluate target interception (only for ML/Phase 2)
        if target_bbox is not None:  # noqa SIM102
            if (
                target_bbox["min_lat"] <= lat <= target_bbox["max_lat"]
                and target_bbox["min_lon"] <= lon <= target_bbox["max_lon"]
            ):
                return "hit_target", np.array(lats_history), np.array(lons_history), transit_time

        # 2. Out of bounds check
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            print('out of bounds')
            return "out_of_bounds", np.array(lats_history), np.array(lons_history), transit_time

        # 3. Calculate next position using RK4
        next_lat, next_lon = rk4_step(ds, current_time, lat, lon, dt, tree_u, shape_u, tree_v, shape_v)

        # 4. Stuck on land check
        if np.isclose(next_lat, lat, rtol=0.0, atol=1e-10) and np.isclose(next_lon, lon, rtol=0.0, atol=1e-10):
            print('stuck on land')
            return "stuck_on_land", np.array(lats_history), np.array(lons_history), transit_time

        # Update states and append to history logs
        lat, lon = next_lat, next_lon
        lats_history.append(lat)
        lons_history.append(lon)

        current_time += np.timedelta64(int(dt), "s")
        transit_time += dt

        print(_step)

    status = "missed_target" if target_bbox is not None else "completed_duration"
    return status, np.array(lats_history), np.array(lons_history), transit_time


def evaluate_and_extract_paths(
    ds_orig: xr.Dataset,
    ds_pred: xr.Dataset,
    grid_data_path: Path,
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

    print('total steps', gt_steps)

    records = []
    trajectory_map = {}

    for p_id, (s_lat, s_lon) in enumerate(seed_particles):
        # --- PHASE 1: GROUND TRUTH TRACKING ---
        _, lats_gt, lons_gt, _ = track_particle_with_history(
            ds=ds_orig,
            grid_data_path=grid_data_path,
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
            grid_data_path=grid_data_path,
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
