import numpy as np
import polars as pl
import xarray as xr
from scipy.fft import fft2, fftshift, fftfreq

from driftnet.config import DataConfig, ExperimentConfig
from driftnet.utils import _get_valid_spatial_slices
from driftnet.metrics.lagrange import haversine_distance

def colocate_velocity_to_rho(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolates Arakawa C-grid U and V components to the Rho (cell center) points.
    U is on the x-faces, V is on the y-faces.
    """
    # Average U in x-direction: shape becomes (ny, nx-1)
    u_rho = 0.5 * (u[:, :-1] + u[:, 1:])
    # Average V in y-direction: shape becomes (ny-1, nx)
    v_rho = 0.5 * (v[:-1, :] + v[1:, :])

    # Trim to matching shapes: (ny-1, nx-1)
    return u_rho[:-1, :], v_rho[:, :-1]

def get_average_grid_spacing(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    """Calculates the mean grid spacing in meters for x and y directions."""
    # dx: distance between adjacent columns
    dx_grid = haversine_distance(lon[:, :-1], lat[:, :-1], lon[:, 1:], lat[:, 1:]) * 1000.0
    # dy: distance between adjacent rows
    dy_grid = haversine_distance(lon[:-1, :], lat[:-1, :], lon[1:, :], lat[1:, :]) * 1000.0

    return float(np.mean(dx_grid)), float(np.mean(dy_grid))

def compute_1d_isotropic_spectrum(u: np.ndarray, v: np.ndarray, dx: float, dy: float):
    """
    Computes the 1D isotropic kinetic energy power spectrum.
    """
    ny, nx = u.shape

    # 1. Remove mean and apply 2D Hanning window
    window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    u_w = (u - np.nanmean(u)) * window
    v_w = (v - np.nanmean(v)) * window

    # 2. Compute 2D FFT (Normalized by number of pixels)
    u_fft = fftshift(fft2(u_w)) / (nx * ny)
    v_fft = fftshift(fft2(v_w)) / (nx * ny)

    # 3. 2D Kinetic Energy Spectrum
    ke_2d = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2)

    # 4. Define wavenumbers (cycles per meter)
    kx = fftshift(fftfreq(nx, d=dx))
    ky = fftshift(fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)

    # Radial wavenumber magnitude
    Kr = np.sqrt(KX**2 + KY**2)

    # 5. Azimuthal binning (Summing energy in radial annuli)
    # Use the smaller Nyquist frequency to avoid corner artifacts in the rectangular 2D spectrum
    kr_max = min(np.max(kx), np.max(ky))
    bins = np.linspace(0, kr_max, min(nx, ny) // 2)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    ke_1d = np.zeros_like(bin_centers)
    kr_flat = Kr.flatten()
    ke_flat = ke_2d.flatten()

    # Integrate (sum) the energy inside each wavenumber ring
    for i in range(len(bins) - 1):
        mask = (kr_flat >= bins[i]) & (kr_flat < bins[i+1])
        if np.any(mask):
            ke_1d[i] = np.sum(ke_flat[mask])

    return bin_centers, ke_1d

def calculate_spectra_for_time(data_config: DataConfig, exp_config: ExperimentConfig, time_str: str):
    """
    Main orchestrator: Loads grid and data, computes spectra for Truth vs ML.
    Returns: wavenumbers, truth_spectrum, ml_spectrum
    """
    # 1. Load and trim the grid coordinates
    x_slice, y_slice = _get_valid_spatial_slices(data_config)
    grid = np.load(data_config.grid_params)
    rho_lon = grid["rho_lon"][y_slice, x_slice]
    rho_lat = grid["rho_lat"][y_slice, x_slice]

    # Mean spacing in meters
    dx, dy = get_average_grid_spacing(rho_lon, rho_lat)

    # 2. Load Ground Truth
    ds_truth = xr.open_zarr(data_config.original_res).isel(x=x_slice, y=y_slice).sel(time_counter=time_str)
    u_t, v_t = colocate_velocity_to_rho(
        ds_truth["velocity"].isel(component=0).values,
        ds_truth["velocity"].isel(component=1).values
    )

    # 3. Load ML Prediction
    ds_pred = xr.open_zarr(exp_config.model_predictions).sel(time_counter=time_str)
    u_p, v_p = colocate_velocity_to_rho(
        ds_pred["velocity"].isel(component=0).values,
        ds_pred["velocity"].isel(component=1).values
    )

    # 4. Compute 1D Spectra
    k, ke_truth = compute_1d_isotropic_spectrum(u_t, v_t, dx, dy)
    _, ke_pred = compute_1d_isotropic_spectrum(u_p, v_p, dx, dy)

    return k, ke_truth, ke_pred


def write_spectra_to_csv(data_config: DataConfig, exp_config: ExperimentConfig,
                         date_to_calc: str = "2001-06-23T00:00:00.000000000"):
    k, ke_truth, ke_pred = calculate_spectra_for_time(data_config, exp_config, date_to_calc)

    df_spectrum = pl.DataFrame({
        "wavenumber": k,
        "KE_truth": ke_truth,
        "KE_pred": ke_pred
    })
    df_spectrum.write_csv(exp_config.metrics / "spectrum.csv")