# MERCURY
# 🌍 Climate Utils – Climate Emulator Toolkit

`climate_utils` is a Python module providing tools to preprocess, visualize, and analyze climate datasets. It is built to support workflows involving historical temperature data, GCM simulations, anomaly construction, regional filtering, and lifting scheme-based reconstruction.

---

## 📦 Features

- 📥 Load and decode climate datasets (NetCDF, CF-compliant)
- 🌡️ Convert temperatures from Kelvin to Celsius
- 🧭 Interpolate data to custom grid resolutions
- 🗺️ Extract regional data via polygons (CSV or KML)
- 📊 Compute global and regional temperature means
- 🔁 Build and visualize temperature anomalies
- 🔍 Analyze climate patterns with the lifting scheme (wavelet-inspired)
- 📈 Plot monthly or yearly data (global or regional)
- 🎯 Reconstruct spatio-temporal temperature maps from reduced features

---

## 🧱 Function Overview

| Function | Description | Parameters |
|----------|-------------|------------|
| `import_data(nomfichier: str, path: str)` | Load NetCDF file using netCDF4 and xarray | File name and path |
| `interpolation(datas: xr.Dataset, degre: float)` | Regrid dataset to target resolution | Dataset and grid step in degrees |
| `convert_K_to_C(dataset1: xr.Dataset)` | Convert `tas` from Kelvin to Celsius | Dataset with temperature |
| `time_convert(dataset1: xr.Dataset)` | Convert CF-time to readable strings | Dataset with time |
| `region_centered(dataset: xr.Dataset, region: str, path: str, nomfichier: str)` | Mask a dataset to a region defined in a CSV | Dataset, region acronym, file path |
| `region_centered_poly(dataset: xr.Dataset, polyn: list)` | Mask a dataset using a polygon of (lon, lat) coords | Dataset and polygon list |
| `build_anomalies(data: xr.Dataset, dataref: xr.Dataset, period: tuple)` | Compute monthly anomalies from reference period | Target dataset, reference dataset, year range |
| `globalmean(tas: xr.DataArray)` | Compute global mean with latitude weighting | Temperature array |
| `mean_region_month(mois: str, region: str, dataset: xr.Dataset)` | Compute mean temperature for a region and month | Month name, region name, dataset |
| `tas_mean_year(tas: xr.Dataset, year: tuple)` | Compute yearly mean (single year or range) | Dataset and year(s) |
| `plot(data_2_5: xr.Dataset, annee: int, mois: str)` | Plot a map for a given year and month | Dataset, year, month name |
| `plot_region(data_2_5: xr.Dataset, annee: int, mois: str, region: str, path: str, nomfichier: str)` | Plot temperature for a specific region | Dataset, year, month, region acronym and path to region file |
| `plot_mean(dataset1: xr.Dataset, years: tuple)` | Plot multi-year mean temperature map | Dataset and (start, end) years |
| `polygon_from_kml(nomfichier: str, path: str)` | Read polygon(s) from a KML file | File name and path |
| `regional_anomalie_byear(anomalies: xr.Dataset, region: list)` | Compute annual mean anomaly for each region | Anomalies and list of regions or polygons |
| `global_anomalie_byear(anomalies: xr.Dataset)` | Compute global annual mean anomaly | Anomalies dataset |
| `build_data_for_lift(data_region: xr.DataArray)` | Flatten a spatio-temporal region into matrix for ML | Monthly regional data |
| `lifting_scheme(data: np.ndarray, y: np.ndarray)` | Apply wavelet-like lifting scheme with KNN regression | Feature matrix and target |
| `apply_lifitng_scheme(data_region: xr.DataArray, y: np.ndarray)` | Full lifting + inverse reconstruction pipeline | Regional data and prediction |
| `rebuild_data_for_mapping(data_region: np.ndarray, data_to_rebuild: np.ndarray)` | Reconstruct map from flat data | Masked region and flat array |
| `show_result(data_reconstructed: np.ndarray, year: int)` | Display a 2D reconstructed map | Reconstructed data and year index |

---
# Required Files & Formats
NetCDF files (*.nc) for historical and GCM datasets

CSV file for region definitions:

Must include Acronym column and polygon coordinates separated by |

Optional: KML files for polygonal regions (via fastkml)


## Dependencies
Install required packages with:  
pip install numpy xarray netCDF4 cftime matplotlib cartopy shapely scikit-learn xesmf palettable pandas fastkml

## 🧪 Example Usage

```python
import climate_utils as cu

# Load historical dataset
ds = cu.import_data("historical_temp.nc", path="./data")

# Convert to Celsius
ds = cu.convert_K_to_C(ds)

# Interpolate to 2.5° grid
ds_interp = cu.interpolation(ds, degre=2.5)

# Plot temperature in January 2000
cu.plot(ds_interp, annee=2000, mois="janvier")

# Build anomalies with respect to pre-industrial period
anomalies = cu.build_anomalies(data=ds, dataref=ds, period=(1850, 1900))



