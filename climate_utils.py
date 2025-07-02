import xesmf as xe
from scipy.constants import convert_temperature
from datetime import datetime
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from palettable.colorbrewer.diverging import RdBu_11
from shapely.geometry import Point, Polygon
import pandas as pd
import numpy as np
import xarray as xr
from scipy.cluster.hierarchy import ward,fcluster,linkage
import os
import cftime
from sklearn.neighbors import KNeighborsRegressor
import netCDF4
import cartopy.crs as ccrs

def import_data(nomfichier,path):   
    file = nomfichier
    file_path = os.path.join(path, file) 
    print("Fichier trouvé :", os.path.exists(file_path))
    try:
        dataset = netCDF4.Dataset(file_path, mode="r")
        print("Le fichier s'ouvre correctement avec netCDF4.")
        dataset.close()
    except Exception as e:
        print("Erreur lors de l'ouverture avec netCDF4 :", e)
    try:
        dataset1 = xr.open_dataset(file_path, decode_times=False, cache=False)
        print("Le fichier s'ouvre correctement avec xarray.")
        return dataset1
    except Exception as e:
        print("Erreur avec xarray :", e)

def interpolation(datas,degre):

    new_lats = np.arange(-90, 90, degre)
    new_lons = np.arange(0, 360, degre)
    ds = datas.assign_coords(lon=(datas.lon % 360))
    ds = ds.sortby('lon')
    # Création du dataset cible
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], new_lats),
            "lon": (["lon"], new_lons),
        }
    )
    
    # Créateur de l'interpolateur
    regridder = xe.Regridder(ds, ds_out, "bilinear", periodic=True)
    
    # Application
    ds_interp = regridder(ds)
    ds_interp.time.attrs["units"] = "days since 1850-01-01"
    ds_interp.time.attrs["calendar"]="noleap"
    return ds_interp

def convert_K_to_C(dataset1):
    
    temp = dataset1
    temp = convert_temperature(temp.tas,'Kelvin','Celsius')
    dataset1.tas.data = temp
    print(dataset1.tas.data)
    return dataset1

def time_convert(dataset1):
    
    time = cftime.num2pydate(dataset1.time,dataset1.time.units,calendar ='standard')
    time_formatee=[]
    for i in range (dataset1.time.shape[0]):
        time_formatee.append(time[i].strftime("%Y-%m-%d %H:%M"))
    return time_formatee

### representation of a location :
def plot(data_2_5,annee,mois):
 
    months = { "janvier":1,"fevrier":2,"mars":3,"avril":4,"mai":5,"juin":6,"juillet":7,"aout":8,"septembre":9,"octobre":10,"novembre":11,"decembre":12}
    for i in range(cftime.num2pydate(data_2_5.time,"days since 1850-1-1",calendar ='standard').shape[0]):
        if annee == cftime.num2pydate(data_2_5.time[i],"days since 1850-1-1",calendar ='standard').year:
                   if cftime.num2pydate(data_2_5.time[i],"days since 1850-1-1",calendar ='standard').month == months[mois]:
                       print(annee,mois)
                       index = i
    fig,ax = plt.subplots(figsize=(15, 10) , subplot_kw = { 'projection' : ccrs.PlateCarree()})
    data_2_5['tas'][index].plot.pcolormesh(ax=ax,transform = ccrs.PlateCarree(), add_colorbar=True)
    ax.set_prop_cycle('color',RdBu_11.mpl_colors)
    ax.stock_img()
    ax.set_title(f'Temperature {cftime.num2pydate(data_2_5.time[index],"days since 1850-1-1",calendar ='standard')}')
    ax.coastlines(resolution = '110m')
    plt.show()

def plot_region(data_2_5,annee,mois,region,path,nomfichier):
 
    months = { "janvier":1,"fevrier":2,"mars":3,"avril":4,"mai":5,"juin":6,"juillet":7,"aout":8,"septembre":9,"octobre":10,"novembre":11,"decembre":12}
    for i in range(cftime.num2pydate(data_2_5.time,"days since 1850-1-1",calendar ='standard').shape[0]):
        if annee == cftime.num2pydate(data_2_5.time[i],"days since 1850-1-1",calendar ='standard').year:
                   if cftime.num2pydate(data_2_5.time[i],"days since 1850-1-1",calendar ='standard').month == months[mois]:
                       print(annee,mois)
                       index = i
    fig,ax = plt.subplots(figsize=(15, 10) , subplot_kw = { 'projection' : ccrs.PlateCarree()})
    data_2_5['tas'][index].plot.pcolormesh(ax=ax,transform = ccrs.PlateCarree(), add_colorbar=True)

    # Sous-ensemble de la région
    region_data = region_centered(data_2_5, region,path,nomfichier)
    region_data = region_data.assign_coords(lon=region_data.lon - 180)


    # Bornes géographiques pour ax.set_extent : [lon_min, lon_max, lat_min, lat_max]
    lon_min = float(region_data.lon.min())
    lon_max = float(region_data.lon.max())
    lat_min = float(region_data.lat.min())
    lat_max = float(region_data.lat.max())
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.set_prop_cycle('color',RdBu_11.mpl_colors)
    ax.stock_img()
    ax.set_title(f'Temperature {cftime.num2pydate(data_2_5.time[index],"days since 1850-1-1",calendar ='standard')}')
    ax.coastlines(resolution = '110m')
    plt.show()
    
def plot_mean(dataset1,years = (1850,None)):

    fig,ax = plt.subplots(figsize=(15, 10) , subplot_kw = { 'projection' : ccrs.PlateCarree()})
    dataset1['tas'].plot.pcolormesh(ax=ax,transform = ccrs.PlateCarree(), add_colorbar=True)
    ax.set_prop_cycle('color',RdBu_11.mpl_colors)
    ax.stock_img()
    ax.set_title(f'Température moyenne ( {years[0]} - {years[1]} )')
    ax.coastlines(resolution = '110m')
    plt.show()
    
def extract_coordinates(value):
    if pd.isna(value):
        return None
    return tuple(map(float, value.split('|')))

def region_centered(dataset,region,path,nomfichier):
    
    file = nomfichier
    file_path = os.path.join(path, file) 
    print("Fichier trouvé :", os.path.exists(file_path))
    
    regions_data = pd.read_csv(file_path,sep=',')
   
    for i in range (regions_data["Acronym"].shape[0]):
        if regions_data["Acronym"][i]==region :
            choix = i

    regions_points = regions_data.iloc[:,4:]
    coordinates_df = regions_points.applymap(extract_coordinates)
    polygon1 = Polygon(coordinates_df.iloc[choix,:].dropna().to_list()).reverse()
    # Création de la grille lat/long
    lat = dataset.lat.values #np.linspace(50, 80, 180)  # Exemple de latitudes dans la plage de votre polygone
    lon = dataset.lon.values #np.linspace(-170, -105, 288)  # Exemple de longitudes dans la plage de votre polygone
    lon = lon -180
    # Initialiser le masque
    mask = np.zeros((len(dataset.lat), len(dataset.lon)), dtype=bool)
    
    # Remplir le masque en vérifiant si chaque point est à l'intérieur du polygone
    for i in range(0, len(dataset.lat)):
        for j in range(0, len(dataset.lon)):
            point = Point(lon[j], lat[i])  # Créer un objet Point avec la longitude et la latitude
            if polygon1.contains(point):
                mask[i][j] = True  # Marquer comme True si le point est dans le polygone
            else:
                mask[i][j] = False  # Sinon, laisser comme False
    #remettre +180 pour revenir a 0-360
    lon=lon+180
    # Convertir le masque en DataArray avec les dimensions correspondantes
    mask_xr = xr.DataArray(mask, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    
    # Appliquer le masque sur tas_mean
    masked_data = dataset.where(mask_xr,drop=True)
    # Afficher le résultat
    return masked_data

def globalmean(tas):
    # Extraire les latitudes
    lat = tas.lat
    
    # Calculer les poids basés sur le cosinus de la latitude (convertie en radians)
    weights = np.cos(np.deg2rad(lat))
    
    # Normaliser les poids pour qu'ils totalisent 1
    weights /= weights.sum()
    
    # Calculer la moyenne globale pondérée
    global_mean_temp = (tas * weights).sum(dim="lat").mean(dim="lon")
    return global_mean_temp
def mean_region_month(mois,region,dataset):
    months = { "janvier":0,"fevrier":1,"mars":2,"avril":3,"mai":4,"juin":5,"juillet":6,"aout":7,"septembre":8,"octobre":9,"novembre":10,"decembre":11}
    print(f"température moyenne pour la région {region} au mois de {mois} : {globalmean(region_centered(dataset.isel(time=months[mois]),region)).tas.values}")

def tas_mean_year(tas, year=(2015, None)):
    time = cftime.num2pydate(tas.time,tas.time.units,calendar ='standard')
    if year[1] is None:
        year = int(year[0])
        
        for i in range(len(time)):
            if time[i].year==year:
                if time[i].month ==1:
                    start = time[i]
                    ind=i
        tas_mean = tas.sel(time=slice(cftime.date2num(time[ind],tas.time.units,calendar='standard'), cftime.date2num(time[ind+11],tas.time.units,calendar='standard'))).weighted(tas.time.diff("time")).mean(dim='time')
        return tas_mean   
    
    else:
        year_start = int(year[0])
        year_end = int(year[1])
        for i in range(len(time)):
            if time[i].year==year_start:
                if time[i].month ==1:
                    start = time[i]
                    ind=i
            if time[i].year==year_end:
                if time[i].month == 1:
                    end=time[i]
                    endi = i
        
        tas_mean = tas.sel(time=slice(cftime.date2num(time[ind],tas.time.units,calendar='standard'), cftime.date2num(time[endi],tas.time.units,calendar='standard'))).weighted(tas.time.diff("time")).mean(dim='time')
        return tas_mean


def build_anomalies(data,dataref,period = (1850,1900)):
    dataset1 = xr.decode_cf(data)
    dataset_ref = xr.decode_cf(dataref)
    year1 = str(period[0])
    year2=str(period[1])
    ref_periode = dataset_ref.sel(time=slice(year1,year2))
    monthly_clim = ref_periode.groupby("time.month").mean(dim="time")
    anomalies = dataset1.groupby("time.month") - monthly_clim
    return anomalies

def region_centered_poly(dataset,polyn): 
    from shapely.geometry import Point, Polygon
    polygon1 = Polygon(polyn).reverse()
    # Création de la grille lat/long
    lat = dataset.lat.values #np.linspace(50, 80, 180)  # Exemple de latitudes dans la plage de votre polygone
    lon = dataset.lon.values #np.linspace(-170, -105, 288)  # Exemple de longitudes dans la plage de votre polygone
    lon = lon -180
    # Initialiser le masque
    mask = np.zeros((len(dataset.lat), len(dataset.lon)), dtype=bool)
    
    # Remplir le masque en vérifiant si chaque point est à l'intérieur du polygone
    for i in range(0, len(dataset.lat)):
        for j in range(0, len(dataset.lon)):
            point = Point(lon[j], lat[i])  # Créer un objet Point avec la longitude et la latitude
            if polygon1.contains(point):
                mask[i][j] = True  # Marquer comme True si le point est dans le polygone
            else:
                mask[i][j] = False  # Sinon, laisser comme False
    #remettre +180 pour revenir a 0-360
    lon=lon+180
    # Convertir le masque en DataArray avec les dimensions correspondantes
    mask_xr = xr.DataArray(mask, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    
    # Appliquer le masque sur tas_mean
    masked_data = dataset.where(mask_xr,drop=True)
    # Afficher le résultat
    return masked_data
def regional_anomalie_byear(anomalies, region):
    if not isinstance(region, list):
        raise TypeError("region doit être une liste de strings ou de polygones")
    
    if isinstance(region[0], str):
        all_regional_anomalies = []
        for reg in region:
            regional_anomalie = []
            for i in range(int(anomalies.time.shape[0]/12)):
                year = int(anomalies.time[0]["time.year"]) + i
                year_str = str(year)
                regional_anomalie.append(globalmean(region_centered(anomalies.sel(time=slice(year_str,year_str)).mean(dim='time'), reg).tas).values)
            all_regional_anomalies.append(regional_anomalie)
        return all_regional_anomalies
    else:
        all_regional_anomalies = []
        for reg in region:
            regional_anomalie = []
            for i in range(int(anomalies.time.shape[0]/12)):
                year = int(anomalies.time[0]["time.year"]) + i
                year_str = str(year)
                regional_anomalie.append(globalmean(region_centered_poly(anomalies.sel(time=slice(year_str,year_str)).mean(dim='time'), reg).tas).values)
            all_regional_anomalies.append(regional_anomalie)
        return all_regional_anomalies

def global_anomalie_byear(anomalie):
    global_anomalie=[]
    for i in range(int(anomalies.time.shape[0]/12)):
        year = int(anomalies.time[0]["time.year"]) + i
        year_str=str(year)
        global_anomalie.append(globalmean(anomalies.sel(time=slice(year_str,year_str)).mean(dim='time')).tas.values)
    return global_anomalie

### FORE THE LIFTING SCHEME OPERATIONS

# Affichage du nombre d’éléments par cluster
def nbr_clust(part):
    
    unique_labels, counts = np.unique(part, return_counts=True)
    print("Nombre d’éléments par cluster :")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label} : {count} éléments")
    from collections import Counter

    counts = Counter(part)
    single_occurrences = [k for k, v in counts.items() if v == 1]
    
    print("Nombres apparaissant une seule fois :", single_occurrences)
    print("Unique clusters:", np.unique(part), "  Count:", len(np.unique(part)))

def single_partitions(x):
    # x: numpy array of shape (n_cells, n_time_steps)
    # Partition the rows of matrix x into approximately nrow(x)/2 groups

    spart = int(np.floor(x.shape[0] / 2))
    
    # Perform hierarchical clustering
    res_clust = linkage(x, method='ward')

    # Cut the dendrogram to get 'spart' clusters
    cut_clust = fcluster(res_clust, spart, criterion='maxclust')
    
    nclust = spart
    while any(np.unique(cut_clust, return_counts=True)[1] == 1):
        nclust = max(round(nclust / 2), 1)
        cut_clust = fcluster(res_clust, nclust, criterion='maxclust')
    if nclust < spart:
        k = 0  # Initialisation de l'index de regroupement
        part_clust = np.full_like(cut_clust, fill_value=np.nan, dtype=float)  # Tableau rempli de NaN
    
        for j in range(1, nclust + 1):
            ind_clust_j = np.where(cut_clust == j)[0]
            even_count = len(ind_clust_j) // 2
            ind_even = np.arange(0, 2 * even_count, 2)  # indices pairs dans le cluster
    
            part_clust[ind_clust_j[ind_even]] = k + 1 + np.arange(len(ind_even))
    
            rest_indices = np.setdiff1d(np.arange(len(ind_clust_j)), ind_even)
            if 2 * even_count == len(ind_clust_j):
                part_clust[ind_clust_j[rest_indices]] = k + 1 + np.arange(len(ind_even))
            else:
                part_clust[ind_clust_j[rest_indices]] = np.concatenate([
                    k + 1 + np.arange(len(ind_even)),
                    [k + len(ind_even)]
                ])
    
            k += len(ind_even)
    else:
        part_clust = cut_clust

    return part_clust



def ls_single_fwd(dtrain):
    # Applies the spatio-temporal lifting-scheme decomposition
    # dtrain is of dimension number of cells by number of time steps

    nmcell = dtrain.shape[0]  # Number of micro cells

    # Structures dynamiques
    npart = []
    wlet = {
        'even': [],
        'odd': [],
        'classif': [],
        'data': [],
        'nave': [],
        'nave_odd': [],
        'nave_even': []
    }

    cell_part = np.arange(nmcell)  # Cells to partition
    j = 0

    while len(cell_part) > 1:  # Loop over stages of the lifting scheme
        print(f"Performing stage {j + 1} of the lifting scheme")

        # Split cell_part into groups
        if len(cell_part) > 2:
            res_clust = single_partitions(dtrain[cell_part, :])
            nbr_clust(res_clust)
        else:
            res_clust = np.ones(len(cell_part), dtype=int)

        npart.append(len(np.unique(res_clust)))

        # Affectation des classes
        if j == 0:
            wlet['classif'].append(res_clust)
        else:
            classif_j = np.empty(nmcell, dtype=int)
            for k in range(npart[j - 1]):
                ind_classk = (wlet['classif'][j - 1] == wlet['classif'][j - 1][wlet['even'][j - 1][k]])
                classif_j[ind_classk] = res_clust[k]
            wlet['classif'].append(classif_j)

        # Séparer les éléments pairs et impairs
        even_j = np.empty(npart[j], dtype=int)
        odd_j = [None] * npart[j]
        for k in range(npart[j]):
            ind_classk = np.where(res_clust == k + 1)[0]
            even_j[k] = cell_part[ind_classk[0]]
            odd_j[k] = cell_part[ind_classk[1:]]
        wlet['even'].append(even_j)
        wlet['odd'].append(odd_j)

        # Suivi du nombre d’éléments (nave)
        if j == 0:
            nave_j = np.array([len(odd_j[k]) + 1 for k in range(npart[j])])
        else:
            ind_even_cur = np.array([np.where(wlet['even'][j - 1] == even)[0][0] for even in even_j])
            nave_j = wlet['nave'][j - 1][ind_even_cur].copy()
            for k in range(len(even_j)):
                indices = []
                for odd in odd_j[k]:
                    idx_arr = np.where(wlet['even'][j - 1] == odd)[0]
                    if len(idx_arr) > 0:
                        indices.append(idx_arr[0])
                if indices:
                    nave_j[k] += np.sum(wlet['nave'][j - 1][indices])
        wlet['nave'].append(nave_j)

        # Prédiction des impairs depuis les pairs
        for k in range(len(even_j)):
            if len(odd_j[k]) > 0:
                dtrain[odd_j[k], :] -= np.tile(dtrain[even_j[k], :], (len(odd_j[k]), 1))

        # Mise à jour des éléments pairs
        for k in range(len(even_j)):
            if len(odd_j[k]) == 0:
                continue
            if j == 0:
                nave_odd_j = np.ones(len(odd_j[k]), dtype=int)
                nave_even_j = 1
            else:
                indices_odd = [np.where(wlet['even'][j - 1] == odd)[0][0] for odd in odd_j[k]]
                nave_odd_j = wlet['nave'][j - 1][indices_odd]
                nave_even_j = wlet['nave'][j - 1][np.where(wlet['even'][j - 1] == even_j[k])[0][0]]

            if len(odd_j[k]) > 1:
                dtrain[even_j[k], :] += np.sum(nave_odd_j[:, np.newaxis] * dtrain[odd_j[k], :], axis=0) / np.sum(
                    np.concatenate(([nave_even_j], nave_odd_j)))
            else:
                dtrain[even_j[k], :] += (dtrain[odd_j[k][0], :] * nave_odd_j) / np.sum(
                    np.concatenate(([nave_even_j], nave_odd_j)))

            # Stocker les valeurs de nave_odd et nave_even
            wlet['nave_odd'].append(nave_odd_j)
            wlet['nave_even'].append(nave_even_j)

        wlet['data'].append(dtrain.copy())
        cell_part = even_j
        j += 1

    return wlet

def ls_bwds(part_wlet, nave_wlet, coeff_wlet, lr_valid):
    npart = [len(e) for e in part_wlet['even']]
    ind_even = part_wlet['even'][-1]
    
    n_rows = coeff_wlet.shape[0] + len(ind_even)
    n_cols = coeff_wlet.shape[1]
    data_train = np.empty((n_rows, n_cols))
    data_train[:] = np.nan  # On initialise à NaN (comme en R)

    mask = np.ones(n_rows, dtype=bool)
    mask[ind_even] = False
    data_train[mask, :] = coeff_wlet
    data_train[ind_even, :] = lr_valid

    for j in reversed(range(len(npart))):  # from last to 0
        for k in range(npart[j]):
            odd_indices = part_wlet['odd'][j][k]
            even_index = part_wlet['even'][j][k]

            if j == 0:
                nave_odd = np.ones(len(odd_indices))
                nave_even = 1
            else:
                nave_odd = [
                    nave_wlet[j - 1][np.where(part_wlet['even'][j - 1] == idx)[0][0]]
                    for idx in odd_indices
                ]
                even_idx = part_wlet['even'][j][k]
                nave_even = nave_wlet[j - 1][np.where(part_wlet['even'][j - 1] == even_idx)[0][0]]


            if len(odd_indices) > 0:
                if len(odd_indices) > 1:
                    weighted_sum = np.sum(
                        np.array(nave_odd)[:, None] * data_train[odd_indices, :], axis=0
                    )
                    data_train[even_index, :] -= weighted_sum / (sum(nave_odd) + nave_even)
                else:
                    data_train[even_index, :] -= (
                        data_train[odd_indices[0], :] * nave_odd[0]
                    ) / (nave_even + nave_odd[0])

        for k in range(npart[j]):
            odd_indices = part_wlet['odd'][j][k]
            even_index = part_wlet['even'][j][k]

            if len(odd_indices) > 0:
                repeated = np.tile(data_train[even_index, :], (len(odd_indices), 1))
                data_train[odd_indices, :] += repeated

       

    return data_train


def build_data_for_lift(data_region):
    all_data = []
    nb_years = int(data_region.values.shape[0]/12)
    for i in range (0,data_region.values.shape[0],12):
        data_map = data_region[i].values.flatten()
        all_data.append([data_map])
    data = np.array(all_data)
    data = data.reshape((nb_years,data_region.values.shape[1]*data_region.values.shape[2]))
    data = data.T
    ar3=data[~np.isnan(data).any(axis=1)]
    return ar3
    
def lifting_scheme(data,y):
    d = ls_single_fwd(data)
    part_wlet = {'even':[],'odd':[]}
    part_wlet['even'] = d['even']
    part_wlet['odd'] = d['odd']
    nave_wlet = d['nave']
    ind_even = d['even'][-1]
    coeff_wlet = np.delete(d['data'][-1], ind_even, axis=0)
    X_train = y
    Y_train = coeff_wlet
    knn = KNeighborsRegressor(n_neighbors=1, weights='distance')  # ou 'uniform'
    knn.fit(X_train.reshape(-1,1), Y_train.T)  

    coeff_pred = knn.predict(y.reshape(-1, 1))
    reconstruct_data = ls_bwds(part_wlet,nave_wlet,coeff_pred.T,y)
    return reconstruct_data
    
def rebuild_data_for_mapping(data_region,data_to_rebuild):
    nb_years = int(data_region.shape[0]/12)
    mask_valid = ~np.isnan(data_region[0])  
    reconstructed_full = np.full((nb_years,data_region.shape[1] ,data_region.shape[2]), np.nan)
    for t in range(nb_years):
        reconstructed_full[t][mask_valid] = data_to_rebuild[:, t]
    return reconstructed_full

def apply_lifitng_scheme(data_region,y):
    return rebuild_data_for_mapping(data_region,lifting_scheme(build_data_for_lift(data_region),y))

def show_result(data_reconstructed, year=None):
    if year is not None:
        year_idx = year
    else:
        year_idx = 0

    if year_idx >= len(data_reconstructed):
        raise ValueError(f"L'année indexée {year_idx} dépasse la taille des données ({len(data_reconstructed)}).")

    carte = data_reconstructed[year_idx]

    # Affichage
    plt.figure(figsize=(8, 6))
    plt.imshow(carte, origin='lower', cmap='coolwarm')  # tu peux changer la colormap si tu veux
    plt.colorbar(label='Température °C')
    plt.title(f'Données reconstruites - Année index {year_idx}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(False)
    plt.show()
    
def polygon_from_kml(nomfichier,path):
    from fastkml import kml

    with open(path+nomfichier, 'rt', encoding='utf-8') as f:
        doc = f.read()

    k = kml.KML()
    k.from_string(doc)
    polygon_from_kml = []
    
    # On parse le document KML une seule fois
    parsed_doc = k.from_string(doc)
    
    for i in range(1, len(parsed_doc.features[0].features)):
        polygon = []
        for point in parsed_doc.features[0].features[i].geometry.coords[0]:
            polygon.append(point[:2])  # On garde seulement (longitude, latitude)
        polygon_from_kml.append(polygon)
    
    return polygon_from_kml
    
def global_anomalie_byear(anomalies):
    global_anomalie=[]
    for i in range(int(anomalies.time.shape[0]/12)):
        year = int(anomalies.time[0]["time.year"]) + i
        year_str=str(year)
        global_anomalie.append(globalmean(anomalies.sel(time=slice(year_str,year_str)).mean(dim='time')).tas.values)
    return global_anomalie






