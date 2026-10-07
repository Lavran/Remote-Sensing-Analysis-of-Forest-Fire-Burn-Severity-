# ============================================================
# 0. SETUP
# ============================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.optimize import curve_fit

import ee
import geemap

import requests
import rasterio
from rasterio.features import geometry_mask

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature

import pymc as pm
import arviz as az

ee.Initialize(project="howe-et-al")
os.chdir(r"C:\Users\lavra\OneDrive\Documents\Python_scripts\Comparing Sentinel-2 and Landsat 8 Howe et al")

# ============================================================
# 1. FUNCTIONS
# ============================================================
def composite(collection, year):
    # extended-composite window: Jun 1 – Sep 30 (Howe et al.).
    # Earth Engine's end date is exclusive, so use 10-01 to include all of Sep 30.
    return collection.filterDate(f'{year}-06-01', f'{year}-10-01').mean()

def nbr(img):
    return img.normalizedDifference(['NIR', 'SWIR2']).rename('NBR')

def ndvi(img):
    return img.normalizedDifference(['NIR', 'RED']).rename('NDVI')

def fc_to_df(fc):
    feats = fc.getInfo()['features']
    df = pd.DataFrame([f['properties'] for f in feats])
    return df.apply(pd.to_numeric, errors='coerce')

def add_indices(pre, post):
    nbr_pre, nbr_post = nbr(pre), nbr(post)
    dnbr  = nbr_pre.subtract(nbr_post).multiply(1000).rename('dNBR')      # Eq. 2
    rbr   = dnbr.divide(nbr_pre.add(1.001)).rename('RBR')                 # Eq. 3
    rdnbr = dnbr.divide(nbr_pre.abs().max(0.001).sqrt()).rename('RdNBR')  # Eq. 4
    ndvi_pre, ndvi_post = ndvi(pre), ndvi(post)
    dndvi  = ndvi_pre.subtract(ndvi_post).multiply(1000)
    rdndvi = dndvi.divide(ndvi_pre.abs().max(0.001).sqrt()).rename('RdNDVI')  # Eq. 7
    return ee.Image([dnbr, rbr, rdnbr, rdndvi])

def sensor(fire_ee, name):
    def landsat():
        def prep(img):
            qa = img.select('QA_PIXEL')
            mask = (qa.bitwiseAnd(1 << 1).eq(0)
                      .And(qa.bitwiseAnd(1 << 3).eq(0))
                      .And(qa.bitwiseAnd(1 << 4).eq(0)))
            out = (img.select(['SR_B5', 'SR_B7', 'SR_B4'], ['NIR', 'SWIR2', 'RED'])
                      .multiply(0.0000275).add(-0.2).updateMask(mask))
            return out.copyProperties(img, ['system:time_start'])
        return ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(fire_ee).map(prep)

    def sentinel():
        cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        def prep(img):
            mask = img.select('cs_cdf').gte(0.6)                      # Cloud Score+
            out = (img.select(['B8', 'B12', 'B4'], ['NIR', 'SWIR2', 'RED'])
                      .multiply(0.0001).updateMask(mask))
            return out.copyProperties(img, ['system:time_start'])
        return (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                  .filterBounds(fire_ee).linkCollection(cs, ['cs_cdf']).map(prep))

    return landsat() if name == 'landsat' else sentinel()

def run_fire(fire, name):
    fire_gdf = CBI[CBI['FireName'] == fire].to_crs(4326)
    fire_ee  = geemap.geopandas_to_ee(fire_gdf)
    year     = pd.to_datetime(fire_gdf['FireDate'].iloc[0]).year
    col  = sensor(fire_ee, name)
    pre  = composite(col, year - 1)
    post = composite(col, year + 1)
    indices = add_indices(pre, post)
    # Both sensors sampled at a common 30 m scale: the validation compares spectral/atmospheric
    # differences, NOT Sentinel's finer resolution (that effect appears only in the maps).
    samples = indices.addBands(canopy).sampleRegions(
        collection=fire_ee, properties=['Cbi'], scale=30)
    df = fc_to_df(samples).query('canopy >= 5')
    df['fire'], df['sensor'] = fire, name
    return df

def fire_image(fire, name):
    fire_gdf = CBI[CBI['FireName'] == fire].to_crs(4326)
    fire_ee  = geemap.geopandas_to_ee(fire_gdf)
    year     = pd.to_datetime(fire_gdf['FireDate'].iloc[0]).year
    col  = sensor(fire_ee, name)
    pre  = composite(col, year - 1)
    post = composite(col, year + 1)
    indices = add_indices(pre, post)
    region  = fire_ee.geometry().bounds().buffer(2000)
    return indices, region

def results(df, by_fire=False):
    index_names = ["dNBR", "RBR", "RdNBR", "RdNDVI"]
    def score(sub):
        rows = []
        for idx in index_names:
            s, i, r, p, se = stats.linregress(sub[idx], sub["Cbi"])
            pred = s * sub[idx] + i
            rmse = np.sqrt(np.mean((sub["Cbi"] - pred) ** 2))
            rows.append({"index": idx, "R2": r**2, "RMSE": rmse, "n": len(sub)})
        return pd.DataFrame(rows)
    if not by_fire:
        return score(df).round(3)
    out = [score(sub).assign(fire=fire) for fire, sub in df.groupby("fire")]
    return pd.concat(out, ignore_index=True).round(3)

def nls_model(cbi, a, b, c):
    return a + b * np.exp(c * cbi)

def rbr_thresholds(df, index='RBR'):
    (a, b, c), _ = curve_fit(nls_model, df['Cbi'], df[index], p0=[0, 10, 1], maxfev=10000)
    return {'moderate': nls_model(1.25, a, b, c),
            'high':     nls_model(2.25, a, b, c)}

def classify(index_img, thr):
    return (index_img.gte(thr['moderate']).add(index_img.gte(thr['high'])).rename('severity'))

def plot_severity(fire, name, thr, ax):
    scale = 30 if name == 'landsat' else 10   # Landsat 30 m; Sentinel 10 m (pseudo for RBR: 20 m SWIR2 resampled)
    img, region = fire_image(fire, name)
    perim_fc = mtbs.filterBounds(region).filter(ee.Filter.eq('Incid_Name', fire.upper()))
    perim    = perim_fc.geometry()

    classes = classify(img.select('RBR'), thr).add(1).clip(perim)
    url = classes.getDownloadURL({'scale': scale, 'region': perim, 'format': 'GEO_TIFF'})
    tif = f'{name}_classes.tif'
    open(tif, 'wb').write(requests.get(url).content)

    perim_gdf = gpd.GeoDataFrame.from_features(perim_fc.getInfo()['features'], crs='EPSG:4326')
    with rasterio.open(tif) as src:
        arr = src.read(1).astype(float)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        outside = geometry_mask(perim_gdf.geometry, out_shape=arr.shape, transform=src.transform)
    arr = np.ma.masked_where(outside, arr)

    ax.set_facecolor('white')
    ax.imshow(arr, extent=extent, origin='upper', transform=ccrs.PlateCarree(),
              cmap=ListedColormap(['#ffffb2', '#fd8d3c', '#bd0026']), vmin=1, vmax=3)
    ax.add_geometries(perim_gdf.geometry, crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black', linewidth=1.5)
    ax.set_title(f'{name.capitalize()} ({scale} m)')
    return extent

def add_scalebar(ax, extent, km=5):
    lat_c = (extent[2] + extent[3]) / 2
    deg   = km / (111.32 * np.cos(np.radians(lat_c)))
    x0    = extent[0] + 0.05 * (extent[1] - extent[0])
    x1    = x0 + deg
    y0    = extent[2] + 0.08 * (extent[3] - extent[2])
    ax.plot([x0, x1], [y0, y0], color='black', lw=3, transform=ccrs.PlateCarree())
    ax.text((x0 + x1) / 2, y0, f'{km} km', transform=ccrs.PlateCarree(),
            ha='center', va='bottom', fontsize=10)

# ============================================================
# 2. DATA — CBI plots, canopy mask, fire perimeters
# ============================================================
CBI = gpd.read_file("cbi_data_review_v4/conus_cbi_v4.shp")
CBI['year'] = pd.to_datetime(CBI['FireDate'], errors='coerce').dt.year

canopy = (ee.ImageCollection('NASA/MEASURES/GFCC/TC/v3')
            .filterDate('2015-01-01', '2015-12-31')
            .select('tree_canopy_cover').mosaic().rename('canopy'))

mtbs = ee.FeatureCollection('projects/sat-io/open-datasets/MTBS/burned_area_boundaries')

# ============================================================
# 3. VALIDATION — index vs field CBI (R2, RMSE)
# ============================================================
fire_list = CBI.loc[CBI['year'] >= 2016, 'FireName'].unique()
landsat = pd.concat([run_fire(f, 'landsat') for f in fire_list], ignore_index=True)
landsat_results      = results(landsat)
landsat_results_fire = results(landsat, by_fire=True)

s2_fires = CBI.loc[CBI['year'] >= 2017, 'FireName'].unique()
sentinel = pd.concat([run_fire(f, 'sentinel') for f in s2_fires], ignore_index=True)
sentinel_results = results(sentinel)

# ============================================================
# 4. FIGURES
# ============================================================
# --- 4a. RBR vs CBI by fire and sensor ---
idx = 'RBR'
ls = landsat.copy(); ls['group'] = ls['fire'] + ' (Landsat)'
s2 = sentinel[sentinel['fire'] == 'Legion Lake'].copy(); s2['group'] = 'Legion Lake (Sentinel)'
combo = pd.concat([ls, s2], ignore_index=True)

def fit_trend(x, y):
    """Exponential NLS (Howe Eq. 8) with a linear fallback for small/failing groups."""
    xs = np.linspace(x.min(), x.max(), 100)
    if len(x) >= 6:
        try:
            (a, b, c), _ = curve_fit(nls_model, x, y, p0=[0, 10, 1], maxfev=10000)
            return xs, nls_model(xs, a, b, c)
        except Exception:
            pass
    s, i, *_ = stats.linregress(x, y)
    return xs, s * xs + i

markers = ['o', 's', '^', 'D', 'v', 'P']
fig, ax = plt.subplots(figsize=(8, 6))
for (grp, sub), m in zip(combo.groupby('group'), markers):
    sc = ax.scatter(sub['Cbi'], sub[idx], marker=m, s=35, alpha=0.6, label=grp)
    color = sc.get_facecolor()[0]
    xs, ys = fit_trend(sub['Cbi'].values, sub[idx].values)
    ax.plot(xs, ys, color=color, linewidth=2)
ax.set_xlabel('Field CBI'); ax.set_ylabel(idx)
ax.set_title(f'{idx} vs field CBI, by fire and sensor (NLS)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('rbr_by_fire.png', dpi=200, bbox_inches='tight')
plt.show()

# --- 4b. Location map (fires by county) ---
study = CBI[CBI['FireName'].isin(fire_list)].to_crs(4326)
study['lon'] = study.geometry.x
study['lat'] = study.geometry.y
cent = study.groupby('FireName')[['lon', 'lat']].mean().reset_index()

groups = {
    'Arizona':      ['Fuller', 'Soshone'],
    'South Dakota': ['Legion Lake'],
    'Tennessee':    ['Chimney Tops', 'Cobbly Nob'],
}

shp = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_2_counties')
counties = gpd.read_file(shp).to_crs(4326)
# grab the county-name field (Natural Earth naming varies by version)
name_col = next((c for c in ['NAME', 'NAME_2', 'NAMELSAD', 'NAMEASCII', 'name']
                 if c in counties.columns), None)
counties['county'] = counties[name_col] if name_col else counties.index.astype(str)

pts = gpd.GeoDataFrame(cent, geometry=gpd.points_from_xy(cent.lon, cent.lat), crs=4326)
pts = gpd.sjoin(pts, counties[['geometry', 'county']], how='left', predicate='within')

fig, axes = plt.subplots(1, len(groups), figsize=(15, 5),
                         subplot_kw={'projection': ccrs.PlateCarree()})
for ax, (region, names) in zip(axes, groups.items()):
    sub = pts[pts['FireName'].isin(names)]
    fire_counties = counties.loc[sub['index_right'].dropna().unique()]
    minx, miny, maxx, maxy = fire_counties.total_bounds
    pad = 0.3
    ax.set_extent([minx - pad, maxx + pad, miny - pad, maxy + pad], crs=ccrs.PlateCarree())
    view = counties.cx[minx - pad:maxx + pad, miny - pad:maxy + pad]
    ax.add_geometries(view.geometry, crs=ccrs.PlateCarree(),
                      facecolor='#f5f5f5', edgecolor='lightgray', linewidth=0.4)
    ax.add_geometries(fire_counties.geometry, crs=ccrs.PlateCarree(),
                      facecolor='#ffe5b4', edgecolor='black', linewidth=0.9)
    ax.scatter(sub.lon, sub.lat, transform=ccrs.PlateCarree(),
               s=70, color='red', edgecolor='black', zorder=5)
    for _, r in sub.iterrows():
        ax.text(r.lon + 0.03, r.lat + 0.03, r.FireName, transform=ccrs.PlateCarree(),
                fontsize=9, fontweight='bold')
    cnames = list(sub['county'].dropna().unique())
    if len(cnames) == 1:
        ax.set_title(f'{cnames[0]} County, {region}')
    elif len(cnames) > 1:
        ax.set_title(' & '.join(cnames) + f' Counties, {region}')
    else:
        ax.set_title(region)
fig.suptitle('CBI Fire Locations', fontsize=14)
plt.savefig('figure1_counties.png', dpi=200, bbox_inches='tight')
plt.show()

# --- 4c. Severity comparison (Legion Lake, Landsat 30 m vs Sentinel 10 m) ---
thr_ls = rbr_thresholds(landsat)
thr_s2 = rbr_thresholds(sentinel)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                               subplot_kw={'projection': ccrs.PlateCarree()})
fig.patch.set_facecolor('white')
plot_severity('Legion Lake', 'landsat',  thr_ls, ax1)
ext = plot_severity('Legion Lake', 'sentinel', thr_s2, ax2)
add_scalebar(ax2, ext)
fig.subplots_adjust(bottom=0.18)
fig.legend(handles=[Patch(facecolor='#ffffb2', label='Low'),
                    Patch(facecolor='#fd8d3c', label='Moderate'),
                    Patch(facecolor='#bd0026', label='High')],
           loc='lower center', bbox_to_anchor=(0.12, 0.02), ncol=1, title='Burn severity')
fig.suptitle('Legion Lake (2017) — RBR burn severity: Landsat 30 m vs Sentinel 10 m', fontsize=14)
fig.savefig('legion_compare.png', dpi=200, bbox_inches='tight')
plt.show()

# ============================================================
# 5. BAYESIAN CALIBRATION (RBR) — extension beyond the paper
# ============================================================
# Data: model RBR as a function of field CBI, with a per-fire index for the hierarchy.
cbi = landsat['Cbi'].values
rbr = landsat['RBR'].values
fire_idx, fire_labels = pd.factorize(landsat['fire'])
n_fires = len(fire_labels)
coords = {'fire': list(fire_labels)}   # names the per-fire dimension so plots label by fire, not a[0..4]

SAMPLE = dict(draws=1000, tune=1000, chains=4, cores=1, random_seed=42,   # cores=1: avoid Windows spawn crash when run as a script
              target_accept=0.95, idata_kwargs=dict(log_likelihood=True))

# --- Model 1: pooled nonlinear (Bayesian version of Howe's NLS, Eq. 8) ---
with pm.Model() as m_pooled:
    a = pm.Normal('a', mu=0, sigma=50)
    b = pm.HalfNormal('b', sigma=50)          # positive: RBR rises with CBI
    c = pm.HalfNormal('c', sigma=1)           # positive rate
    sigma = pm.HalfNormal('sigma', sigma=100)
    mu = a + b * pm.math.exp(c * cbi)
    pm.Normal('obs', mu=mu, sigma=sigma, observed=rbr)
    idata_pooled = pm.sample(**SAMPLE)

# --- Model 2: + hierarchy (per-fire offset, non-centered) ---
with pm.Model(coords=coords) as m_hier:
    mu_a     = pm.Normal('mu_a', mu=0, sigma=50)
    sigma_a  = pm.HalfNormal('sigma_a', sigma=50)
    a_offset = pm.Normal('a_offset', mu=0, sigma=1, dims='fire')
    a = pm.Deterministic('a', mu_a + a_offset * sigma_a, dims='fire')
    b = pm.HalfNormal('b', sigma=50)
    c = pm.HalfNormal('c', sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=100)
    mu = a[fire_idx] + b * pm.math.exp(c * cbi)
    pm.Normal('obs', mu=mu, sigma=sigma, observed=rbr)
    idata_hier = pm.sample(**SAMPLE)

# --- Model 3: + heteroscedastic noise (scatter grows with CBI) ---
with pm.Model(coords=coords) as m_het:
    mu_a     = pm.Normal('mu_a', mu=0, sigma=50)
    sigma_a  = pm.HalfNormal('sigma_a', sigma=50)
    a_offset = pm.Normal('a_offset', mu=0, sigma=1, dims='fire')
    a = pm.Deterministic('a', mu_a + a_offset * sigma_a, dims='fire')
    b = pm.HalfNormal('b', sigma=50)
    c = pm.HalfNormal('c', sigma=1)
    d = pm.Normal('d', mu=4, sigma=1)          # baseline log-scatter
    e = pm.Normal('e', mu=0, sigma=1)          # how log-scatter changes with CBI
    sigma = pm.Deterministic('sigma', pm.math.exp(d + e * cbi))
    mu = a[fire_idx] + b * pm.math.exp(c * cbi)
    pm.Normal('obs', mu=mu, sigma=sigma, observed=rbr)
    idata_het = pm.sample(**SAMPLE)

# --- Out-of-sample model comparison (LOO cross-validation) ---
loo_compare = az.compare({'pooled': idata_pooled,
                          'hierarchical': idata_hier,
                          'heteroscedastic': idata_het})
print(loo_compare)
loo_compare.to_csv('loo_compare.csv')   # saved so the report can display it as a table

# --- Posterior predictive check (final model) ---
pm.sample_posterior_predictive(idata_het, model=m_het, extend_inferencedata=True)
az.plot_ppc(idata_het, num_pp_samples=100)
plt.savefig('bayes_ppc.png', dpi=200, bbox_inches='tight')
plt.show()

# --- Posterior of the per-fire intercepts (partial pooling) ---
axes = az.plot_forest(idata_het, var_names=['a'], combined=True)
ax = axes[0]
# strip ArviZ's 'a[...]' variable wrapper so rows read as plain fire names
ax.set_yticklabels([t.get_text().replace('a[', '').strip('[]')
                    for t in ax.get_yticklabels()])
ax.set_title('Per-fire baseline intercept (posterior)')
plt.savefig('bayes_offsets.png', dpi=200, bbox_inches='tight')
plt.show()

# --- Fit metrics on the RBR scale ---
post   = idata_het.posterior
a_hat  = post['a'].mean(('chain', 'draw')).values
b_hat  = post['b'].mean().item()
c_hat  = post['c'].mean().item()
mu_pred = a_hat[fire_idx] + b_hat * np.exp(c_hat * cbi)
resid   = rbr - mu_pred
rmse = np.sqrt(np.mean(resid ** 2))
r2   = 1 - np.sum(resid ** 2) / np.sum((rbr - rbr.mean()) ** 2)
print('R2 =', round(r2, 3), '  RMSE(RBR) =', round(rmse, 1))
