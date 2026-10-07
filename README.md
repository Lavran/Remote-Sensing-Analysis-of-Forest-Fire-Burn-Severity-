# Remote Sensing Analysis of Forest Fire Burn Severity

Hierarchical Bayesian evaluation of Landsat 8 and Sentinel-2 burn severity indices (dNBR, RBR, RdNBR, RdNDVI) against field Composite Burn Index (CBI) plots from five US fires, building on Howe et al. (2022).

The full write-up is the Quarto document `Hierarchical Bayesian Evaluation of Landsat 8 and Sentinel-2 Derived Burn Severity Indices Against the Composite Burn Index.qmd`, rendered to the `.html` file of the same name. GitHub displays HTML files as source code, so download the HTML and open it in a browser to read the report.

## Abstract

Understanding post-fire burn severity is essential for predicting ecosystem recovery, erosion, habitat change, and management needs. The Composite Burn Index (CBI) provides a field measure of ecological change, but field observations are too sparse to characterize most burned landscapes without satellite data. This study evaluated four satellite burn severity indices, dNBR, RBR, RdNBR, and RdNDVI, against CBI measurements from five fires in the United States. Landsat 8 Collection 2 surface reflectance imagery was processed for all five fires, and a direct comparison with Sentinel-2 imagery was conducted for Legion Lake, the only fire with suitable imagery from both sensors. RBR produced the strongest pooled relationship with CBI in this dataset, although index performance varied among fires. Landsat 8 and Sentinel-2 produced similar validation metrics at Legion Lake, while their mapped severity patterns differed at native output resolutions. The relationship between RBR and CBI was then evaluated with pooled, hierarchical, and hierarchical heteroscedastic Bayesian models. Plot-level cross-validation ranked both hierarchical models above the pooled model, suggesting that accounting for differences among fires improved prediction within the represented fires. The heteroscedastic model did not materially improve mean prediction, but posterior predictive checks showed that it represented the increase in residual variation at higher CBI values more accurately. These results show that variation among fires and across the severity gradient is an important component of satellite burn severity calibration and should be represented directly rather than summarized by a single pooled fit statistic.

## Contents

| File | Description |
|---|---|
| `Hierarchical Bayesian Evaluation of Landsat 8 and Sentinel-2 Derived Burn Severity Indices Against the Composite Burn Index.qmd` | Quarto source of the report: methods, results, discussion and code |
| `Hierarchical Bayesian Evaluation of Landsat 8 and Sentinel-2 Derived Burn Severity Indices Against the Composite Burn Index.html` | Rendered report |
| `burn_severity_analysis.py` | Full pipeline as a script: Earth Engine processing, index validation, figures, and the three PyMC models with the LOO comparison |
| `loo_compare.csv` | PSIS-LOO model comparison, written by the script and read by the report |
| `figure1_counties.png`, `legion_compare.png`, `bayes_offsets.png`, `bayes_ppc.png` | Figures written by the script and embedded in the report |
| `rbr_by_fire.png` | RBR versus CBI by fire, written by the script (the report draws this figure itself) |
| `cbi_data_review_v4.zip` | CBI plot shapefile used in the analysis |
| `requirements.txt` | Python packages |

## Reproducing the analysis

1. Install [Quarto](https://quarto.org) and the Python packages: `pip install -r requirements.txt`.
2. Authenticate Earth Engine (`earthengine authenticate`) and replace the Cloud project in `ee.Initialize(project="howe-et-al")` with your own, in both the script and the report.
3. Unzip `cbi_data_review_v4.zip` into a folder named `cbi_data_review_v4`, so the shapefile is at `cbi_data_review_v4/conus_cbi_v4.shp`.
4. Run `burn_severity_analysis.py` from this folder. It writes `loo_compare.csv` and the figures listed above. The script sets its working directory with `os.chdir` to a path on the author's computer, so edit or remove that line first.
5. Render the report with `quarto render`. The validation tables and the RBR-by-fire figure are computed during rendering. The Bayesian results, maps and location figure are read from the files written in step 4.

## Data sources

- CBI plots: USGS national Composite Burn Index database ([doi:10.5066/P97UMU6K](https://doi.org/10.5066/P97UMU6K))
- Landsat 8 Collection 2 Level-2 surface reflectance and Sentinel-2 harmonized Level-1C imagery with Cloud Score+, through Google Earth Engine
- GFCC 2015 tree canopy cover and MTBS burned-area boundaries, through Google Earth Engine

Full references are in the report.

## Author

Lavran Pagano
