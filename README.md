# The Robin Hood Protocol

All of the source code, results and figures in this repository support my dissertation on:

**Transfer Learning for FX Volatility-Surface Reconstruction under Controlled Missingness and Observable Market Regimes**

This repository includes all of the code (data preparation, models, evaluation functions, etc.), as well as all of the saved experiment results (in a JSON format), and exported CSVs and figures that support my research. In particular, my research asks if learning from liquid FX markets can be used to help recover deliberately hidden volatility quotes in less liquid markets, as opposed to using other simple benchmarks.

## Contents

| Location | Contents |
|---|---|
| `src/` | Data preparation, models, evaluation functions, plotting code. |
| `results/` | Saved experiment results in JSON format. |
| `outputs/` | Supporting CSV tables, chart exports. |
| `figures/` | Figures exported for the paper, named after their figure numbers. |
| `requirements.txt` | List of required Python packages. |

## Where to find the results

As stated above, I have kept track of all of the filenames mentioned in the paper.

- **Figure 5.7:** `results/v2_regime_switching_framework.json` — results for the regime-switching framework that uses market conditions to select which method to use to perform the reconstruction.
- **Figure 5.8:** `results/v2_outage_aware_transfer_vae_experiment.json` — comparison of ordinary vs. outage-aware VAE training.
- **Figure 5.8:** `results/v2_dynamic_pca_experiment.json` — results for the dynamic-PCA benchmark on consecutive quote outages.

Experiment results and CSV tables are organised by experiment names.

## Running the code

Python 3.11 or later must be installed. From the repository folder, install the dependencies listed in `requirements.txt` by running:

```bash
python -m pip install -r requirements.txt
```

Using the saved results provided, regenerate the missingness schematic, switching-rule comparison and consecutive-outage comparison by running:

```bash
python src/plot_final_charts.py
```

Charts generated will be placed into `outputs/`. All of the final numbered images shown in the paper are located in `figures/`.

## Access to data and limitations

Bloomberg raw data and quote-level processed datasets are not available within this repository. Separately authorised access to these input data sources would be needed to rerun the data preparation and model experiments.
