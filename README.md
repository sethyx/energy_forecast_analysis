# Energy Forecast Analysis

A Python-based energy consumption forecasting system using AutoGluon TimeSeries models to predict household power usage.

## Overview

This project implements time series forecasting for household power consumption using multiple models:

### Statistical Models
- AutoETS
- AutoARIMA

### Machine Learning Models
- DeepAR 
- WaveNet

### NLP-based Models
- ChronosZeroShot
- ChronosFineTuned

## Datasets

The system works with two datasets:

1. **Home Assistant Dataset** (`datasets/ha_converted.csv`)
2. **UCI Power Consumption Dataset** (`datasets/uci_dataset.csv`)

## Project Structure

```
├── autogluon_power_forecast_ha/    # Home Assistant model files
├── autogluon_power_forecast_uci/   # UCI dataset model files
├── datasets/
│   ├── ha_convert.py               # Home Assistant data converter
│   ├── ha_converted.csv            # HA Converted dataset
│   ├── ha_raw.csv                  # HA Raw dataset
│   ├── uci_dataset.csv             # UCI Resampled dataset
│   ├── uci_raw.txt.zip             # UCI Raw dataset (compressed), from https://doi.org/10.24432/C58K54
│   └── uci_resample.py             # UCI data resampling script
├── output/
│   ├── images/                     # Exported plot images
│   ├── daily_calc.xlsx             # Daily calculations
│   ├── ha_output.txt               # HA prediction results
│   └── uci_output.txt              # UCI prediction results
├── forecast.py                     # Main forecasting script
└── LICENSE                         # Apache 2.0 license
```

## Usage

Run forecasting with different datasets:

```bash
# Home Assistant dataset
python forecast.py --ha --plot --daily

# UCI dataset
python forecast.py --uci --plot --daily
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--ha` | Use Home Assistant dataset |
| `--uci` | Use UCI dataset |
| `--plot` | Generate forecast plots |
| `--daily` | Generate daily consumption forecasts |

## Requirements

- Python 3.9+
- AutoGluon TimeSeries 1.4.0+
- pandas
- matplotlib

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citations

- Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54.
- Shchur, O., Turkmen, C., Erickson, N., Shen, H., Shirkov, A., Hu, T., Wang, Y. (2023). AutoGluon-TimeSeries: AutoML for Probabilistic Time Series Forecasting. arXiv preprint arXiv:2308.05566.
- Salinas, D., Flunkert, V., Gasthaus, J. (2017). DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks. arXiv preprint arXiv:1704.04110.
- Oord, A.V.D., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., Kavukcuoglu, K. (2016). WaveNet: A Generative Model for Raw Audio. arXiv preprint arXiv:1609.03499.
- Borovykh, A., Bohte, S., Oosterlee, C.W. (2017). Conditional Time Series Forecasting with Convolutional Neural Networks. arXiv preprint arXiv:1703.04691.
- Ansari, A.F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S.S., Arango, S.P., Kapoor, S., Zschiegner, J., Maddix, D.C., Wang, H., Mahoney, M.W., Torkkola, K., Wilson, A.G., Bohlke-Schneider, M., Wang, Y. (2024). Chronos: Learning the Language of Time Series. arXiv preprint arXiv:2403.07815.
