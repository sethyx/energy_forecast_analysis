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
│   ├── ha_convert.py              # Home Assistant data converter
│   ├── ha_converted.csv           # Converted HA dataset
│   ├── uci_dataset.csv            # UCI power consumption dataset
│   └── uci_resample.py           # UCI data resampling script
├── output/
│   ├── daily_calc.xlsx           # Daily calculations
│   ├── ha_output.txt             # HA prediction results
│   └── uci_output.txt            # UCI prediction results
├── forecast.py                    # Main forecasting script
└── LICENSE                        # Apache 2.0 license
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
