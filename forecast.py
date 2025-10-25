import pandas as pd
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
import traceback
import argparse
import matplotlib

def parse_arguments():
    """
    Parse command line arguments for the forecasting script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
            --ha: Use Home Assistant dataset
            --uci: Use UCI dataset
            --plot: Generate forecast plots
            --daily: Generate daily forecasts
    """
    parser = argparse.ArgumentParser(description='Energy consumption forecasting with AutoGluon')
    parser.add_argument('--ha', action='store_true', help='Load Home Assistant dataset from csv/ha_converted.csv')
    parser.add_argument('--uci', action='store_true', help='Load UCI dataset from csv/uci_dataset.csv')
    parser.add_argument('--plot', action='store_true', help='Enable plot generation')
    
    args = parser.parse_args()
    if not (args.ha or args.uci):
        parser.error("At least one dataset must be specified (--ha or --uci)")
    return args

def load_dataset(args):
    """
    Load and prepare the dataset based on command line arguments.
    
    Args:
        args (argparse.Namespace): Command line arguments indicating dataset choice
        
    Returns:
        TimeSeriesDataFrame: Prepared dataset in AutoGluon format
        
    Raises:
        SystemExit: If the dataset file is not found
    """
    filename = "uci_dataset.csv" if args.uci else "ha_converted.csv"
    try:
        df = pd.read_csv(
            os.path.join("datasets", filename),
            parse_dates=["timestamp"],
            index_col="timestamp"
        )
        print("Successfully loaded dataset")
        df["item_id"] = "household_energy"
        return TimeSeriesDataFrame(df.reset_index())
    except FileNotFoundError:
        print("Error: dataset file not found in datasets/" + filename)
        exit()

def split_data(full_data, args, prediction_length=48):
    """
    Split the dataset into training and test sets.
    
    Args:
        full_data (TimeSeriesDataFrame): Complete dataset
        args (argparse.Namespace): Command line arguments
        prediction_length (int): Number of time steps to predict (default: 48 hours)
        
    Returns:
        tuple: (train_data, test_data) splits of the TimeSeriesDataFrame
    """
    shift_test_data_rows = 0 if args.uci else 1800
    train_data, test_data = full_data.train_test_split(prediction_length, len(full_data) - shift_test_data_rows)
    print(f"Training data size: {len(train_data)} points")
    print(f"Test data size:     {len(test_data)} points")
    return train_data, test_data

def get_or_train_model(train_data, args, prediction_length):
    """
    Load existing model or train a new one if not available.
    
    Args:
        train_data (TimeSeriesDataFrame): Training dataset
        args (argparse.Namespace): Command line arguments
        prediction_length (int): Number of time steps to predict
        
    Returns:
        TimeSeriesPredictor: Trained or loaded AutoGluon predictor
    """
    # Model configuration
    model_path = "autogluon_energy_forecast_uci" if args.uci else "autogluon_energy_forecast_ha"

    # Load existing model if available
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        return TimeSeriesPredictor.load(path=model_path)

    # Train new model with multiple forecasting approaches
    print(f"Training new model, will save to {model_path}")
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=model_path,
        target="hourly_consumption",
        eval_metric="WQL"
    )

    # Define model configurations for different approaches
    hyperparameters = {
        "AutoETS": {},
        "AutoARIMA": {},
        "DeepAR": {},
        "WaveNet": {},
        "Chronos": [
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    }

    predictor.fit(
        train_data,
        num_val_windows=3,
        hyperparameters=hyperparameters,
        enable_ensemble=False # Disable ensemble to evaluate individual models only
    )
    return predictor

def generate_hourly_plots(predictor, train_data, test_data, model_groups):
    """
    Generate forecast plots for each model group.
    
    Args:
        predictor (TimeSeriesPredictor): Trained predictor
        train_data (TimeSeriesDataFrame): Training dataset
        test_data (TimeSeriesDataFrame): Test dataset
        model_groups (dict): Dictionary of model groups and their models
    """
    # Plot styling configuration
    rc_params = {
        "font.size": 10,
        "figure.figsize": [10, 6],
        "figure.dpi": 100,
    }

    # Color scheme for different model types
    colors = {'stat': 'C3', 'ml': 'C1', 'nlp': 'C2'}
    
    with plt.rc_context(rc_params):
        # Generate plots for each model group
        for group in model_groups:
            fig, axes = plt.subplots(ncols=1, nrows=2)
            fig.tight_layout()
            axes = axes.ravel()
            
            for idx, model in enumerate(model_groups[group][:2]):  # Only first two models per group
                try:
                    plot_model_forecast(
                        predictor, train_data, test_data, model, 
                        axes[idx], colors[group]
                    )
                except Exception as e:
                    print(f"Could not generate plot for {model}. Error: {e}")
                    traceback.print_exc()
            
            plt.tight_layout(pad=2, w_pad=4, h_pad=2)
            plt.show()

def generate_daily_plots(daily_real, model_groups):
    """
    Generate bar plots showing daily consumption differences for all models on one chart.
    
    Args:
        daily_real (DataFrame): DataFrame containing daily actual and predicted values
        model_groups (dict): Dictionary of model groups and their models
    """
    # Plot styling configuration
    rc_params = {
        "font.size": 10,
        "figure.figsize": [12, 6],
        "figure.dpi": 100,
    }

    # Color scheme matching hourly plots
    colors = {'stat': 'C3', 'ml': 'C1', 'nlp': 'C2'}
    
    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot bars for each model
        x = range(len(daily_real['date']))
        num_models = sum(len(models[:2]) for models in model_groups.values())
        width = 0.8 / num_models  # Width of the bars
        max_diff = max(abs(daily_real[[f'{model} daily residual' for group in model_groups 
                                     for model in model_groups[group]]]).max())
        
        bar_index = 0
        for group in model_groups:
            for model in model_groups[group]:  # Only first two models per group
                diff_col = f'{model} daily residual'
                if diff_col in daily_real.columns:
                    offset = width * (bar_index - (num_models - 1)/2)
                    bar_alpha = 0.4 if bar_index % 2 else 0.8
                    bars = ax.bar(
                        [xi + offset for xi in x],
                        daily_real[diff_col],
                        width,
                        label=model.replace("[bolt_small]", ""),
                        color=colors[group],
                        alpha=bar_alpha
                    )
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2.,
                            height+0.1 if height >= 0 else height-0.1,
                            f'{height:.2f}',
                            ha='center',
                            va='bottom' if height >= 0 else 'top',
                            rotation=0,
                            fontsize=8
                        )
                    bar_index += 1
        
        # Customize the plot
        ax.set_xlabel('Dátum')
        ax.set_ylabel('Reziduum (kWh)\nPozitív = Alulbecslés, Negatív = Túlbecslés')
        ax.set_ylim(-max_diff * 1.1, max_diff * 1.1)  # Set x-axis limits symmetrically
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        # Set x-axis labels to dates
        ax.set_xticks(x)
        daily_real['date'] = pd.to_datetime(daily_real['date'])
        ax.set_xticklabels(daily_real['date'].dt.strftime('%Y-%m-%d'))
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        plt.tight_layout()
        plt.show()

def plot_model_forecast(predictor, train_data, test_data, model, ax, color):
    """
    Plot forecast results for a single model.
    
    Args:
        predictor (TimeSeriesPredictor): Trained predictor
        train_data (TimeSeriesDataFrame): Training dataset
        test_data (TimeSeriesDataFrame): Test dataset
        model (str): Name of the model to plot
        ax (matplotlib.axes.Axes): Matplotlib axis for plotting
        color (str): Color to use for the forecast plot
    """
    # Generate predictions and prepare data
    model_predictions = predictor.predict(train_data, model=model)
    data = predictor._check_and_prepare_data_frame(test_data)
    
    point_forecast_column = "0.5"
    available_quantile_levels = [float(q) for q in model_predictions.columns if q != "mean"]
    quantile_levels = [min(available_quantile_levels), max(available_quantile_levels)]

    # Configure plot formatting and style
    formatter = matplotlib.dates.DateFormatter('%Y-%m-%d')
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_title(model.replace("[bolt_small]", ""))
    ax.grid()
    ax.set_xlabel("Time")
    ax.set_ylabel("Hourly consumption (kWh)")
    
    # Plot observed data
    ts = data.loc['household_energy'][predictor.target].iloc[-168:]
    ax.plot(ts, label="Observed", color="C0")
    
    # Plot forecast
    forecast = model_predictions.loc['household_energy']
    point_forecast = forecast[point_forecast_column]
    ax.plot(point_forecast, color=color, label="Forecast")
    
    # Plot confidence intervals
    if quantile_levels:
        for q in quantile_levels:
            ax.fill_between(forecast.index, point_forecast, forecast[str(q)], 
                          color=color, alpha=0.2)

def daily_forecast(predictor, train_data, test_data, model_groups, prediction_length):
    """
    Calculate daily consumption forecasts for all models.
    
    Args:
        predictor (TimeSeriesPredictor): Trained predictor
        train_data (TimeSeriesDataFrame): Training dataset
        test_data (TimeSeriesDataFrame): Test dataset
        model_groups (dict): Dictionary of model groups and their models
        prediction_length (int): Number of time steps to predict
        
    Returns:
        tuple: (hourly_real, daily_real) DataFrames with actual and predicted values
    """
    actuals_df = test_data.tail(prediction_length).copy().reset_index()
    hourly_real = actuals_df.copy().reset_index()
    actuals_df['date'] = pd.to_datetime(actuals_df['timestamp']).dt.date
    
    daily_real = actuals_df.groupby('date')['hourly_consumption'].sum().reset_index()
    daily_real.rename(columns={'hourly_consumption': 'daily_real_consumption'}, inplace=True)
    hourly_real.rename(columns={'hourly_consumption': 'hourly_real_consumption'}, inplace=True)
    
    for group in model_groups:
        for model in model_groups[group]:
            print(f"Calculating daily values for model {model}.")
            model_predictions = predictor.predict(train_data, model=model)
            
            hourly_real, daily_real = add_model_predictions(
                model_predictions, model, hourly_real, daily_real
            )
    
    return hourly_real, daily_real

def add_model_predictions(predictions, model_name, hourly_real, daily_real):
    """
    Add model predictions to the hourly and daily results DataFrames.
    
    Args:
        predictions (DataFrame): Model predictions
        model_name (str): Name of the model
        hourly_real (DataFrame): DataFrame with hourly actual values
        daily_real (DataFrame): DataFrame with daily actual values
        
    Returns:
        tuple: Updated (hourly_real, daily_real) DataFrames
    """
    forecast_df = predictions[['mean']].copy().reset_index()
    hourly_forecast = predictions[['mean']].copy().reset_index()
    
    forecast_df['date'] = pd.to_datetime(forecast_df['timestamp']).dt.date
    daily_forecast = forecast_df.groupby('date')['mean'].sum().reset_index()
    daily_forecast.rename(columns={'mean': f'{model_name} daily'}, inplace=True)

    daily_residual = daily_forecast[['date']].copy()
    daily_residual[f'{model_name} daily residual'] = daily_real['daily_real_consumption'] - daily_forecast[f'{model_name} daily']

    hourly_forecast.rename(columns={'mean': f'{model_name} mean'}, inplace=True)
    
    hourly_real = pd.merge(hourly_real, hourly_forecast, on=['timestamp', 'item_id'])
    daily_real = pd.merge(daily_real, daily_forecast, on='date')
    daily_real = pd.merge(daily_real, daily_residual, on='date')
    daily_mae = daily_residual[f'{model_name} daily residual'].abs().mean()

    print(f"  {model_name} daily MAE: {daily_mae:.2f} kWh")
    
    return hourly_real, daily_real

def main():
    """
    Main execution function for the energy forecasting script.
    
    Workflow:
    1. Parse command line arguments
    2. Load and prepare dataset
    3. Split data into train/test sets
    4. Load or train forecasting models
    5. Generate predictions
    6. Evaluate models
    7. Generate plots and daily forecasts if requested
    """
    args = parse_arguments()
    prediction_length = 48  # 2 days forecast
    
    print("--- Starting: Loading and Preparing Data ---")
    full_data = load_dataset(args)
    print("Data preparation complete.")
    print("-" * 40 + "\n")
    
    print("--- Starting: Splitting Data into Training and Test Sets ---")
    train_data, test_data = split_data(full_data, args, prediction_length)
    print("-" * 40 + "\n")
    
    print("--- Starting: Model Training ---")
    predictor = get_or_train_model(train_data, args, prediction_length)
    print("Model ready.")
    print("-" * 40 + "\n")
    
    print("--- Starting: Generating Forecasts ---")
    predictions = predictor.predict(train_data)
    print("Forecasts generated successfully.")
    print("Forecast details:")
    print(predictions.head())
    print("-" * 40 + "\n")
    
    print("--- Starting: Model Evaluation ---")
    leaderboard = predictor.leaderboard(
        test_data,
        extra_metrics=["MAE", "WAPE"],
        display=True
    )
    print("\nEvaluation complete. Leaderboard displayed above.")
    print("-" * 40 + "\n")
    
    model_groups = {
        'stat': ['AutoETS', 'AutoARIMA'],
        'ml': ['DeepAR', 'WaveNet'],
        'nlp': ['ChronosZeroShot[bolt_small]', 'ChronosFineTuned[bolt_small]']
    }
    
    print("--- Starting: Calculating Daily Forecasts ---")
    hourly_real, daily_real = daily_forecast(
        predictor, train_data, test_data, model_groups, prediction_length
    )
    print("\nDaily consumption summary:")
    print(daily_real.round(2).to_string(index=False))
    #print("\nHourly consumption summary:")
    #print(hourly_real.round(2).to_string(index=False))

    if args.plot:
        print("--- Starting: Generating Forecasts for Each Model ---")
        generate_hourly_plots(predictor, train_data, test_data, model_groups)
        generate_daily_plots(daily_real, model_groups)
    
    print("--- Forecasting Research Script Finished ---")

if __name__ == "__main__":
    main()
