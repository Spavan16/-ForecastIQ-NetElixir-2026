import argparse
import sys
from pathlib import Path
import pandas as pd

from src.models import EnsembleForecaster
from src.utils import get_logger

logger = get_logger("CLI_PredictEngine")

def main():
    parser = argparse.ArgumentParser(description="ForecastIQ Elite Multi-Horizon Forecasting CLI")
    parser.add_argument("--features", type=str, default="features.pkl", help="Input feature artifact file (Parquet/Pickle/CSV).")
    parser.add_argument("--model", type=str, default="./pickle/model.pkl", help="Path to pickled ensemble model.")
    parser.add_argument("--output", type=str, default="./output/predictions.csv", help="Where to write predictions.csv.")
    args = parser.parse_args()

    feat_path = Path(args.features).resolve()
    model_path = Path(args.model).resolve()
    out_path = Path(args.output).resolve()

    logger.info(f"Starting Multi-Horizon Probabilistic Prediction run...")

    # If user provided parquet but it fell back to .pkl, detect automatically
    if not feat_path.exists() and feat_path.with_suffix(".pkl").exists():
        feat_path = feat_path.with_suffix(".pkl")

    if not feat_path.exists():
        logger.error(f"Features file '{feat_path}' not found. Run generate_features.py first.")
        sys.exit(1)

    # Load recent features to determine exact start_date
    try:
        if feat_path.suffix == ".parquet":
            try:
                df = pd.read_parquet(feat_path)
            except ImportError:
                df = pd.read_pickle(feat_path.with_suffix(".pkl"))
        elif feat_path.suffix == ".csv":
            df = pd.read_csv(feat_path)
        else:
            df = pd.read_pickle(feat_path)

        max_date = pd.to_datetime(df['date']).max()
        start_date = max_date + pd.Timedelta(days=1)
        logger.info(f"Loaded test dataset features. Evaluated forecast start date: {start_date.strftime('%Y-%m-%d')}.")
    except Exception as e:
        logger.error(f"Failed to parse features file '{feat_path}': {str(e)}. Cannot determine start date.")
        sys.exit(1)

    # Load Model Artifact
    forecaster = EnsembleForecaster(model_path=model_path)
    if not forecaster.load_models():
        logger.warning(f"Pickled model missing or invalid at '{model_path}'. Training instant backup ensemble on test data.")
        try:
            daily = df.groupby('date').agg({'revenue': 'sum', 'spend': 'sum'}).reset_index()
            daily['month'] = daily['date'].dt.month
            daily['quarter'] = daily['date'].dt.quarter
            daily['week'] = daily['date'].dt.isocalendar().week.astype(int)
            daily['day_of_week'] = daily['date'].dt.dayofweek
            daily['is_weekend'] = daily['day_of_week'].isin([5, 6]).astype(int)
            daily['season_encoded'] = daily['month'].apply(lambda m: 0 if m in [12,1,2] else (1 if m in [3,4,5] else (2 if m in [6,7,8] else 3)))
            forecaster.train_overall_models(daily)
            forecaster.train_dimension_models(df)
            forecaster.save_models()
        except Exception as e:
            logger.error(f"Fatal exception during instant backup training: {str(e)}")
            sys.exit(1)

    # Produce COMPLETE Master Output CSV
    try:
        master_preds_df = forecaster.produce_full_predictions_table(start_date)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        master_preds_df.to_csv(out_path, index=False)
        logger.info(f"Pristine multi-dimensional probabilistic predictions successfully exported to {out_path}")
    except Exception as e:
        logger.error(f"Fatal error generating master predictions CSV: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
