import os
# See src/predict.py for the full explanation: skips a joblib/loky Windows subprocess
# probe that can print a non-fatal but console-alarming traceback on restricted systems.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import argparse
import sys
from pathlib import Path
import pandas as pd
import pickle

from src.validation import ValidationEngine
from src.features import FeatureEngineer
from src.utils import get_logger

logger = get_logger("CLI_GenerateFeatures")

def main():
    parser = argparse.ArgumentParser(description="ForecastIQ Feature Generation Pipeline")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory containing input channel CSV files.")
    parser.add_argument("--out", type=str, default="features.pkl", help="Output Parquet/Pickle/CSV feature file path.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out).resolve()

    logger.info(f"Starting End-to-End Feature Generation Pipeline dynamic read from {data_dir}...")

    if not data_dir.exists():
        logger.error(f"Provided DATA_DIR '{data_dir}' does not exist.")
        sys.exit(1)

    # 1. Validation & Ingestion Engine
    val_engine = ValidationEngine(data_dir=data_dir)
    try:
        unified_df, validation_summary = val_engine.run_full_ingestion()
        logger.info(f"Unified cross-channel dataset constructed: {len(unified_df)} records. Data Quality Score: {validation_summary['data_quality_score']}/100.")
    except Exception as e:
        logger.error(f"Fatal error during dataset validation & ingestion: {str(e)}")
        sys.exit(1)

    # 2. Feature Engineering Suite
    feat_engine = FeatureEngineer()
    all_features = feat_engine.generate_all_features(unified_df)

    # BUG 1 fix (CRITICAL): the engineered feature set (rolling windows, ROAS, CPC, CTR,
    # conversion rate, volatility, trend indicators) was computed above and then silently
    # discarded — the code below was saving raw `unified_df` (10 raw columns) instead.
    # Confirmed via binary inspection: features.pkl previously had zero engineered columns.
    #
    # Fix: merge the daily-level engineered columns from FeatureEngineer back onto the
    # row-level unified_df (joined on `date`), so the saved artifact keeps BOTH:
    #   - per-row granularity (channel, campaign_id, campaign_name, campaign_type) needed
    #     by downstream dimension-level model training (models.py train_dimension_models)
    #   - the actual engineered daily features needed by the overall ensemble
    daily_engineered = all_features["daily"]
    raw_cols = {'date', 'spend', 'revenue', 'clicks', 'impressions', 'conversions'}
    engineered_cols = [c for c in daily_engineered.columns if c not in raw_cols]
    enriched_df = unified_df.merge(
        daily_engineered[['date'] + engineered_cols], on='date', how='left'
    )
    logger.info(
        f"Engineered {len(engineered_cols)} feature columns "
        f"(time, marketing ratios, rolling windows, volatility, trend) and merged "
        f"onto {len(unified_df)} raw rows. Total columns: {len(enriched_df.columns)}."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Robust multi-format saving to ensure 100% reliability regardless of PyArrow/Fastparquet dependencies
    try:
        if out_path.suffix == ".parquet":
            try:
                enriched_df.to_parquet(out_path, index=False)
            except ImportError:
                logger.warning("PyArrow/Fastparquet not installed. Falling back to universal Pickle format.")
                out_path = out_path.with_suffix(".pkl")
                enriched_df.to_pickle(out_path)
        elif out_path.suffix == ".csv":
            enriched_df.to_csv(out_path, index=False)
        else:
            enriched_df.to_pickle(out_path)
        logger.info(f"Feature artifact successfully written to {out_path}")
    except Exception as e:
        logger.error(f"Error saving feature file: {str(e)}. Forcing universal backup saving.")
        backup_path = out_path.parent / "features_backup.pkl"
        enriched_df.to_pickle(backup_path)

if __name__ == "__main__":
    main()
