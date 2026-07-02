import pandas as pd
import numpy as np
from typing import Dict, List, Any
from src.utils import get_logger

logger = get_logger("FeatureEngineering")

class FeatureEngineer:
    """
    Elite Feature Engineering Suite for ForecastIQ.
    Constructs rich Time, Performance, Marketing, Channel, and Volatility features.
    """
    def __init__(self, rolling_windows: List[int] = None):
        if rolling_windows is None:
            rolling_windows = [7, 14, 30]
        self.rolling_windows = rolling_windows

    def _get_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return 'WINTER'
        elif month in [3, 4, 5]:
            return 'SPRING'
        elif month in [6, 7, 8]:
            return 'SUMMER'
        else:
            return 'FALL'

    def create_daily_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs advanced multi-channel daily aggregate features."""
        logger.info("Constructing Daily Aggregate Features...")
        
        # Aggregate daily
        daily = df.groupby('date').agg({
            'spend': 'sum',
            'revenue': 'sum',
            'clicks': 'sum',
            'impressions': 'sum',
            'conversions': 'sum'
        }).reset_index()

        # Sort strictly by date
        daily = daily.sort_values('date').reset_index(drop=True)

        # 1. Time Features
        daily['month'] = daily['date'].dt.month
        daily['quarter'] = daily['date'].dt.quarter
        daily['week'] = daily['date'].dt.isocalendar().week.astype(int)
        daily['day_of_week'] = daily['date'].dt.dayofweek
        daily['is_weekend'] = daily['day_of_week'].isin([5, 6]).astype(int)
        daily['season'] = daily['month'].apply(self._get_season)
        # Encode season as categorical/numeric
        season_map = {'WINTER': 0, 'SPRING': 1, 'SUMMER': 2, 'FALL': 3}
        daily['season_encoded'] = daily['season'].map(season_map)

        # 2. Marketing Features
        daily['cpc'] = daily['spend'] / (daily['clicks'] + 1e-5)
        daily['cpa'] = daily['spend'] / (daily['conversions'] + 1e-5)
        daily['ctr'] = daily['clicks'] / (daily['impressions'] + 1e-5)
        daily['conversion_rate'] = daily['conversions'] / (daily['clicks'] + 1e-5)
        daily['roas'] = daily['revenue'] / (daily['spend'] + 1e-5)

        # Clip outrageous ratios
        daily['cpc'] = daily['cpc'].clip(lower=0, upper=100)
        daily['cpa'] = daily['cpa'].clip(lower=0, upper=1000)
        daily['ctr'] = daily['ctr'].clip(lower=0, upper=1.0)
        daily['conversion_rate'] = daily['conversion_rate'].clip(lower=0, upper=1.0)

        # 3. Performance & Volatility Features (Rolling Windows)
        for w in self.rolling_windows:
            daily[f'rolling_revenue_{w}d'] = daily['revenue'].shift(1).rolling(w, min_periods=1).mean().fillna(daily['revenue'].mean())
            daily[f'rolling_spend_{w}d'] = daily['spend'].shift(1).rolling(w, min_periods=1).mean().fillna(daily['spend'].mean())
            daily[f'rolling_roas_{w}d'] = daily['roas'].shift(1).rolling(w, min_periods=1).mean().fillna(daily['roas'].mean())
            daily[f'rolling_conversion_rate_{w}d'] = daily['conversion_rate'].shift(1).rolling(w, min_periods=1).mean().fillna(daily['conversion_rate'].mean())
            
            # Volatility
            daily[f'moving_std_revenue_{w}d'] = daily['revenue'].shift(1).rolling(w, min_periods=1).std().fillna(0.0)
            daily[f'moving_std_spend_{w}d'] = daily['spend'].shift(1).rolling(w, min_periods=1).std().fillna(0.0)
            daily[f'moving_std_roas_{w}d'] = daily['roas'].shift(1).rolling(w, min_periods=1).std().fillna(0.0)
            daily[f'variance_revenue_{w}d'] = daily[f'moving_std_revenue_{w}d'] ** 2

            # Trend indicators (percentage change over rolling window)
            daily[f'trend_revenue_{w}d'] = (daily['revenue'].shift(1) - daily[f'rolling_revenue_{w}d']) / (daily[f'rolling_revenue_{w}d'] + 1e-5)

        return daily

    def create_channel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs advanced channel-level breakdown features with spend/revenue shares."""
        logger.info("Constructing Channel Breakdowns & Share Features...")
        
        # Calculate daily aggregate spend and revenue first
        daily_totals = df.groupby('date').agg(
            total_daily_spend=('spend', 'sum'),
            total_daily_revenue=('revenue', 'sum')
        ).reset_index()

        channel_daily = df.groupby(['date', 'channel']).agg({
            'spend': 'sum',
            'revenue': 'sum',
            'clicks': 'sum',
            'impressions': 'sum',
            'conversions': 'sum'
        }).reset_index()

        # Merge daily totals to compute shares
        merged = pd.merge(channel_daily, daily_totals, on='date', how='left')
        merged['spend_share'] = merged['spend'] / (merged['total_daily_spend'] + 1e-5)
        merged['revenue_share'] = merged['revenue'] / (merged['total_daily_revenue'] + 1e-5)
        merged['roas'] = merged['revenue'] / (merged['spend'] + 1e-5)
        merged['cpc'] = merged['spend'] / (merged['clicks'] + 1e-5)
        merged['ctr'] = merged['clicks'] / (merged['impressions'] + 1e-5)

        # Fill NaNs
        merged.fillna(0.0, inplace=True)
        return merged

    def create_campaign_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs campaign-level performance and contribution features."""
        logger.info("Constructing Campaign Breakdowns & Features...")
        
        camp = df.copy()
        camp['roas'] = camp['revenue'] / (camp['spend'] + 1e-5)
        camp['cpc'] = camp['spend'] / (camp['clicks'] + 1e-5)
        camp['ctr'] = camp['clicks'] / (camp['impressions'] + 1e-5)
        camp['cpa'] = camp['spend'] / (camp['conversions'] + 1e-5)
        return camp.fillna(0.0)

    def generate_all_features(self, unified_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Runs the entire feature engineering pipeline and returns all feature entities."""
        daily_features = self.create_daily_aggregate_features(unified_df)
        channel_features = self.create_channel_features(unified_df)
        campaign_features = self.create_campaign_features(unified_df)

        logger.info("Elite Feature Engineering pipeline successfully executed.")
        return {
            "daily": daily_features,
            "channel": channel_features,
            "campaign": campaign_features
        }
