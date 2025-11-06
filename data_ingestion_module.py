"""
Data Ingestion Module for CAISO Energy Market Data
Integrates multiple data sources: gridstatus, OASIS, EIA, Open-Meteo
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# Optional: Install with: pip install gridstatus
try:
    from gridstatus import CAISO
    GRIDSTATUS_AVAILABLE = True
except ImportError:
    GRIDSTATUS_AVAILABLE = False
    print("⚠️ gridstatus not installed. Install with: pip install gridstatus")


class CAISODataIngestion:
    """
    Comprehensive data ingestion for CAISO market data
    """
    
    def __init__(self):
        if GRIDSTATUS_AVAILABLE:
            self.caiso = CAISO()
        self.cache = {}
        
    def get_load_data(self, start_date, end_date, interval='5min'):
        """
        Fetch CAISO load data
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date for data retrieval
        end_date : str or datetime  
            End date for data retrieval
        interval : str
            Data interval ('5min', '1h', 'D')
            
        Returns:
        --------
        pd.DataFrame with timestamp and load columns
        """
        if not GRIDSTATUS_AVAILABLE:
            print("Using sample data (gridstatus not available)")
            return self._generate_sample_load(start_date, end_date, interval)
        
        try:
            print(f"📥 Fetching CAISO load data from {start_date} to {end_date}...")
            
            # Get load data
            df = self.caiso.get_load(start=start_date, end=end_date)
            
            # Standardize column names
            if 'Load' in df.columns:
                df = df.rename(columns={'Load': 'load'})
            
            # Ensure timestamp column
            if df.index.name == 'Time' or 'Time' in df.columns:
                df = df.reset_index()
                df = df.rename(columns={'Time': 'timestamp'})
            
            print(f"✓ Retrieved {len(df)} load records")
            return df[['timestamp', 'load']]
            
        except Exception as e:
            print(f"⚠️ Error fetching load data: {e}")
            print("Using sample data instead...")
            return self._generate_sample_load(start_date, end_date, interval)
    
    def get_fuel_mix(self, start_date, end_date):
        """
        Fetch CAISO generation fuel mix data
        
        Returns:
        --------
        pd.DataFrame with timestamp and generation by fuel type
        """
        if not GRIDSTATUS_AVAILABLE:
            print("Using sample fuel mix (gridstatus not available)")
            return self._generate_sample_fuel_mix(start_date, end_date)
        
        try:
            print(f"📥 Fetching CAISO fuel mix data...")
            
            df = self.caiso.get_fuel_mix(start=start_date, end=end_date)
            
            # Standardize format
            if 'Time' in df.columns or df.index.name == 'Time':
                df = df.reset_index()
                df = df.rename(columns={'Time': 'timestamp'})
            
            # Common fuel type columns in CAISO
            fuel_cols = ['Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 
                        'Small hydro', 'Coal', 'Nuclear', 'Natural Gas', 
                        'Large Hydro', 'Batteries', 'Imports', 'Other']
            
            # Standardize column names to lowercase with underscores
            rename_dict = {col: col.lower().replace(' ', '_') for col in fuel_cols if col in df.columns}
            df = df.rename(columns=rename_dict)
            
            print(f"✓ Retrieved {len(df)} fuel mix records")
            return df
            
        except Exception as e:
            print(f"⚠️ Error fetching fuel mix: {e}")
            return self._generate_sample_fuel_mix(start_date, end_date)
    
    def get_lmp_prices(self, start_date, end_date, node='TH_NP15_GEN-APND'):
        """
        Fetch Locational Marginal Prices (LMP) data
        
        Parameters:
        -----------
        node : str
            Trading hub or node (default: TH_NP15_GEN-APND for NP15 hub)
            Common nodes: TH_SP15_GEN-APND (SP15), TH_ZP26_GEN-APND (ZP26)
        """
        if not GRIDSTATUS_AVAILABLE:
            print("Using sample price data (gridstatus not available)")
            return self._generate_sample_prices(start_date, end_date)
        
        try:
            print(f"📥 Fetching LMP prices for {node}...")
            
            df = self.caiso.get_lmp(
                start=start_date,
                end=end_date,
                market='DAY_AHEAD_HOURLY',
                locations=[node]
            )
            
            if 'Time' in df.columns or df.index.name == 'Time':
                df = df.reset_index()
                df = df.rename(columns={'Time': 'timestamp'})
            
            # Rename LMP column
            if 'LMP' in df.columns:
                df = df.rename(columns={'LMP': 'price'})
            
            print(f"✓ Retrieved {len(df)} price records")
            return df[['timestamp', 'price']]
            
        except Exception as e:
            print(f"⚠️ Error fetching prices: {e}")
            return self._generate_sample_prices(start_date, end_date)
    
    def get_weather_data(self, start_date, end_date, latitude=38.5816, longitude=-121.4944):
        """
        Fetch weather data from Open-Meteo API
        Default location: Sacramento, CA (CAISO headquarters)
        
        Parameters:
        -----------
        latitude : float
            Latitude coordinate
        longitude : float
            Longitude coordinate
        """
        try:
            print(f"📥 Fetching weather data for location ({latitude}, {longitude})...")
            
            # Format dates for API
            start = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            
            # Open-Meteo API endpoint
            url = "https://archive-api.open-meteo.com/v1/archive"
            
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start,
                "end_date": end,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation",
                "timezone": "America/Los_Angeles"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse weather data
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['hourly']['time']),
                'temperature': data['hourly']['temperature_2m'],
                'humidity': data['hourly']['relative_humidity_2m'],
                'wind_speed': data['hourly']['wind_speed_10m'],
                'solar_radiation': data['hourly']['shortwave_radiation']
            })
            
            print(f"✓ Retrieved {len(df)} weather records")
            return df
            
        except Exception as e:
            print(f"⚠️ Error fetching weather data: {e}")
            return self._generate_sample_weather(start_date, end_date)
    
    def get_renewable_forecast(self, start_date, end_date):
        """
        Fetch renewable generation forecasts
        """
        if not GRIDSTATUS_AVAILABLE:
            print("Using sample renewable forecast")
            return self._generate_sample_renewables(start_date, end_date)
        
        try:
            print(f"📥 Fetching renewable forecasts...")
            
            # Get solar and wind forecasts if available
            df_solar = self.caiso.get_solar_forecast(start=start_date, end=end_date)
            df_wind = self.caiso.get_wind_forecast(start=start_date, end=end_date)
            
            # Merge forecasts
            df = pd.merge(df_solar, df_wind, on='timestamp', how='outer')
            
            print(f"✓ Retrieved renewable forecasts")
            return df
            
        except Exception as e:
            print(f"⚠️ Error fetching renewable forecasts: {e}")
            return self._generate_sample_renewables(start_date, end_date)
    
    def create_integrated_dataset(self, start_date, end_date, include_weather=True):
        """
        Create a comprehensive integrated dataset with all data sources
        
        Returns:
        --------
        pd.DataFrame with all relevant features for ML modeling
        """
        print("\n" + "="*60)
        print("🔄 Creating Integrated Dataset")
        print("="*60 + "\n")
        
        # 1. Load data (base dataset)
        df_load = self.get_load_data(start_date, end_date)
        
        # 2. Fuel mix / Generation data
        df_fuel = self.get_fuel_mix(start_date, end_date)
        
        # 3. Price data
        df_price = self.get_lmp_prices(start_date, end_date)
        
        # 4. Weather data (if requested)
        if include_weather:
            df_weather = self.get_weather_data(start_date, end_date)
        
        # Merge all datasets
        print("\n🔗 Merging datasets...")
        
        # Start with load data
        df = df_load.copy()
        
        # Merge fuel mix
        if not df_fuel.empty:
            df = pd.merge(df, df_fuel, on='timestamp', how='left')
        
        # Merge prices
        if not df_price.empty:
            # Interpolate prices to match load frequency if needed
            df = pd.merge(df, df_price, on='timestamp', how='left')
            df['price'] = df['price'].interpolate(method='linear')
        
        # Merge weather
        if include_weather and not df_weather.empty:
            df = pd.merge(df, df_weather, on='timestamp', how='left')
            # Forward fill weather data (typically hourly to 5-min intervals)
            weather_cols = ['temperature', 'humidity', 'wind_speed', 'solar_radiation']
            for col in weather_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill')
        
        # Calculate net load (if we have solar and wind)
        if 'solar' in df.columns and 'wind' in df.columns:
            df['net_load'] = df['load'] - df['solar'] - df['wind']
        
        # Add calendar features
        df = self._add_calendar_features(df)
        
        print(f"\n✓ Integrated dataset created:")
        print(f"   - Records: {len(df)}")
        print(f"   - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   - Features: {len(df.columns)}")
        print(f"   - Missing data: {df.isnull().sum().sum()} values")
        
        print("\n" + "="*60)
        
        return df
    
    def _add_calendar_features(self, df):
        """Add holiday and calendar-specific features"""
        df = df.copy()
        
        # US Federal Holidays (major ones affecting energy demand)
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df['timestamp'].min(), end=df['timestamp'].max())
        
        df['is_holiday'] = df['timestamp'].dt.date.isin(holidays.date).astype(int)
        
        # Day after holiday (often affects demand)
        df['is_day_after_holiday'] = df['timestamp'].dt.date.isin(
            (holidays + timedelta(days=1)).date
        ).astype(int)
        
        return df
    
    # ========================================================================
    # Sample Data Generation (for testing without API access)
    # ========================================================================
    
    def _generate_sample_load(self, start_date, end_date, interval='5min'):
        """Generate realistic sample load data"""
        dates = pd.date_range(start=start_date, end=end_date, freq=interval)
        
        hour = dates.hour + dates.minute/60
        day_of_week = dates.dayofweek
        
        # Base load with duck curve characteristics
        base_load = 25000 + 8000 * np.sin((hour - 6) * np.pi / 12)
        solar_dip = -5000 * np.maximum(0, np.sin((hour - 6) * np.pi / 12)) * (hour > 8) * (hour < 18)
        evening_peak = 3000 * np.exp(-((hour - 19)**2) / 2)
        
        # Weekend effect
        weekend_factor = np.where(day_of_week >= 5, 0.85, 1.0)
        
        # Seasonal effect
        day_of_year = dates.dayofyear
        seasonal = 2000 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Noise
        noise = np.random.normal(0, 500, len(dates))
        
        load = (base_load + solar_dip + evening_peak + seasonal) * weekend_factor + noise
        
        return pd.DataFrame({'timestamp': dates, 'load': load})
    
    def _generate_sample_fuel_mix(self, start_date, end_date):
        """Generate sample fuel mix data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='5min')
        hour = dates.hour + dates.minute/60
        
        # Solar (peaks at midday)
        solar = np.maximum(0, 8000 * np.sin((hour - 6) * np.pi / 12) * (hour > 6) * (hour < 19))
        
        # Wind (more random)
        wind = 3000 + 2000 * np.random.randn(len(dates)).cumsum() / 100
        wind = np.clip(wind, 500, 6000)
        
        # Hydro (seasonal, relatively stable)
        hydro = 5000 + 1000 * np.sin(2 * np.pi * dates.dayofyear / 365)
        
        # Nuclear (baseload)
        nuclear = np.ones(len(dates)) * 2000
        
        # Natural gas (fills the gap)
        load = 25000 + 8000 * np.sin((hour - 6) * np.pi / 12)
        natural_gas = load - solar - wind - hydro - nuclear
        natural_gas = np.maximum(natural_gas, 0)
        
        return pd.DataFrame({
            'timestamp': dates,
            'solar': solar,
            'wind': wind,
            'hydro': hydro,
            'nuclear': nuclear,
            'natural_gas': natural_gas
        })
    
    def _generate_sample_prices(self, start_date, end_date):
        """Generate sample price data correlated with load"""
        dates = pd.date_range(start=start_date, end=end_date, freq='5min')
        hour = dates.hour + dates.minute/60
        
        # Base price with demand correlation
        base_price = 30 + 15 * np.sin((hour - 6) * np.pi / 12)
        
        # Peak pricing
        peak_price = 20 * ((hour >= 17) & (hour <= 21))
        
        # Weekend discount
        weekend_discount = -5 * (dates.dayofweek >= 5)
        
        # Volatility
        noise = 5 * np.random.randn(len(dates))
        
        price = base_price + peak_price + weekend_discount + noise
        price = np.clip(price, 10, 200)
        
        return pd.DataFrame({'timestamp': dates, 'price': price})
    
    def _generate_sample_weather(self, start_date, end_date):
        """Generate sample weather data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='1h')
        hour = dates.hour
        day_of_year = dates.dayofyear
        
        # Temperature (seasonal + daily cycle)
        base_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal
        daily_temp = 5 * np.sin(2 * np.pi * (hour - 6) / 24)  # Daily cycle
        temperature = base_temp + daily_temp + np.random.normal(0, 2, len(dates))
        
        # Humidity (inverse of temperature generally)
        humidity = 70 - 0.5 * temperature + np.random.normal(0, 10, len(dates))
        humidity = np.clip(humidity, 20, 100)
        
        # Wind speed
        wind_speed = 5 + 3 * np.random.randn(len(dates))
        wind_speed = np.clip(wind_speed, 0, 20)
        
        # Solar radiation (peaks at noon)
        solar_radiation = np.maximum(0, 800 * np.sin((hour - 6) * np.pi / 12))
        
        return pd.DataFrame({
            'timestamp': dates,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'solar_radiation': solar_radiation
        })
    
    def _generate_sample_renewables(self, start_date, end_date):
        """Generate sample renewable forecasts"""
        dates = pd.date_range(start=start_date, end=end_date, freq='1h')
        hour = dates.hour
        
        solar_forecast = np.maximum(0, 8000 * np.sin((hour - 6) * np.pi / 12))
        wind_forecast = 3000 + 1000 * np.random.randn(len(dates))
        wind_forecast = np.clip(wind_forecast, 500, 6000)
        
        return pd.DataFrame({
            'timestamp': dates,
            'solar_forecast': solar_forecast,
            'wind_forecast': wind_forecast
        })


# ============================================================================
# Usage Example
# ============================================================================

def main():
    """
    Example usage of the CAISO Data Ingestion system
    """
    print("\n" + "="*60)
    print("CAISO Data Ingestion System")
    print("="*60 + "\n")
    
    # Initialize ingestion system
    ingestion = CAISODataIngestion()
    
    # Define date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Create integrated dataset
    df = ingestion.create_integrated_dataset(
        start_date=start_date,
        end_date=end_date,
        include_weather=True
    )
    
    # Display sample
    print("\n📊 Dataset Preview:")
    print(df.head(10))
    
    print("\n📈 Dataset Statistics:")
    print(df.describe())
    
    # Save to file
    output_file = 'caiso_integrated_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Data saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    df = main()
