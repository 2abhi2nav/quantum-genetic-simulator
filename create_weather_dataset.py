import pandas as pd
import numpy as np
import os

# --- Simulation Parameters ---
START_YEAR = 2019
END_YEAR = 2020 # Extended from 2019 to 2020
LOCATION = "Wardha, India"

def generate_weather_data():
    """
    Generates multiple years of realistic, daily weather data.
    """
    # Create a date range for the specified years
    dates = pd.date_range(start=f'{START_YEAR}-01-01', end=f'{END_YEAR}-12-31', freq='D')
    num_days = len(dates)
    
    # --- FIX IS HERE: Convert pandas Series to a mutable NumPy array ---
    day_of_year = dates.dayofyear.to_numpy()
    
    # --- Simulate Temperature (°C) ---
    temp_avg = 28 + 12 * np.sin(2 * np.pi * (day_of_year - 90) / 365.25)
    temp_noise = np.random.normal(0, 1.5, num_days)
    temp_max = temp_avg + np.random.uniform(3, 5, num_days) + temp_noise
    temp_min = temp_avg - np.random.uniform(3, 5, num_days) + temp_noise
    
    # --- Simulate Rainfall (mm) ---
    rainfall = np.zeros(num_days)
    # Model a monsoon season (June-Sept)
    monsoon_indices = np.where((dates.month >= 6) & (dates.month <= 9))[0]
    non_monsoon_indices = np.where((dates.month < 6) | (dates.month > 9))[0]
    
    # Higher chance of rain during monsoon
    rainfall[monsoon_indices] = np.random.exponential(15, len(monsoon_indices)) * (np.random.rand(len(monsoon_indices)) < 0.45)
    # Occasional non-monsoon rain
    rainfall[non_monsoon_indices] = np.random.exponential(5, len(non_monsoon_indices)) * (np.random.rand(len(non_monsoon_indices)) < 0.03)

    # --- Simulate Humidity (%) ---
    humidity_base = 65 - 25 * np.sin(2 * np.pi * (day_of_year - 60) / 365.25)
    humidity_noise = np.random.normal(0, 5, num_days)
    humidity = humidity_base + humidity_noise
    humidity[rainfall > 0] += np.random.uniform(10, 20, np.sum(rainfall > 0))
    humidity = np.clip(humidity, 20, 99)

    # --- Simulate Solar Radiation (MJ/m²) ---
    radiation_base = 20 + 8 * np.sin(2 * np.pi * (day_of_year - 120) / 365.25)
    radiation_noise = np.random.normal(0, 1, num_days)
    solar_radiation = radiation_base + radiation_noise
    solar_radiation[rainfall > 5] *= np.random.uniform(0.3, 0.5, np.sum(rainfall > 5))
    solar_radiation = np.clip(solar_radiation, 5, 30)

    # --- Create DataFrame ---
    df = pd.DataFrame({
        'Date': dates,
        'Location': LOCATION,
        'Temperature_Max_C': np.round(temp_max, 1),
        'Temperature_Min_C': np.round(temp_min, 1),
        'Rainfall_mm': np.round(rainfall, 1),
        'Humidity_Percent': np.round(humidity, 1),
        'Solar_Radiation_MJ_m2': np.round(solar_radiation, 2)
    })
    
    return df

# --- Main Script to Generate the Dataset ---
if __name__ == "__main__":
    print(f"Generating expanded synthetic weather dataset for '{LOCATION}' for {START_YEAR}-{END_YEAR}...")
    
    weather_df = generate_weather_data()
    
    file_path = f"weather_climate_{START_YEAR}_{END_YEAR}.csv"
    weather_df.to_csv(file_path, index=False)
    
    print(f"\nDataset successfully created with {len(weather_df)} rows.")
    print(f"File saved as '{file_path}' in the directory '{os.getcwd()}'")
    
    print("\n--- Dataset Preview ---")
    print(weather_df.head())
    print("...")
    print(weather_df.tail())
