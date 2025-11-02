import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings

class CropPerformanceSimulator:
    """
    This class loads all datasets and trains a model to simulate crop performance.
    It acts as the fitness function for the genetic algorithms.
    """
    def __init__(self):
        print("Initializing Crop Performance Simulator...")
        self.model = None
        self.weather_data = None
        self.feature_names = None # Store feature names
        self._train_surrogate_model()
        self._load_weather_data()
        print("Simulator ready.")

    def _train_surrogate_model(self):
        """
        Loads the geno-pheno data and trains a model to predict traits from genes.
        """
        try:
            df = pd.read_csv('geno_pheno_dataset.csv')
        except FileNotFoundError:
            print("[!] ERROR: 'geno_pheno_dataset.csv' not found. Please run create_dataset.py first.")
            exit()

        gene_columns = [col for col in df.columns if 'Gene_' in col]
        phenotype_columns = ["Grain_Yield_kg_ha", "Drought_Resistance_Score", "Heat_Tolerance_Score"]

        X = df[gene_columns]
        y = df[phenotype_columns]

        # Store feature names after training
        self.feature_names = gene_columns

        # Train a RandomForest model to learn the gene -> phenotype relationship
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        print("Trained surrogate model on genotype-phenotype data.")

    def _load_weather_data(self):
        """
        Loads the weather data for use in simulations.
        Adjust the filename based on the generator script (e.g., using 2019-2020 data).
        """
        weather_filename = "weather_climate_2019_2020.csv" # Check if this matches your generated file
        try:
            self.weather_data = pd.read_csv(weather_filename)
            # Ensure Date column is datetime
            self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'])
            # Identify stressful days for more effective testing
            self.drought_days = self.weather_data.sort_values(by='Rainfall_mm').head(50)
            self.heat_days = self.weather_data.sort_values(by='Temperature_Max_C', ascending=False).head(50)
            print(f"Loaded weather data from {weather_filename} and identified stressful climate scenarios.")
        except FileNotFoundError:
            print(f"[!] ERROR: '{weather_filename}' not found. Please run create_weather_dataset.py first.")
            exit()

    def run_simulation(self, genotype_bitstring, num_scenarios=5):
        """
        This is the core Monte Carlo simulation.
        It predicts a genotype's performance across multiple random weather scenarios.
        """
        # Convert bitstring to numpy array first
        genotype_array = np.array([int(bit) for bit in genotype_bitstring]).reshape(1, -1)

        # Create a pandas DataFrame with the genotype data AND the correct column names
        # Use the stored feature names
        genotype_df = pd.DataFrame(genotype_array, columns=self.feature_names)

        # 1. Predict base traits from the trained model using the DataFrame
        # Suppress the feature names warning during prediction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            base_yield, base_drought_res, base_heat_res = self.model.predict(genotype_df)[0]

        # 2. Select random weather scenarios (Monte Carlo part)
        # Ensure enough scenarios exist if num_scenarios is large
        num_drought = min(num_scenarios // 2, len(self.drought_days))
        num_heat = min(num_scenarios - num_drought, len(self.heat_days))

        if num_drought > 0:
             sim_drought_days = self.drought_days.sample(num_drought)
        else:
             sim_drought_days = pd.DataFrame()

        if num_heat > 0:
             sim_heat_days = self.heat_days.sample(num_heat)
        else:
             sim_heat_days = pd.DataFrame()

        if sim_drought_days.empty and sim_heat_days.empty:
            # Handle case where no stressful days could be sampled (unlikely with 50)
            return base_yield # Return base yield if no scenarios run

        scenarios = pd.concat([sim_drought_days, sim_heat_days])

        total_adjusted_yield = 0
        actual_scenarios_run = 0

        # 3. Apply stress effects based on weather
        for _, weather_day in scenarios.iterrows():
            adjusted_yield = base_yield

            # Apply heat stress (only if heat tolerance is positive)
            if base_heat_res > 0 and weather_day['Temperature_Max_C'] > 38:
                heat_penalty = (weather_day['Temperature_Max_C'] - 38) * (1 - base_heat_res / 100) # Percentage reduction
                adjusted_yield -= heat_penalty * 5 # Scale the penalty magnitude

            # Apply drought stress (only if drought resistance is positive)
            if base_drought_res > 0 and weather_day['Rainfall_mm'] < 1:
                drought_penalty = (1 - weather_day['Rainfall_mm']) * (1 - base_drought_res / 100) # Percentage reduction
                adjusted_yield -= drought_penalty * 3 # Scale the penalty magnitude

            total_adjusted_yield += max(0, adjusted_yield)
            actual_scenarios_run += 1

        # The final fitness is the average performance across all scenarios run
        if actual_scenarios_run > 0:
            final_fitness_score = total_adjusted_yield / actual_scenarios_run
        else:
            final_fitness_score = base_yield # Default to base if no scenarios ran

        return final_fitness_score
