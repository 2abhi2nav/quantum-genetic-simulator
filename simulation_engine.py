import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings

class CropPerformanceSimulator:

    def __init__(self):
        self.model = None
        self.weather_data = None
        self.feature_names = None
        self._train_surrogate_model()
        self._load_weather_data()
        print("Simulator ready")

    def _train_surrogate_model(self):

        try:
            df = pd.read_csv('geno_pheno_dataset.csv')
        except FileNotFoundError:
            print("[!] ERROR: 'geno_pheno_dataset.csv' not found. Please run create_dataset.py first.")
            exit()

        gene_columns = [col for col in df.columns if 'Gene_' in col]
        phenotype_columns = ["Grain_Yield_kg_ha", "Drought_Resistance_Score", "Heat_Tolerance_Score"]

        X = df[gene_columns]
        y = df[phenotype_columns]

        self.feature_names = gene_columns

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        print("Training complete")

    def _load_weather_data(self):
 
        weather_filename = "weather_climate_2019_2020.csv" 
        try:
            self.weather_data = pd.read_csv(weather_filename)
            self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'])
            self.drought_days = self.weather_data.sort_values(by='Rainfall_mm').head(50)
            self.heat_days = self.weather_data.sort_values(by='Temperature_Max_C', ascending=False).head(50)
            print(f"Loaded weather data from {weather_filename}")
        except FileNotFoundError:
            print(f"[!] ERROR: '{weather_filename}' not found. Please run create_weather_dataset.py first.")
            exit()

    # monte carlo simulation
    def run_simulation(self, genotype_bitstring, num_scenarios=5):

        genotype_array = np.array([int(bit) for bit in genotype_bitstring]).reshape(1, -1)

        genotype_df = pd.DataFrame(genotype_array, columns=self.feature_names)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            base_yield, base_drought_res, base_heat_res = self.model.predict(genotype_df)[0]

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
            return base_yield 

        scenarios = pd.concat([sim_drought_days, sim_heat_days])

        total_adjusted_yield = 0
        actual_scenarios_run = 0

        # fitness function
        for _, weather_day in scenarios.iterrows():
            adjusted_yield = base_yield

            # heat
            if base_heat_res > 0 and weather_day['Temperature_Max_C'] > 38:
                heat_penalty = (weather_day['Temperature_Max_C'] - 38) * (1 - base_heat_res / 100) 
                adjusted_yield -= heat_penalty * 5

            # drought
            if base_drought_res > 0 and weather_day['Rainfall_mm'] < 1:
                drought_penalty = (1 - weather_day['Rainfall_mm']) * (1 - base_drought_res / 100)
                adjusted_yield -= drought_penalty * 3 

            total_adjusted_yield += max(0, adjusted_yield)
            actual_scenarios_run += 1

        if actual_scenarios_run > 0:
            final_fitness_score = total_adjusted_yield / actual_scenarios_run
        else:
            final_fitness_score = base_yield

        return final_fitness_score
