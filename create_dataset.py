import pandas as pd
import numpy as np
import os

# --- Simulation Parameters ---
NUM_VARIETIES = 500 # Increased from 200 to 500
NUM_GENES = 20

# --- Define the "Biological Rules" for the Simulation ---
YIELD_GENES = [1, 4, 8, 11, 15]
DROUGHT_GENES = [2, 6, 9, 13, 17]
HEAT_GENES = [0, 5, 10, 16]
SYNERGY_GENES_DROUGHT = [6, 13]
SYNERGY_GENES_YIELD = [4, 15]
DETRIMENTAL_GENE = 19

def generate_phenotypes(genotype):
    """
    Generates phenotype scores based on a given genotype using our rules.
    """
    # --- Calculate Grain Yield ---
    yield_score = 50.0
    yield_score += np.sum(genotype[YIELD_GENES]) * 5.5
    if all(genotype[i] == 1 for i in SYNERGY_GENES_YIELD):
        yield_score += 40.0
    if genotype[DETRIMENTAL_GENE] == 1:
        yield_score -= 30.0
        
    # --- Calculate Drought Resistance ---
    drought_score = 30.0
    drought_score += np.sum(genotype[DROUGHT_GENES]) * 8.0
    if all(genotype[i] == 1 for i in SYNERGY_GENES_DROUGHT):
        drought_score += 50.0

    # --- Calculate Heat Tolerance ---
    heat_score = 40.0
    heat_score += np.sum(genotype[HEAT_GENES]) * 7.0
    
    # Add random noise for realism
    yield_score *= np.random.uniform(0.95, 1.05)
    drought_score *= np.random.uniform(0.95, 1.05)
    heat_score *= np.random.uniform(0.95, 1.05)
    
    return max(0, yield_score), max(0, drought_score), max(0, heat_score)

# --- Main Script to Generate the Dataset ---
if __name__ == "__main__":
    print("Generating expanded synthetic genotype-phenotype dataset...")
    
    gene_columns = [f"Gene_{i}" for i in range(NUM_GENES)]
    phenotype_columns = ["Grain_Yield_kg_ha", "Drought_Resistance_Score", "Heat_Tolerance_Score"]
    
    genotypes = np.random.randint(0, 2, size=(NUM_VARIETIES, NUM_GENES))
    
    data = []
    for i in range(NUM_VARIETIES):
        variety_id = f"Variety_{i+1:03d}"
        genotype = genotypes[i]
        yield_val, drought_val, heat_val = generate_phenotypes(genotype)
        row = [variety_id] + list(genotype) + [yield_val, drought_val, heat_val]
        data.append(row)
        
    df = pd.DataFrame(data, columns=["Variety_ID"] + gene_columns + phenotype_columns)
    
    for col in phenotype_columns:
        df[col] = df[col].round(2)
        
    file_path = "geno_pheno_dataset.csv"
    df.to_csv(file_path, index=False)
    
    print(f"\nDataset successfully created with {NUM_VARIETIES} varieties.")
    print(f"File saved as '{file_path}' in the directory '{os.getcwd()}'")
    print("\n--- Dataset Preview ---")
    print(df.head())
    print("...")
    print(df.tail())
