from datasets import load_dataset

# Load BEAD dataset - 3-Aspects configuration
print('Loading BEAD dataset - 3-Aspects configuration...')
ds_3aspects = load_dataset('shainar/BEAD', '3-Aspects')
print('3-Aspects dataset loaded successfully!')
print(ds_3aspects)

# Load BEAD dataset - Full_Annotations configuration  
print('\nLoading BEAD dataset - Full_Annotations configuration...')
ds_full = load_dataset('shainar/BEAD', 'Full_Annotations')
print('Full_Annotations dataset loaded successfully!')
print(ds_full)

# Convert to pandas DataFrames
print('\nConverting to pandas DataFrames...')
df_3aspects = ds_3aspects['aspects'].to_pandas()
df_full = ds_full['full'].to_pandas()

print(f'3-Aspects DataFrame shape: {df_3aspects.shape}')
print(f'Full_Annotations DataFrame shape: {df_full.shape}')

# Save datasets to data directory
import os
os.makedirs('../data', exist_ok=True)
df_full.to_csv('../data/bead_full_dataset.csv', index=False)
df_full.to_parquet('../data/bead_full_dataset.parquet', index=False)
print('\nDatasets saved to ../data/ directory')
print('\nDatasets loaded and converted successfully!')