import pandas as pd
import xgboost as xgb
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import Meredig, TMetalFraction

df = pd.read_csv('Test.csv', skiprows=0) # Import csv with compositions & lattice types
df['predicted_debye_T'] = 0*len(df)
df['predicted_density'] = 0*len(df)

# Generate Features for machine learning models
stc=StrToComposition()
df = stc.featurize_dataframe(df,col_id='full_formula')

tmf = TMetalFraction()
df = tmf.featurize_dataframe(df=df, col_id='composition')

md = Meredig()
df = md.featurize_dataframe(df=df, col_id='composition')

df = df.drop(columns=['composition'])

lattice_types = ['cubic',
                 'hexagonal',
                 'monoclinic',
                 'orthorhombic',
                 'rhombohedral',
                 'tetragonal',
                 'triclinic']

for lat in lattice_types:
    df[f'lattice_type_{lat}'] = 0*len(df)

for i in range(len(df)):
    if df['lattice_type'].iloc[i] not in lattice_types:
        df_sample_lat = df['lattice_type'].iloc[i]
        print(f'Lattice type description not supported: "{df_sample_lat}"\n\
Supported Lattice type descriptions\n{lattice_types}')
    for j in lattice_types:
        if df['lattice_type'].loc[i] == j:
            df[f'lattice_type_{j}'].loc[i] = 1
        else:
            df[f'lattice_type_{j}'].loc[i] = 0
            
# Import Machine learning models: Density, Debye Temperature
density_booster = xgb.XGBRegressor()
density_booster.load_model('XGB_density_model_ADS.json')

debye_booster = xgb.XGBRegressor()
debye_booster.load_model('XGB_debye_T_model_ADS.json')

# Machine Learn the samples
dens_X = df.values[:,4:] 
dens_pred = density_booster.predict(dens_X)
df['predicted_density'] = dens_pred

debye_X = df.values[:,3:]
debye_T_pred = debye_booster.predict(debye_X)
df['predicted_debye_T'] = debye_T_pred

# Export to out csv
df.to_csv('Test_out.csv')