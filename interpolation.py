import numpy as np
import pandas as pd
from scipy import interpolate

# Note: for this interpolation raceline_awsim_30km_from_garage
# current 131 wants 300
filename = 'aichallenge/workspace/src/aichallenge_submit/simple_trajectory_generator/data/raceline_awsim_30km_from_garage.csv'

df = pd.read_csv(filename)

t_original = np.linspace(0,1, len(df))
t_new = np.linspace(0,1,300)

# Container for new data
df_interp = pd.DataFrame({'t': t_new})

# Interpolate each column
for col in df.columns:
    f = interpolate.interp1d(t_original, df[col], kind='linear')  
    df_interp[col] = f(t_new)

# Drop helper time column
df_interp = df_interp.drop(columns='t')

# Save to new CSV
df_interp.to_csv("raceline_awsim_30km_from_garage_interpolated.csv", index=False)