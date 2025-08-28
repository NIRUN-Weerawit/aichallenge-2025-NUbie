import numpy as np
import pandas as pd
from scipy import interpolate

# Note: for this interpolation raceline_awsim_30km_from_garage
# current 131 wants 300
filename = 'aichallenge/workspace/src/aichallenge_submit/simple_trajectory_generator/data/raceline_awsim_30km_from_garage_interpolated.csv'

df = pd.read_csv(filename)

df["speed"] = df['speed']*2 

df.to_csv('test.csv',index=False)