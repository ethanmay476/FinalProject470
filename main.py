#%%
# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
#%%
# Defining data for the dataframe
current_dir = os.getcwd()

data_path = os.path.join(current_dir, 'spambase.csv')

data = pd.read_csv(data_path)

