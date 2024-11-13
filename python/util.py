import pandas as pd
import numpy as np

test = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_1.csv")
plays = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/plays.csv")

def process_tracking_data(df, plays):
    df = pd.merge(df, plays,
                  how = 'inner',
                  on = ['gameId', 'playId'])

    df['x'] = np.where(df['playDirection'] == 'left', 120 - df['x'], df['x'])
    df['y'] = np.where(df['playDirection'] == 'left', 160 / 3 - df['y'], df['y'])
    df['dir'] = np.where(df['playDirection'] == 'left', df['dir'] + 180, df['dir'])
    df['dir'] = np.where(df['dir'] > 360, df['dir'] - 360, df['dir'])
    df['o'] = np.where(df['playDirection'] == 'left', df['o'] + 180, df['o'])
    df['o'] = np.where(df['o'] > 360, df['o'] - 360, df['o'])
    df['absoluteYardlineNumber'] = np.where(df['playDirection'] == 'left', 120 - df['absoluteYardlineNumber'], df['absoluteYardlineNumber'])
    df['rel_x'] = df['x'] - df['absoluteYardlineNumber']

    return df

print(test)

# TODO - test all of this
# this is all just the python equivalent of what Ajay had in the R script

test_new = process_tracking_data(test, plays)

print(test_new[['gameId', 'playId', 'nflId', 'frameId', 'x', 'rel_x', 'y', 'dir', 'o', 'absoluteYardlineNumber']])