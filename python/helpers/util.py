import pandas as pd
import numpy as np
from typing import List


""" 
week1 = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/tracking_week_1.csv")
plays = pd.read_csv("C:/Users/Michael Egle/BDB2025/data/plays.csv")
 """
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
    df['on_defense'] = np.where(df['club'] == df['defensiveTeam'], 1, 0)

    df = df[df['displayName'] != "football"]
    return df

# Taken from this on stackoverflow:
# https://stackoverflow.com/questions/23478297/pandas-dataframe-or-panel-to-3d-numpy-array
def make_cube(df: pd.DataFrame, idx_cols: List[str]) -> np.ndarray:
    """Make an array cube from a Dataframe

    Args:
        df: Dataframe
        idx_cols: columns defining the dimensions of the cube

    Returns:
        multi-dimensional array
    """
    assert len(set(idx_cols) & set(df.columns)) == len(idx_cols), 'idx_cols must be subset of columns'

    df = df.set_index(keys=idx_cols)  # don't overwrite a parameter, thus copy!
    idx_dims = [len(level) + 1 for level in df.index.levels]
    idx_dims.append(len(df.columns))

    cube = np.empty(idx_dims)
    cube.fill(np.nan)
    cube[tuple(np.array(df.index.to_list()).T)] = df.values

    return cube

# TODO - test all of this
# this is all just the python equivalent of what Ajay had in the R script
""" 
week1_new = process_tracking_data(week1, plays)

print(week1_new[['gameId', 'playId', 'nflId', 'frameId', 'x', 'rel_x', 'y', 'dir', 'o', 'absoluteYardlineNumber']])

train_df, test_df = train_test_split(week1_new, test_size = 0.2, random_state = 30)
train_df, val_df = train_test_split(week1_new, test_size = 0.25, random_state = 30)

print(train_df) """