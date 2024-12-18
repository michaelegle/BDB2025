import pandas as pd
import numpy as np
from typing import List
import torch
import math

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
    df['dir'] = df['dir'] / 180 * np.pi
    df['o'] = np.where(df['playDirection'] == 'left', df['o'] + 180, df['o'])
    df['o'] = np.where(df['o'] > 360, df['o'] - 360, df['o'])
    df['o'] = df['o'] / 180 * np.pi
    df['s_x'] = np.cos(df['dir']) * df['s']
    df['s_y'] = np.sin(df['dir']) * df['s']
    df['a_x'] = np.cos(df['dir']) * df['a']
    df['a_y'] = np.sin(df['dir']) * df['a']
    df['absoluteYardlineNumber'] = np.where(df['playDirection'] == 'left', 120 - df['absoluteYardlineNumber'], df['absoluteYardlineNumber'])
    df['rel_x'] = df['x'] - df['absoluteYardlineNumber']
    df['on_defense'] = np.where(df['club'] == df['defensiveTeam'], 1, 0)

    df = df[df['displayName'] != "football"]

    # Update the pass coverage labels
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == 'Cover-3 Seam', 'Cover-3', df['pff_passCoverage'])
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == 'Cover 6-Left', 'Cover-6', df['pff_passCoverage'])
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == 'Cover-6 Right', 'Cover-6', df['pff_passCoverage'])
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == 'Cover-3 Cloud Right', 'Cover-3', df['pff_passCoverage'])
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == 'Cover-3 Cloud Left', 'Cover-3', df['pff_passCoverage'])
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == 'Cover-3 Double Cloud', 'Cover-3', df['pff_passCoverage'])
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == 'Cover-1 Double', 'Cover-1', df['pff_passCoverage'])
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == '2-Man', 'Cover-2', df['pff_passCoverage'])
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == 'Red Zone', 'Red Zone/Goal Line', df['pff_passCoverage'])
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == 'Red-Zone', 'Red Zone/Goal Line', df['pff_passCoverage'])
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == 'Goal Line', 'Red Zone/Goal Line', df['pff_passCoverage'])
    df['pff_passCoverage'] = np.where(df['pff_passCoverage'] == 'Prevent', 'Miscellaneous', df['pff_passCoverage'])

    df = df[df['pff_passCoverage'].notnull()]
    df = df[df['qbKneel'] == 0]
    df = df[df['qbSpike'] == False]
    df = df.sort_values(['gameId', 'playId', 'frameId', 'y'])
    return df

def add_relative_features(df, players):



    # This will add features such as orientation to QB as well as the other columns relative to the offensive players
    df = pd.merge(df, players[['nflId', 'position']],
                  how = 'inner',
                  on = 'nflId')
    
    # code adapted to Python from ngscleanR R package https://github.com/guga31bb/ngscleanR/blob/master/R/cleaning_functions.R 
    df['qb_x'] = np.where(df['position'] == 'QB', df['x'], None)
    df['qb_y'] = np.where(df['position'] == 'QB', df['y'], None)
    df['qb_x'] = df.groupby(['gameId', 'playId', 'frameId'])['qb_x'].apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    df['qb_y'] = df.groupby(['gameId', 'playId', 'frameId'])['qb_y'].apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

    df['dis_to_qb_x'] = df['qb_x'] - df['x']
    df['dis_to_qb_y'] = df['qb_y'] - df['y']

    df['rel_o_vs_qb'] = np.arctan2(df['dis_to_qb_y'], df['dis_to_qb_x']) * 180 / np.pi
    df['rel_o_vs_qb'] = np.where(df['rel_o_vs_qb'] < 0, df['rel_o_vs_qb'] + 360, df['rel_o_vs_qb'])
    df['rel_o_vs_qb'] = np.where(df['rel_o_vs_qb'] > 360, df['rel_o_vs_qb'] - 360, df['rel_o_vs_qb'])
    df['rel_o_vs_qb'] = np.minimum(np.absolute(360 - (df['o'] - df['rel_o_vs_qb'])), df['o'] - df['rel_o_vs_qb'])

    off_df = df[df['on_defense'] == 0]
    def_df = df[df['on_defense'] == 1]

    off_df = off_df[['gameId', 'playId', 'frameId', 'nflId', 'x', 'y', 's_x', 's_y', 'a_x', 'a_y']]

    new_df = pd.merge(def_df, off_df,
                      how = 'left',
                      on = ['gameId', 'playId', 'frameId'],
                      suffixes = ['', '_off'])
    
    print(new_df)
    print(new_df.columns)
    # values relative to the given offensive player
    new_df['rel_off_x'] = new_df['x'] - new_df['x_off']
    new_df['rel_off_y'] = new_df['y'] - new_df['y_off']
    new_df['rel_off_s_x'] = new_df['s_x'] - new_df['s_x_off']
    new_df['rel_off_s_y'] = new_df['s_y'] - new_df['s_y_off']
    new_df['rel_off_a_x'] = new_df['a_x'] - new_df['a_x_off']
    new_df['rel_off_a_y'] = new_df['a_y'] - new_df['a_y_off']


    return new_df

def reformat_model_data(df, week_number, device):
    # Only take relevant columns
    df_x = df[['x', 'rel_x', 'y', 'dir', 'o', 's_x', 's_y', 'a_x', 'a_y', 'rel_off_x', 'rel_off_y', 'rel_off_s_x', 'rel_off_s_y', 'rel_off_a_x', 'rel_off_a_y', 'gameId', 'playId', 'frameId', 'nflId', 'nflId_off']]
    df_y = df[['gameId', 'playId', 'frameId', 'pff_passCoverage']].drop_duplicates()

    print(df_x)

    features = ['rel_off_x', 'rel_off_y', 'rel_off_s_x', 'rel_off_s_y', 'rel_off_a_x', 'rel_off_a_y', 'x', 'rel_x', 'y', 'dir', 'o', 's_x', 's_y', 'a_x', 'a_y']


    df_x_longer = df_x.melt(id_vars = ['nflId', 'gameId', 'playId', 'frameId', 'nflId_off'],
                            value_vars = features,
                            var_name = 'variable',
                            value_name = 'value')

    df_x_longer['value_rank'] = df_x_longer.groupby(['gameId', 'playId', 'frameId', 'nflId', 'variable'])['nflId_off'].rank(method = 'dense')
    df_x_longer = df_x_longer.reset_index()
    df_x_longer['value_rank'] = df_x_longer['value_rank'].astype(int)
    df_x_longer['new_variable_name'] = df_x_longer['variable'] + '_' + df_x_longer['value_rank'].astype(str)


    df_x_pivoted = df_x_longer.pivot(index = ['gameId', 'playId', 'frameId', 'nflId'],
                                     columns = 'new_variable_name',
                                     values = 'value')

    data_path = 'data/x_data_order_week_{}.csv'.format(week_number)
    df_x_pivoted = df_x_pivoted.reset_index()
    df_x_pivoted[['gameId', 'playId', 'frameId', 'nflId']].drop_duplicates().to_csv(data_path)

    df_x_pivoted.drop(['gameId', 'playId', 'frameId', 'nflId'], axis = 1, inplace = True)

    # reshape() is weird in how it fills in the 4d tensor. So this is a brief rundown of how this works
    # create 4d tensor of size (unique frames) x (11 defensive players) x (number of features) x (11 offensive players to be compared)
    x_tensor = df_x_pivoted.values.reshape(df_x_pivoted.shape[0] // 11, 11, len(features), 11)
    # Then transpose the tensor on the feature and defensive player axes, so we now have dimensions of:
    # (unique frames) x (number of features) x (11 defensive players) x (11 offensive players to be compared)
    x_tensor = torch.tensor(x_tensor).transpose(1, 2).to(device = device).to(torch.float32)

    print(x_tensor)
    #torch.set_printoptions(sci_mode = False)


    # TODO - might be best to have a more sensical ordering for these indices, but since you can just map them back together at the end, it's not a huge deal
    coverage_class_to_index = {
        "Cover-3": 0,
        "Quarters": 1,
        "Cover-1": 2,
        "Cover-6": 3, 
        "Cover-2": 4,
        "Cover-0": 5,
        "Red Zone/Goal Line": 6,
        "Miscellaneous": 7,
        "Bracket": 8
    }

    y_tensor = torch.tensor([coverage_class_to_index[t] for t in df_y['pff_passCoverage']]).to(device=device)

    #print(subset_labels['pff_passCoverage'].unique())
    return x_tensor, y_tensor

# TODO
# Might want to make this more advanced at some point
def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    #print(predicted)
    #print(labels)
    correct_predictions = (predicted == labels).sum().item()
    return(correct_predictions / labels.size(0))