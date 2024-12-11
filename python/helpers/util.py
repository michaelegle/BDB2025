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

    # df = df.sort_values('y')
    df = df[df['pff_passCoverage'].notnull()]
    #df = df[df['on_defense'] == 1]
    df = df[df['qbKneel'] == 0]
    df = df[df['qbSpike'] == False]
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

    # TODO - build in the join for offense/defense
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

# TODO
# Might want to make this more advanced at some point
def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    #print(predicted)
    #print(labels)
    correct_predictions = (predicted == labels).sum().item()
    return(correct_predictions / labels.size(0))