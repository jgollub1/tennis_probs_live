import pandas as pd
import numpy as np
from globals import *
from data_functions import *

# only run this once
# def build_test_df():
#     df = concat_data(1968, 2018, TOUR)
#     df = format_match_df(df, TOUR, RET_STRINGS, ABD_STRINGS)
#     test_df = df[(df.match_year >= 2012) & (df.match_year <= 2015)].dropna().reset_index()
#     test_df.to_csv('match_data_constructed/test_df.csv')

#     active_df, elo_df = generate_elo(df, counts_538)
#     active_df.to_csv('match_data_constructed/test_df_active_elo.csv')


def test_elo(df, counts_538):
    active_df, elo_df = generate_elo(df, counts_538)
    active_df_test = pd.read_csv('match_data_constructed/test_df_active_elo.csv')

    assert(np.array_equal([elo_df.w_elo_538[28], 1722.09]))
    assert(np.array_equal([elo_df.l_elo_538[28], 1724.75]))

    for col in ['elo', 'hard_elo', 'clay_elo', 'grass_elo']:
        print 'testing: ', col
        assert(np.array_equal(active_df[col], active_df_test[col]))

def test_52_stats(df):
    return

def test_52_adj_stats(df):
    return

def test_52_commop_stats(df):
    return
