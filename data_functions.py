import os
import sys
sys.path.insert(0, '{}/sackmann'.format(os.getcwd()))
import numpy as np
import pandas as pd
import elo_538 as elo
import re, datetime
import tennisGameProbability
import tennisMatchProbability
import tennisSetProbability
import tennisTiebreakProbability
from tennisMatchProbability import matchProb
from helper_functions import adj_stats_52, stats_52, tny_52, normalize_name
from sklearn import linear_model

# TO DO: add switches or global indicators for surface stats

'''
concatenate original match dataframes from years
(start_y, end_y)
'''
def concat_data(start_y, end_y, tour):
    atp_year_list = []
    for i in xrange(start_y, end_y+1):
        f_name = "match_data/"+tour+"_matches_{0}.csv".format(i)
        atp_year_list.append(pd.read_csv(f_name))
    return pd.concat(atp_year_list, ignore_index = True)

'''
data cleaning and formatting
normalize_name() is specific to atp/wta...
'''
def format_match_df(df,tour,ret_strings=[],abd_strings=[]):
    cols = [u'tourney_id', u'tourney_name', u'surface', u'draw_size', u'tourney_date', \
            u'match_num', u'winner_name', u'loser_name', u'score', u'best_of', u'w_svpt', \
            u'w_1stWon', u'w_2ndWon', u'l_svpt', u'l_1stWon', u'l_2ndWon']
    df = df[cols]
    df = df.rename(columns={'winner_name':'w_name','loser_name':'l_name','tourney_id':'tny_id',\
                            'tourney_name':'tny_name','tourney_date':'tny_date'})
    df['w_name'] = [normalize_name(x,tour) for x in df['w_name']]
    df['l_name'] = [normalize_name(x,tour) for x in df['l_name']]
    df['tny_name'] = ['Davis Cup' if 'Davis Cup' in s else s for s in df['tny_name']]
    df['tny_name'] = [s.replace('Australian Chps.','Australian Open').replace('Australian Open-2',\
                'Australian Open').replace('U.S. National Chps.','US Open') for s in df['tny_name']]
    df['is_gs'] = (df['tny_name'] == 'Australian Open') | (df['tny_name'] == 'Roland Garros') |\
                  (df['tny_name'] == 'Wimbledon')       | (df['tny_name'] == 'US Open')

    # format dates
    df['tny_date'] = [datetime.datetime.strptime(str(x), "%Y%m%d").date() for x in df['tny_date']]
    df['match_year'] = [x.year for x in df['tny_date']]
    df['match_month'] = [x.month for x in df['tny_date']]
    df['match_year'] = df['match_year'] + (df['match_month'] == 12) # correct december start dates
    df['match_month'] = [1 if month==12 else month for month in df['match_month']] # to following year
    df['score'] = [re.sub(r"[\(\[].*?[\)\]]", "", str(s)) for s in df['score']] # str(s) fixes any nans
    df['score'] = ['RET' if 'RET' in s else s for s in df['score']]
    df['w_swon'], df['l_swon'] = df['w_1stWon']+df['w_2ndWon'], df['l_1stWon']+df['l_2ndWon']
    df['w_rwon'], df['l_rwon'] = df['l_svpt']-df['l_swon'], df['w_svpt']-df['w_swon']
    df['w_rpt'], df['l_rpt'] = df['l_svpt'], df['w_svpt']
    df.drop(['w_1stWon','w_2ndWon','l_1stWon','l_2ndWon'], axis=1, inplace=True)

    # get rid of leading 0s in tny_id
    df['tny_id'] = ['-'.join(t.split('-')[:-1] + [t.split('-')[-1][1:]]) \
                    if t.split('-')[-1][0]=='0' else t for t in df['tny_id']]
    # get rid of matches involving a retirement
    df['score'] = ['ABN' if score.split(' ')[-1] in abd_strings else score for score in df['score']]
    ret_d = set(ret_strings)
    df = df.loc[[i for i in range(len(df)) if df['score'][i] not in ret_d]]
    df = df.sort_values(by=['tny_date','tny_name','match_num'], ascending=True).reset_index(drop=True)
    return df

'''
original dataset labels columns by 'w_'/'l_'
randomly assigning 'w'/'l' to 'p0','p1'
'''
def change_labels(df, cols):
    # change w,l TO p0,p1
    for col in cols:
        df['p0'+col] = [df['l'+col][i] if df['winner'][i] else df['w'+col][i] for i in xrange(len(df))]
        df['p1'+col] = [df['w'+col][i] if df['winner'][i] else df['l'+col][i] for i in xrange(len(df))]

    # add s/r pct columns
    p_hat = np.sum([df['p0_52_swon'],df['p1_52_swon']])/np.sum([df['p0_52_svpt'],df['p1_52_svpt']])
    for label in ['p0','p1']:
        df[label+'_s_pct'] = [p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_swon']/df[label+'_52_svpt'])]
        df[label+'_r_pct'] = [1-p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_rwon']/df[label+'_52_rpt'])]
        df[label+'_sf_s_pct'] = [p_hat if x==0 else x for x in np.nan_to_num(df[label+'_sf_52_swon']/df[label+'_sf_52_svpt'])]
        df[label+'_sf_r_pct'] = [1-p_hat if x==0 else x for x in np.nan_to_num(df[label+'_sf_52_rwon']/df[label+'_sf_52_rpt'])]

    for label in ['w', 'l']:
        df.drop([label + col for col in cols], axis=1, inplace=True)

    df['tny_name'] = [s if s==s else 'Davis Cup' for s in df['tny_name']]
    return df

'''
from start_ind (a year before start_year), collect cumulative
12-month s/r stats prior to each match
'''
def get_current_52_stats(df, start_ind):
    players_stats = {}
    active_players = {}
    w_l = ['p0', 'p1']
    start_date = (df['match_year'][start_ind],df['match_month'][start_ind])
    avg_stats = stats_52(start_date)
    avg_stats.update(start_date,(6.4,10,3.6,10)) # set as prior so first row is not nan

    for i, row in df[start_ind:].iterrows():
        date = row['match_year'],row['match_month']
        avg_stats.set_month(date)
        for k,label in enumerate(w_l):
            if row[label+'_name'] not in players_stats:
                players_stats[row[label+'_name']] = stats_52(date)
            # store serving stats prior to match, update current month
            players_stats[row[label+'_name']].set_month(date)
            if row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt']:
                match_stats = (row[label+'_swon'],row[label+'_svpt'],row[w_l[1-k]+'_svpt']-\
                                row[w_l[1-k]+'_swon'],row[w_l[1-k]+'_svpt'])
                players_stats[row[label+'_name']].update(date,match_stats)
                avg_stats.update(date,match_stats)

            active_players[row[label+'_name']] = 1 # log active player

    # update every player to current month
    for player in active_players.keys():
        players_stats[player].set_month(date)

    players = active_players.keys()
    current_52_stats = [[player] + list(np.sum(players_stats[player].last_year,axis=0)) \
                        for player in players]
    # avg_52_stats = np.sum(avg_stats.last_year,axis=0)
    cols = ['player','52_swon','52_svpt','52_rwon','52_rpt']
    current_stats_df = pd.DataFrame(current_52_stats, columns=cols)
    current_stats_df['52_s_pct'] = current_stats_df['52_swon']/current_stats_df['52_svpt']
    current_stats_df['52_r_pct'] = current_stats_df['52_rwon']/current_stats_df['52_rpt']
    return current_stats_df[current_stats_df['52_svpt']>0] # return players active in past 12 months

'''
generate 12-month stats for Barnett-Clarke model
as well as variations (adjusted, EM-normalized)
'''
def generate_stats(df, start_ind):
    df = generate_52_stats(df,start_ind)
    df = generate_52_adj_stats(df,start_ind)
    df = generate_tny_stats(df,start_ind)
    df = generate_52_commop_stats(df, start_ind)

    cols = ['_name','_elo_538','_sf_elo_538', #'_elo','_sf_elo'
        '_sf_52_swon','_sf_52_svpt','_sf_52_rwon','_sf_52_rpt',
        '_swon', '_svpt', '_rwon', '_rpt',
        '_52_swon', '_52_svpt','_52_rwon','_52_rpt','_52_s_adj','_52_r_adj']

    df['winner'] = np.random.choice([0,1], df.shape[0])
    df = change_labels(df, cols)

    df['elo_diff'] = df['p0_elo_538'] - df['p1_elo_538']
    df['sf_elo_diff'] = df['p0_sf_elo_538'] - df['p1_sf_elo_538']

    # # dataframe with only official matches
    # df = df[df['winner']!='None']
    # df = df.reset_index(drop=True)
    # cols = ['52_s_adj','52_r_adj']

    em_cols = ['s_pct', 'r_pct', '52_s_adj', '52_r_adj']
    df = generate_sr_pct(df)

    # FIX for correct em stat sample sizes
    df = df.loc[start_ind:].reset_index(drop=True)
    df = generate_EM_stats(df, em_cols)
    return df

'''
add s/r pct columns, replacing nan with overall avg
'''
def generate_sr_pct(df):
    p_hat = np.sum([df['p0_52_swon'],df['p1_52_swon']])
    p_hat = p_hat/np.sum([df['p0_52_svpt'],df['p1_52_svpt']])
    for label in ['p0','p1']:
        # df[label+'_s_pct'] = [p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_swon']/df[label+'_52_svpt'])]
        # df[label+'_r_pct'] = [1-p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_rwon']/df[label+'_52_rpt'])]
        df[label+'_s_pct'] = np.nan_to_num(df[label+'_52_swon']/df[label+'_52_svpt'])
        df[label+'_r_pct'] = np.nan_to_num(df[label+'_52_rwon']/df[label+'_52_rpt'])
        df[label+'_s_pct'] = df[label+'_s_pct'] + (p_hat) * (df[label+'_s_pct'] == 0)
        df[label+'_r_pct'] = df[label+'_r_pct'] + (1-p_hat)*(df[label+'_r_pct'] == 0)
        # df[label+'_sf_s_pct'] = [p_hat if x==0 else x for x in np.nan_to_num(df[label+'_sf_52_swon']/df[label+'_sf_52_svpt'])]
        # df[label+'_sf_r_pct'] = [1-p_hat if x==0 else x for x in np.nan_to_num(df[label+'_sf_52_rwon']/df[label+'_sf_52_rpt'])]

        # finally, generate the observed service percentages in each match
        df[label+'_s_pct_obsv'] = np.nan_to_num(df[label+'_swon']/df[label+'_svpt'])
    return df

def finalize_df(df):
    # generate serving probabilities for Klaassen-Magnus model
    df['match_id'] = range(len(df))
    df['tny_stats'] = [df['avg_52_s'][i] if df['tny_stats'][i]==0 else df['tny_stats'][i] for i in xrange(len(df))]
    df['p0_s_kls'] = df['tny_stats']+(df['p0_s_pct']-df['avg_52_s']) - (df['p1_r_pct']-df['avg_52_r'])
    df['p1_s_kls'] = df['tny_stats']+(df['p1_s_pct']-df['avg_52_s']) - (df['p0_r_pct']-df['avg_52_r'])
    df['p0_s_kls_EM'] = df['tny_stats']+(df['p0_s_pct_EM']-df['avg_52_s']) - (df['p1_r_pct_EM']-df['avg_52_r'])
    df['p1_s_kls_EM'] = df['tny_stats']+(df['p1_s_pct_EM']-df['avg_52_s']) - (df['p0_r_pct_EM']-df['avg_52_r'])

    df['p0_s_sf_kls'] = df['tny_stats']+(df['p0_sf_s_pct']-df['sf_avg_52_s']) - (df['p1_sf_r_pct']-df['sf_avg_52_r'])
    df['p1_s_sf_kls'] = df['tny_stats']+(df['p1_sf_s_pct']-df['sf_avg_52_s']) - (df['p0_sf_r_pct']-df['sf_avg_52_r'])
    # df['p0_s_sf_kls_EM'] = df['tny_stats']+(df['p0_sf_s_pct_EM']-df['sf_avg_52_s']) - (df['p1_sf_r_pct_EM']-df['sf_avg_52_r'])
    # df['p1_s_sf_kls_EM'] = df['tny_stats']+(df['p1_sf_s_pct_EM']-df['sf_avg_52_s']) - (df['p0_sf_r_pct_EM']-df['sf_avg_52_r'])
    df['p0_s_adj_kls'] = df['tny_stats']+(df['p0_52_s_adj']) - (df['p1_52_r_adj'])
    df['p1_s_adj_kls'] = df['tny_stats']+(df['p1_52_s_adj']) - (df['p0_52_r_adj'])
    # df['p0_s_adj_kls_EM'] = df['tny_stats']+(df['p0_52_s_adj_EM']) - (df['p1_52_r_adj_EM'])
    # df['p1_s_adj_kls_EM'] = df['tny_stats']+(df['p1_52_s_adj_EM']) - (df['p0_52_r_adj_EM'])

    # generate match probabilities and z-scores for Klaassen method, with and w/o JS estimators
    df['match_prob_kls'] = [matchProb(row['p0_s_kls'],1-row['p1_s_kls']) for i,row in df.iterrows()]
    df['match_prob_kls_EM'] = [matchProb(row['p0_s_kls_EM'],1-row['p1_s_kls_EM']) for i,row in df.iterrows()]
    df['match_prob_sf_kls'] = [matchProb(row['p0_s_sf_kls'],1-row['p1_s_sf_kls']) for i,row in df.iterrows()]
    # df['match_prob_sf_kls_EM'] = [matchProb(row['p0_s_sf_kls_EM'],1-row['p1_s_sf_kls_EM']) for i,row in df.iterrows()]
    df['match_prob_adj_kls'] = [matchProb(row['p0_s_adj_kls'],1-row['p1_s_adj_kls']) for i,row in df.iterrows()]
    # df['match_prob_adj_kls_EM'] = [matchProb(row['p0_s_adj_kls_EM'],1-row['p1_s_adj_kls_EM']) for i,row in df.iterrows()]

    # generate win probabilities from elo differences
    df['elo_prob'] = (1+10**(df['elo_diff']/-400.))**-1
    # df['elo_prob_538'] = (1+10**(df['elo_diff_538']/-400.))**-1
    df['sf_elo_prob'] = [(1+10**(diff/-400.))**-1 for diff in df['sf_elo_diff']]
    # df['sf_elo_prob_538'] = [(1+10**(diff/-400.))**-1 for diff in df['sf_elo_diff_538']]

    # elo-induced serve percentages
    df = generate_elo_induced_s(df, 'elo',start_ind=0)
    # df = generate_elo_induced_s(df, 'logit_elo_538',start_ind=0)
    return df

'''
returns two dataframes
1) contains up-to-date player stats through date of most recent match
2) contains every match with elo/serve/return/etc stats
'''
def generate_dfs(df, counts_i, start_ind):
    active_df, df = generate_elo(df, counts_i)
    df = generate_stats(df, start_ind) # 52, adj, tny, etc.
    df = finalize_df(df)
    return active_df, df

'''
iterate through every historical match, providing
up-to-date elo ratings for each player prior to match
generates two dataframes
1) match dataframe with each player's pre-match elo ratings
2) player dataframe with each player's current elo ratings
   (through input df's most recent match)
** considers surface ratings as well
'''
def generate_elo(df, counts_i):
    players_list = np.union1d(df.w_name, df.l_name)
    player_count = len(players_list)
    initial_elos = [elo.Rating() for __ in range(player_count)]
    players_elo = dict(zip(players_list, initial_elos))
    sf_elo = {}
    for sf in ('Hard','Clay','Grass'):
        initial_elos = [elo.Rating() for __ in range(player_count)]
        sf_elo[sf] = dict(zip(players_list, initial_elos))

    elo_1s, elo_2s = [],[]
    sf_elo_1s, sf_elo_2s = [],[]
    elo_obj = elo.Elo_Rater()

    current_month = df['match_month'][0]
    active_players = {current_month: set([])} # active in past twelve months

    # update player elo from every recorded match
    for i, row in df.iterrows():
        if row['match_month'] != current_month:
            current_month = row['match_month']
            active_players[current_month] = set([])

        sf,is_gs = row['surface'],row['is_gs']
        w_name, l_name = row['w_name'], row['l_name']
        w_elo,l_elo = players_elo[w_name],players_elo[l_name]
        active_players[current_month].add(w_name) # track current players
        active_players[current_month].add(l_name)
        elo_1s.append(w_elo.value)
        elo_2s.append(l_elo.value)
        elo_obj.rate_1vs1(w_elo,l_elo,is_gs,counts_i)

        if sf in ('Hard','Clay','Grass'):
            w_sf_elo,l_sf_elo = sf_elo[sf][w_name],sf_elo[sf][l_name]
            sf_elo_1s.append(w_sf_elo.value)
            sf_elo_2s.append(l_sf_elo.value)
            elo_obj.rate_1vs1(w_sf_elo,l_sf_elo,is_gs,counts_i)
        else:
            sf_elo_1s.append(w_elo.value)
            sf_elo_2s.append(l_elo.value)

    players = active_players.values()
    players = list(set.union(*players))
    active_players_elo = [[players_elo[player].value] for player in players]
    active_players_elo = dict(zip(players, active_players_elo))

    for sf in ('Hard','Clay','Grass'):
        for player in players:
            active_players_elo[player] += [sf_elo[sf][player].value]
    active_df = pd.DataFrame([key]+val for key,val in active_players_elo.iteritems())
    active_df.columns = ['player', 'elo', 'hard_elo', 'clay_elo', 'grass_elo']
    active_df = active_df.sort_values(by=['elo'], ascending=False)

    tag = '_538' if counts_i else ''
    df['w_elo'+tag], df['l_elo'+tag] = elo_1s, elo_2s
    df['w_sf_elo'+tag], df['l_sf_elo'+tag] = sf_elo_1s, sf_elo_2s
    return active_df, df

'''
collect 12-month s/r average performance by player
'''
def generate_52_stats(df,start_ind):
    players_stats = {}
    start_date = (df['match_year'][start_ind],df['match_month'][start_ind])
    avg_stats = stats_52(start_date)
    # set as prior so first row is not nan
    avg_stats.update(start_date,(6.4,10,3.6,10))
    # array w/ 2x1 arrays for each player's 12-month serve/return performance
    match_52_stats = np.zeros([2,len(df),4])
    avg_52_stats = np.zeros([len(df),4]) # avg tour-wide stats for serve, return

    s_players_stats = {}
    s_avg_stats = {}
    for surface in ('Hard','Clay','Grass'):
        s_players_stats[surface] = {}
        s_avg_stats[surface] = stats_52((df['match_year'][0],df['match_month'][0]))
        s_avg_stats[surface].update(start_date,(6.4,10,3.6,10))
    s_match_52_stats = np.zeros([2,len(df),4])
    s_avg_52_stats = np.zeros([len(df),4])

    w_l = ['w','l']
    for i, row in df.loc[start_ind:].iterrows():
        surface = row['surface']
        date = row['match_year'],row['match_month']

        avg_stats.set_month(date)
        avg_52_stats[i] = np.sum(avg_stats.last_year,axis=0)
        for k,label in enumerate(w_l):
            if row[label+'_name'] not in players_stats:
                players_stats[row[label+'_name']] = stats_52(date)
            # store serving stats prior to match, update current month
            players_stats[row[label+'_name']].set_month(date)
            match_52_stats[k][i] = np.sum(players_stats[row[label+'_name']].last_year,axis=0) # all four stats per player
            # update serving stats if not null
            if row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt']:
                match_stats = (row[label+'_swon'],row[label+'_svpt'],row[w_l[1-k]+'_svpt']-\
                                row[w_l[1-k]+'_swon'],row[w_l[1-k]+'_svpt'])
                players_stats[row[label+'_name']].update(date,match_stats)
                avg_stats.update(date,match_stats)

        # repeat above process for surface-specific stats
        if surface not in ('Hard','Clay','Grass'):
            continue
        s_avg_stats[surface].set_month(date)
        s_avg_52_stats[i] = np.sum(s_avg_stats[surface].last_year,axis=0)
        for k,label in enumerate(w_l):
            if row[label+'_name'] not in s_players_stats[surface]:
                s_players_stats[surface][row[label+'_name']] = stats_52(date)

            # store serving stats prior to match, from current month
            s_players_stats[surface][row[label+'_name']].set_month(date)
            s_match_52_stats[k][i] = np.sum(s_players_stats[surface][row[label+'_name']].last_year,axis=0)
            # update serving stats if not null
            if row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt']:
                match_stats = (row[label+'_swon'],row[label+'_svpt'],row[w_l[1-k]+'_svpt']-\
                                row[w_l[1-k]+'_swon'],row[w_l[1-k]+'_svpt'])
                s_players_stats[surface][row[label+'_name']].update(date,match_stats)
                s_avg_stats[surface].update(date,match_stats)

    # sf_ tags are optional
    for k,label in enumerate(w_l):
        df[label+'_52_swon'] = match_52_stats[k][:,0]
        df[label+'_52_svpt'] = match_52_stats[k][:,1]
        df[label+'_52_rwon'] = match_52_stats[k][:,2]
        df[label+'_52_rpt'] = match_52_stats[k][:,3]
        df[label+'_sf_52_swon'] = s_match_52_stats[k][:,0]
        df[label+'_sf_52_svpt'] = s_match_52_stats[k][:,1]
        df[label+'_sf_52_rwon'] = s_match_52_stats[k][:,2]
        df[label+'_sf_52_rpt'] = s_match_52_stats[k][:,3]

    with np.errstate(divide='ignore', invalid='ignore'):
        df['avg_52_s'] = np.divide(avg_52_stats[:,0],avg_52_stats[:,1])
        df['avg_52_r'] = np.divide(avg_52_stats[:,2],avg_52_stats[:,3])
        df['sf_avg_52_s'] = np.divide(s_avg_52_stats[:,0],s_avg_52_stats[:,1])
        df['sf_avg_52_r'] = np.divide(s_avg_52_stats[:,2],s_avg_52_stats[:,3])
    return df

'''
Efron-Morris estimators for 52-week serve and return percentages
Calculates B_i coefficients in terms of service points
Feed any existing col where ['p0_'+col, 'p1_'+col] within df.columns
'''
def generate_EM_stats(df,cols):
    for col in cols:
        stat_history = np.concatenate([df['p0_'+col],df['p1_'+col]],axis=0)
        n = len(stat_history)/2
        group_var = np.var(stat_history)
        sr_col = 'svpt' if '_s_' in col else 'rpt'
        num_points = np.concatenate([df['p0_52_'+sr_col],df['p1_52_'+sr_col]])
        p_hat = np.mean(stat_history)
        sigma2_i = np.divide(p_hat*(1-p_hat),num_points,where=num_points>0)
        tau2_hat = np.nanvar(stat_history)
        B_i = sigma2_i/(tau2_hat+sigma2_i)

        stat_history[stat_history!=stat_history] = p_hat
        group_var = np.var(stat_history)
        df['p0_'+col+'_EM'] = df['p0_'+col]+B_i[:n]*(p_hat-df['p0_'+col])
        df['p1_'+col+'_EM'] = df['p1_'+col]+B_i[n:]*(p_hat-df['p1_'+col])
        print col, p_hat
    return df # ok if p_hats don't add up because they're avg of averages


'''
Efron-Morris estimators for 52-week serve and return percentages
Calculates B_i coefficients in terms of service points
Feed any existing col within df.columns
'''
def generate_EM_stats_current(df,cols):
    for col in cols:
        stat_history = df[col]
        n = len(stat_history)
        group_var = np.var(stat_history)
        num_points = df['52_svpt'] if col=='52_swon' else df['52_rpt']
        p_hat = np.mean(stat_history)
        sigma2_i = np.divide(p_hat*(1-p_hat),num_points,where=num_points>0)
        tau2_hat = np.nanvar(stat_history)
        B_i = sigma2_i/(tau2_hat+sigma2_i)

        stat_history[stat_history!=stat_history] = p_hat
        group_var = np.var(stat_history)
        df[col+'_EM'] = df[col]+B_i*(p_hat-df[col])
        print col, p_hat
    return df # ok if p_hats don't add up because they're avg of averages

'''
collects 12-month s/r stats relative to historical opponents
columns '52_s_adj','52_r_adj' represent how well a player
performs above average
'''
def generate_52_adj_stats(df,start_ind=0):
    players_stats = {}
    match_52_stats = np.zeros([2,len(df),2]) # 2x1 arrays for x_i, x_j's 12-month s/r performance

    w_l = ['w','l']
    for i, row in df.loc[start_ind:].iterrows():
        surface = row['surface']
        date = row['match_year'],row['match_month']
        avg_52_s,avg_52_r = row['avg_52_s'],row['avg_52_r']
        match_stats = [[],[]]

        # add new players to the dictionary
        for k,label in enumerate(w_l):
            if row[label+'_name'] not in players_stats:
                players_stats[row[label+'_name']] = adj_stats_52(date)

        # store pre-match adj stats
        for k,label in enumerate(w_l):
            players_stats[row[label+'_name']].set_month(date)

            # fill in player's adjusted stats prior to start of match
            match_52_stats[k][i] = players_stats[row[label+'_name']].adj_sr
            # update serving stats if not null
            if row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt']:
                sv_stats = (row[label+'_swon'],row[label+'_svpt'],row[label+'_rwon'],row[label+'_rpt'])
                opp_r_ablty = players_stats[row[w_l[1-k]+'_name']].adj_sr[1]+avg_52_r
                opp_s_ablty = players_stats[row[w_l[1-k]+'_name']].adj_sr[0]+avg_52_s
                opp_stats = (opp_r_ablty*row[label+'_svpt'], opp_s_ablty*row[label+'_rpt'])
                match_stats[k] = sv_stats+opp_stats

        # update players' adjusted scores based on pre-match adjusted ratings
        for k,label in enumerate(w_l):
            if row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt']:
                players_stats[row[label+'_name']].update(date,match_stats[k])

    for k,label in enumerate(w_l):
        df[label+'_52_s_adj'] = match_52_stats[k][:,0]
        df[label+'_52_r_adj'] = match_52_stats[k][:,1]
    return df

'''
collect yearly tournament serve averages for 'f_av'
in Barnette-Clark equation
'''
def generate_tny_stats(df,start_ind=0):
    tny_stats = {}
    tny_52_stats = np.zeros(len(df))
    for i, row in df.loc[start_ind:].iterrows():
        if row['tny_name']=='Davis Cup':
            continue

        year,t_id = row['tny_id'].split('-')
        year = int(year)
        match_stats = (row['w_swon']+row['l_swon'],row['w_svpt']+row['l_svpt'])
        # handle nan cases, provide tny_stats if possible
        if row['w_swon']!=row['w_swon']:
            if t_id in tny_stats:
                if year-1 in tny_stats[t_id].historical_avgs:
                    swon,svpt = tny_stats[t_id].historical_avgs[year-1]
                    tny_52_stats[i] = swon/float(svpt)
            continue
        # create new object if needed, then update
        elif t_id not in tny_stats:
            tny_stats[t_id] = tny_52(year)
        tny_52_stats[i] = tny_stats[t_id].update(year,match_stats)

    df['tny_stats'] = tny_52_stats
    return df

def generate_logit_probs(df,cols,col_name):
    lm = linear_model.LogisticRegression(fit_intercept = True)
    df_train = df[df['match_year'].isin([2011,2012,2013])]
    df_train = df_train[df_train['winner'].isin([0,1])]
    df_train['winner'] = df_train['winner'].astype(int)
    lm.fit(df_train[cols].values.reshape([df_train.shape[0],len(cols)]),np.asarray(df_train['winner']))
    print 'cols: ', cols
    print 'lm coefficients: ', lm.coef_
    df[col_name] = lm.predict_proba(df[cols].values.reshape([df.shape[0],len(cols)]))[:,0]
    return df

'''
approximate inverse elo-->s_pct calculator
'''
def elo_induced_s(prob,s_total):
    s0 = s_total/2
    current_prob = .5
    diff = s_total/4
    while abs(current_prob-prob)>.001:
        if current_prob < prob:
            s0 += diff
        else:
            s0 -= diff
        diff /= 2
        current_prob = matchProb(s0,1-(s_total-s0))
    return s0,s_total-s0

'''
import to set s_total with JS-normalized percentages
'''
def generate_elo_induced_s(df,col,start_ind=0):
    df['s_total'] = df['p0_s_kls_EM'] + df['p1_s_kls_EM']
    induced_s = np.zeros([len(df),2])
    for i, row in df.loc[start_ind:].iterrows():
        induced_s[i] = elo_induced_s(row[col+'_prob'],row['s_total'])
    df['p0_s_kls_'+col] = induced_s[:,0]
    df['p1_s_kls_'+col] = induced_s[:,1]
    return df
