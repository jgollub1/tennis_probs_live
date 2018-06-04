# modify this for your own path
# SCRIPT_PATH = '/Users/jacobgollub/Desktop/tennis_probs_live/match_data'
PROBS_PATH = '/Users/jacobgollub/Desktop/github_repos/tennis_probs_live/sackmann'
TOUR = 'atp'
START_YEAR = 2015
CURRENT_YEAR = 2018
DATE = '5_30_{}'.format(CURRENT_YEAR)
RET_STRINGS = ('ABN','DEF','In Progress','RET','W/O',' RET',' W/O','nan','walkover')
ABD_STRINGS = ('abandoned','ABN','ABD','DEF','def','unfinished','Walkover')

import sys
# sys.path.insert(0,SCRIPT_PATH)
sys.path.insert(0,PROBS_PATH)
from helper_functions import *
from data_functions import *
pd.options.mode.chained_assignment = None

if __name__=='__main__':
	# NOTE: could you use 'winner_id' to identify rather than correcting name mispellings?
	print 'main'
	match_df = concat_data(1968, 2018, TOUR)
	match_df = format_match_df(match_df,TOUR,ret_strings=RET_STRINGS,abd_strings=ABD_STRINGS)
	start_ind = match_df[match_df['match_year']>=START_YEAR-1].index[0]
	current_elo_ratings, match_df = generate_dfs(match_df, 1, start_ind)
	# match_df = generate_stats(match_df, start_ind) # 52, etc adj, tny
	current_52_stats = get_current_52_stats(match_df, start_ind)

	# merge relevant current statistics
	current_df = current_elo_ratings.merge(current_52_stats, on='player')
	current_df = generate_EM_stats_current(current_df, cols=['52_s_pct','52_r_pct']) # add EM normalization
	current_df.to_csv('match_data_constructed/current_match_df_{}'.format(DATE), index=False)
	print 'match_data_constructed/current_match_df_{}'.format(DATE) + ' constructed'

	# TO DO: produce s/r_kls, elo_induced
	match_df = match_df[match_df['match_year']>=START_YEAR].reset_index(drop=True) # shave off prev years
	match_df.to_csv('match_data_constructed/match_df_{}'.format(DATE), index=False)
	print 'match_data_constructed/match_df_{}'.format(DATE) + ' constructed'

	


	# TO DO: adj_stats, tourney_stats, elo_diffs, s/r_kls, elo_induced_stats

	# see connect_df() for switching perspective
	# need to paste equations for probs, kls percentages


	# # depending on ONLY_PBP, this will have point-by-point matches, or all
	# # tour-level matches from START_DATE to present
	# df = generate_stats(df,CURRENT_YEAR)
	# df = finalize_df(df)
	# name = 'elo_atp_matches_all_'+DATE
	# print name + '.csv saved to my_data'
	# df.to_csv('match_data/'+name+'.csv')
