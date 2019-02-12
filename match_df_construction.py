import os
import sys
from datetime import datetime as dt
from helper_functions import *
from data_functions import *

pd.options.mode.chained_assignment = None
sys.path.insert(0, '{}/sackmann'.format(os.getcwd()))
START_YEAR = 2010
TOUR = 'atp'
RET_STRINGS = ('ABN','DEF','In Progress','RET','W/O',' RET',' W/O','nan','walkover')
ABD_STRINGS = ('abandoned','ABN','ABD','DEF','def','unfinished','Walkover')
DATE = dt.now().strftime(('%m_%d_%Y'))

if __name__=='__main__':
	print 'main'
	print 'currently here: ', os.getcwd()
	match_df = concat_data(1968, 2018, TOUR)
	match_df = format_match_df(match_df,TOUR,ret_strings=RET_STRINGS,abd_strings=ABD_STRINGS)
	start_ind = match_df[match_df['match_year']>=START_YEAR-1].index[0]
	current_elo_ratings, match_df = generate_dfs(match_df, 1, start_ind)
	start_ind = 0
	current_52_stats = get_current_52_stats(match_df, start_ind)

	current_file_path = 'match_data_constructed/current_match_df_{}.csv'.format(DATE)
	current_df = current_elo_ratings.merge(current_52_stats, on='player')
	current_df = generate_EM_stats_current(current_df, cols=['52_s_pct','52_r_pct'])
	current_df.to_csv(current_file_path, index=False)
	print '{} constructed '.format(current_file_path)

	match_file_path = 'match_data_constructed/match_df_{}.csv'.format(DATE)
	match_df = match_df[match_df['match_year']>=START_YEAR].reset_index(drop=True) # shave off prev years
	match_df.to_csv(match_file_path, index=False)
	print '{} constructed'.format(match_file_path)

	


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
