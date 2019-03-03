import os
import sys
from globals import *
from helper_functions import *
from data_functions import *

sys.path.insert(0, '{}/sackmann'.format(os.getcwd()))

if __name__=='__main__':
	print 'main: ', os.getcwd()
	current_year = int(DATE.split('_')[-1])
	current_df, match_df = generate_dfs(TOUR, START_YEAR, current_year, RET_STRINGS, ABD_STRINGS, COUNTS_538)

	current_file_path = 'match_data_constructed/current_match_df_{}.csv'.format(DATE)
	current_df.to_csv(current_file_path, index=False)
	print '{} constructed '.format(current_file_path)

	match_file_path = 'match_data_constructed/match_df_{}.csv'.format(DATE)
	match_df.to_csv(match_file_path, index=False)
	print '{} constructed'.format(match_file_path)


	# see connect_df() for switching perspective
	# need to paste equations for probs, kls percentages


	# # depending on ONLY_PBP, this will have point-by-point matches, or all
	# # tour-level matches from START_DATE to present
	# df = generate_stats(df,CURRENT_YEAR)
	# df = finalize_df(df)
	# name = 'elo_atp_matches_all_'+DATE
	# print name + '.csv saved to my_data'
	# df.to_csv('match_data/'+name+'.csv')
