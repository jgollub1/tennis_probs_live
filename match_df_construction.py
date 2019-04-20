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


	# TODO: add point-by-point dataset
