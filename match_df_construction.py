import os
import sys
from globals import *
from helper_functions import *
from data_functions import *

sys.path.insert(0, '{}/sackmann'.format(os.getcwd()))

# only need to run once, make sure to sort all matches in concat_data()
def format_match_data():
	return

def write_dataframe(df, prefix):
	file_path = 'match_data_constructed/{}_df_{}.csv'.format(prefix, DATE)
	df.to_csv(file_path, index=False)
	print '{} constructed'.format(file_path)

if __name__=='__main__':
	print 'main: ', os.getcwd()
	current_year = int(DATE.split('_')[-1])
	current_df, match_df = generate_dfs(TOUR, START_YEAR, current_year, RET_STRINGS, ABD_STRINGS, COUNTS_538)

	write_dataframe(current_df, 'current_match')

	write_dataframe(match_df, 'match')
