import os
import sys
from globals import *
from data_functions import *

sys.path.insert(0, '{}/sackmann'.format(os.path.dirname(os.path.abspath(__file__))))

# only need to run once, make sure to sort all matches in concat_data()
def format_match_data():
	print 'formatting match data...'
	for file_name in os.listdir('match_data'):
		df = pd.read_csv('match_data/{}'.format(file_name))
		formatted_df = format_match_df(df, TOUR, RET_STRINGS, ABD_STRINGS)
		formatted_df.to_csv('match_data_formatted/{}'.format(file_name), index=False)
	return

def write_dataframe(df, prefix):
	file_path = 'match_data_constructed/{}_df_{}.csv'.format(prefix, DATE)
	df.to_csv(file_path, index=False)
	print '{} constructed'.format(file_path)

if __name__=='__main__':
	format_match_data()

	match_df = generate_df(TOUR, START_YEAR, CURRENT_YEAR, RET_STRINGS, ABD_STRINGS, COUNTS_538)

	# write_dataframe(current_df, 'current_match')

	write_dataframe(match_df, 'match')
