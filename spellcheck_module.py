


import pandas as pd
import csv




def check_dictionary(word,test_dictionary):
	new_word = word
	temp_df = test_dictionary[test_dictionary['word'] == word]
	if not temp_df.empty:
		new_word = "+~/ {} -~/ {} $~/".format(word,temp_df['suggestion1'].to_string(index=False,header=False))
	return new_word
