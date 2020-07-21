from collections import Counter
import pandas as pd
import csv
import re
import os


def cleanReview(old_review):
	old_review = str(old_review)
	new_review = re.sub(r'[^a-zA-Z\s]',' ', old_review) #remove uncommon symbols
	return new_review





cnt = Counter()
source_str = './cleaned'
destination_str = ''

for file in os.listdir(source_str):               #file traversal
	print('Working on {}'.format(file))
	data = pd.read_csv('{}/{}'.format(source_str,file))
	freq='1'
	height = data.shape[0]
	for row in range(0,height):
		copying_row = data.iloc[row].copy(deep=True)				#Copy First Row		
		review = copying_row['review']
		# review = cleanReview(review)
		review = review.split()
		for item in review:
			cnt[item] += 1
			print("{} {}".format(item,cnt[item]))
print('Writing to processed csv summary')


outfile = 'summary.csv'
with open(outfile, encoding='utf-8-sig', mode='w') as fp:
	fp.write('word,freq\n')
	for key, value in  sorted(cnt.items(), key=lambda pair: pair[1], reverse=True):
		fp.write('{},{}\n'.format(key, value))  