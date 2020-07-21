import pandas as pd
import csv
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, sent_tokenize


stopwords = set(stopwords.words('english'))
tokenizer_words = TweetTokenizer()

#----------------------------FUNCTIONS--------------------------------------------------
def clean_review(old_review):
	old_review = str(old_review)

	new_review = re.sub(r'\[.*?\]', '', old_review)		#remove [h1], [table], etc
	new_review = re.sub(r'http\S+', '', new_review)		#removes hyperlinks

	#Custom Ratings
	new_review = re.sub(r'☐\s?\S*','',new_review)		#removes lines with box
	new_review = re.sub(r'☑','good ',new_review)

	#Convert scores
	#  0/10 - 4/10 	= bad score
	#  5/10 		= neutral score
	#  6-10 - 11/10 = good score
	new_review = re.sub(r'0/5','(bad score) ',new_review)
	new_review = re.sub(r'1/5','(bad score) ',new_review)
	new_review = re.sub(r'2/5','(bad score) ',new_review)
	new_review = re.sub(r'3/5','(neutral score) ',new_review)
	new_review = re.sub(r'4/5','(good score) ',new_review)
	new_review = re.sub(r'5/5','(good score) ',new_review)

	new_review = re.sub(r'0/10','(bad score) ',new_review)
	new_review = re.sub(r'1/10','(bad score) ',new_review)
	new_review = re.sub(r'2/10','(bad score) ',new_review)
	new_review = re.sub(r'3/10','(bad score) ',new_review)
	new_review = re.sub(r'4/10','(bad score) ',new_review)
	new_review = re.sub(r'5/10','(neutral score) ',new_review)
	new_review = re.sub(r'6/10','(good score) ',new_review)
	new_review = re.sub(r'7/10','(good score) ',new_review)
	new_review = re.sub(r'8/10','(good score) ',new_review)
	new_review = re.sub(r'9/10','(good score) ',new_review)
	new_review = re.sub(r'10/10','(good score) ',new_review)
	new_review = re.sub(r'11/10','(good score) ',new_review)


	#remove uncommon symbols				# add ' to exceptions 
	new_review = re.sub(r'[^a-zA-Z0-9\:\.\,\(\)\'\-\+\/\>\<\&\%\!\=\_\*\s]',' ', new_review) 

	#insert spaces on symbols
	new_review = re.sub(r'\-'," - ",new_review)
	new_review = re.sub(r'\+'," + ",new_review)
	new_review = re.sub(r'\>'," > ",new_review)
	new_review = re.sub(r'\<'," < ",new_review)
	new_review = re.sub(r'\&'," & ",new_review)
	new_review = re.sub(r'\%'," % ",new_review)
	new_review = re.sub(r'\!'," ! ",new_review)
	new_review = re.sub(r'\='," = ",new_review)
	new_review = re.sub(r'\_'," _ ",new_review)
	new_review = re.sub(r'\*'," * ",new_review)	
	new_review = re.sub(r'\.',' . ',new_review)
	new_review = re.sub(r'\,',' , ',new_review)
	new_review = re.sub(r'\(',' ( ',new_review)
	new_review = re.sub(r'\)',' ) ',new_review)
	new_review = re.sub(r'\|',' | ',new_review)
	new_review = re.sub(r'\:',' : ',new_review)
	new_review = re.sub(r'\\',' \ ',new_review)

	new_review = re.sub(r'Product received for free',' ',new_review)	#Remove Product Recieved for Free
	new_review = re.sub(r'Early Access Review',' ',new_review)	#Remove Early Access Review
	return new_review


def text_filter(text):
	if (len(text.strip()) > 1):
		return 1
	else:
		return 0

def split_pros_cons(review):
	review = re.sub(r'Pros :'," {split_HERE} pros : ",review,flags=re.I)
	review = re.sub(r'Cons :'," {split_HERE} cons : ",review,flags=re.I)
	output = review.split("{split_HERE}")
	return output

def tokenize_review(review):	
	return [tokenizer_words.tokenize(t) for t in nltk.sent_tokenize(review)]