#Author: Ian Michael Urriza
#Function: To split whole paragraphs into single sentences. Assume that data is already stored in DataFrame.

import pandas as pd
import csv
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer

#--------------------------Initalization for NLTK--------------------------------------
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
stopwords = set(stopwords.words('english'))
tokenizer_words = TweetTokenizer()
#---------------------------Lists------------------------------------------------------
#https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he shall",
"he'll've": "he shall have",
"he's": "he has",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has",
"I'd": "I had",
"I'd've": "I would have",
"I'll": "I shall",
"I'll've": "I shall have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had",
"it'd've": "it would have",
"it'll": "it shall",
"it'll've": "it shall have",
"it's": "it has",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she shall",
"she'll've": "she shall have",
"she's": "she has",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that has",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there has",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they shall",
"they'll've": "they shall have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall",
"what'll've": "what shall have",
"what're": "what are",
"what's": "what has",
"what've": "what have",
"when's": "when has",
"when've": "when have",
"where'd": "where did",
"where's": "where has",
"where've": "where have",
"who'll": "who shall",
"who'll've": "who shall have",
"who's": "who has",
"who've": "who have",
"why's": "why has",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall",
"you'll've": "you shall have",
"you're": "you are",
"you've": "you have"
}

#----------------------------FUNCTIONS--------------------------------------------------
def clean_review(old_review):
	new_review = str(old_review)

	# new_review = re.sub(r'[^a-zA-Z0-9\:\.\,\(\)\+\/\*\s]',' ', new_review)
	new_review = new_review.lower()

	return new_review
def replace_contractions(token):
	str_token = str(token)
	if str_token in contractions:
		return contractions[str_token]
	else:
		return str_token
def split_review(cleaned_review):
	return re.split(r"\.|[\n]",str(cleaned_review))

def text_filter(text):
	if (len(text.strip()) > 1):
		return 1
	else:
		return 0
source_str = './need_cleaning'
#----------------------------MAIN--------------------------------------------------
print('Start Program')
if not os.path.exists(source_str):				#Check if Input Exists
	exit()


print('Start Loop')

for file in os.listdir(source_str):               #file traversal
	print('Working on {}'.format(file))
	data = pd.read_csv('{}/{}'.format(source_str,file),index_col=0)				#Load original dataframe

	new_data = pd.DataFrame(columns=data.columns)
	height = data.shape[0]

	for row in range(0,height):
		copying_row = data.iloc[row].copy(deep=True)				#Copy Row
		cleaned_review = clean_review(copying_row['review'])

		tokens_sentences = [tokenizer_words.tokenize(t) for t in nltk.sent_tokenize(cleaned_review)]

		for sentence_list in tokens_sentences:

			#sentence_list is a list of tokens
			new_sentence_list = []
			flag = 0
			for token in sentence_list:
				# print(token)										#do not add remove stopwords
				if "'" in token:				#detect contractions
					new_token = replace_contractions(token)
					# if new_token != token:
					# 	print(new_token)
				token = stemmer.stem(token)
				new_sentence_list.append(token)
			print("---\nOld: {} \nNew: {}\n---".format(' '.join(sentence_list), ' '.join(new_sentence_list)))
			# print(new_sentence_list)
			# sentence = ' '.join(new_sentence_list)
			# if flag == 1:
			# 	print(sentence)
			# 	flag = 0

			# print(sentence)
			# if (text_filter(sentence)):
			# 	copying_row['review'] = sentence
			# 	new_data = new_data.append(copying_row,ignore_index=True)		#Append to Another Table
