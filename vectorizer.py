import nltk
import os
import pandas as pd
import InputPipeline
import numpy as np

from nltk.stem import PorterStemmer 
from nltk.tokenize import TweetTokenizer, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import confusion_matrix


tokenizer_words = TweetTokenizer()
stemmer = PorterStemmer()

def merge(list1, list2): # https://www.geeksforgeeks.org/python-merge-two-lists-into-list-of-tuples/ 
	merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
	return merged_list 
#comment start

# #----------------------------MAIN--------------------------------------------------

#CLASSIFICATION
source_str = "./cleaned_sorted"

complete_frame = []
review_class = 1
#						-- Import Files --
for file in os.listdir(source_str):               #file traversal
	print('Working on {}'.format(file))
	
	if(review_class == 1):
		import_file = pd.read_csv('{}/{}'.format(source_str,file))
		import_file['Class'] = review_class
		complete_frame = import_file				#Load original dataframe
		review_class = review_class + 1 
	else:
		import_file = pd.read_csv('{}/{}'.format(source_str,file))
		import_file['Class'] = review_class
		complete_frame = complete_frame.append(import_file,ignore_index=True)	
		review_class = review_class + 1



review_list = complete_frame['review'].tolist()
review_output = merge(complete_frame['Class'].tolist(),complete_frame['Polarity'].tolist())


# #							stem data using PorterStemmer
# timer = 0
stemmed_review_list = []
for sentence in review_list:
	# tokenized_sentence = tokenizer_words.tokenize(sentence.lower())
	stemmed_review_list.append(sentence.lower())
# 	# print("-----")
# 	# print(tokenized_sentence)

# 	#-------------------------------------------------------- Stemming only
	# tokenized_sentence = tokenizer_words.tokenize(sentence.lower())
# 	# stemmed_sentence_list = [stemmer.stem(i) for i in tokenized_sentence]
# 	# stemmed_sentence = " ".join(stemmed_sentence_list)
# 	#-------------------------------------------------------- 
# 	# nltk.pos_tag(tokenized_sentence)
# 	#-------------------------------------------------------- Stemming + POS Tagging

	# tokenized_sentence = tokenizer_words.tokenize(sentence.lower())
# 	stemmed_sentence_list = nltk.pos_tag(tokenized_sentence)
# 	stemmed_pos_sentence_list = []
# 	for i, j  in stemmed_sentence_list:
# 		stemmed_pos_sentence_list.append("{}_{}".format(i,j))
# 	# print(stemmed_sentence_list)
# 	stemmed_sentence = " ".join(stemmed_pos_sentence_list)
# 	#-------------------------------------------------------- 

# 	# print(stemmed_sentence_list)
# 	# print(stemmed_sentence)
# 	# print("-----")

# 	# print(stemmed_sentence)
# 	stemmed_review_list.append(stemmed_sentence)

# # for i,j in zip(review_list,stemmed_review_list):
# 	# print("---\n {} \n {} \n---".format(i,j))
# #					--------------------------------------------
#						--------Training - Test Split------------




data_train, data_test, label_train, label_test = train_test_split(stemmed_review_list, review_output, test_size = 0.3, random_state = 7)
# print(len(data_train))
# print(len(data_test))
#					--------------------------------------------


#						------------Vectorization----------------
tf = TfidfVectorizer(ngram_range=(1,1))
tf.fit(data_train)

data_train_tf	= tf.transform(data_train)
data_test_tf	= tf.transform(data_test)


# print(len(data_train))
# print(len(data_test))
# print(tf.get_feature_names())
#						----------------------------------------------
#					------------ Model---------------------


#					----------------Category Classification------------------------
category_lsvc = CalibratedClassifierCV(LinearSVC(multi_class='ovr'))
category_lsvc.fit(data_train_tf,[i[0] for i in label_train])
category_prediction_output = category_lsvc.predict(data_test_tf)
# category_probability_output = category_lsvc.predict_proba(data_test_tf)
# class1_probability	= [i[0] for i in category_probability_output]
# class2_probability	= [i[1] for i in category_probability_output]
# class3_probability	= [i[2] for i in category_probability_output]


print(category_lsvc.score(data_test_tf,[i[0] for i in label_test]))

# for i in range(0,100):
	# print("{} {}".format(category_lsvc.predict_proba(data_test_tf[i]),category_lsvc.predict(data_test_tf[i])))


#					---OUTPUT---
# data = {"test_review":data_test,"test_label":[i[0] for i in label_test],"output":category_prediction_output,"prob1":class1_probability,"prob2":class2_probability,"prob3":class3_probability,"polarity":[i[1] for i in label_test]}
# df = pd.DataFrame(data)
# df.to_csv("output.csv",index=False)
#					------------
#					---List Outputs---
category_output_dict = {}
for i,j in zip(label_test,category_prediction_output):
	temp_text = "{}_{}".format(i[0],j)
	if temp_text in category_output_dict:
		category_output_dict[temp_text] += 1
	else:
		category_output_dict[temp_text] = 1

for key, value in sorted(category_output_dict.items()):
	print("{} {}".format(key,value))
#					------------------




# polarity_test_df = pd.DataFrame(data = {'review':data_test,
# 												'category_label':[i[0] for i in label_test],
# 												'category_prediction':category_prediction_output,
# 												'polarity_label':[i[1] for i in label_test]
# 												}
# 										)
# print(polarity_test_df.head(10))
# print("Before Cut: {}".format(polarity_test_df.shape))
# polarity_test_df = polarity_test_df[polarity_test_df['category_label'] == polarity_test_df['category_prediction']]
# print("After Cut: {}".format(polarity_test_df.shape))
# print(polarity_test_df.head(10))



# tf_transform = tf.transform(polarity_test_df['review'])

#					----------------Polarity Classification------------------------
# for i in label_train:
#     print(i[1])
# print("Polarity Test Start")
# polarity_lsvc = CalibratedClassifierCV(LinearSVC(multi_class='ovr'))
# polarity_lsvc.fit(data_train_tf,[i[1] for i in label_train])
# polarity_prediction_output = polarity_lsvc.predict(tf_transform)
# polarity_probability_output = polarity_lsvc.predict_proba(polarity_test_df['review_tf'])

# print(polarity_lsvc.score(tf_transform,polarity_test_df['polarity_label']))


#					-------------------------------------------------
# input_pipline = InputPipeline.InputPipeline()
# input_str = "You feel an evil presence watching you... This is the sort of game that you boot up thinking you'll spend an hour or two on, only to one day wake up face down on your keyboard surrounded by bottles and plates wondering where the past 3 weeks went. 100 hours in and I've still not defeated the current 'final' boss, something that's a mixture of the games' excellent difficulty curve, and a melancholy realisation that once I defeat him, my time with the game will naturally come to an end. Primarily, Terraria is a sandbox game. You appear in a new world with some basic equipment and no real instruction, eventually you will build a small house to survive the monsters that surface during the night, discover some form of corruption eating away at the world, encounter new NPCs and face off against powerful Boss enemies. Despite all this, you are given little direction, these are merely facets of a larger game that allows players to do as they wish, encountering all the world has to offer at mostly their own pace. Part of the great design behind the title is in the difficulty curve I mentioned earlier, meaning challenges pitted against a player start easy and steadily increase at a rate easy to handle. Although traditional RPG elements are largely absent, a discrete levelling system is present in the form of Bosses that have been defeated, meaning if you don't progress through these enemies then the gameplay doesn't become more difficult. In fact, the first boss doesn't spawn until you're suitably equipped with armour and health. My hours in the game are split roughly equally between single and multiplayer. Single player stands up on its own perfectly fine, but in my opinion the game is much more rewarding gathering some friends and working together to tackle the bosses, as the feeling of reaching new goals is shared, and the ammount of 'grinding' for materials is spread across multiple people. It also helps to appreciate the brilliant soundtrack, as if you voice chat with your group then spontaneous humming along loudly is to be expected. Enemy design is exciting and unique, with floating eyes, possessed suits of armour, giant robotic worms, and a ninja suspended in a giant ball of slime all making appearances. That being said, whoever on the design team is responsible for 'hellbats' has earned a special sort of hatred from myself...All in all Terraria is an amazing experience from beginning to end, its tone is humourous at times whist still provoking a feeling of wonderment and mystery in its exploration, and fearful excitement at its combat. It's a game I will come back to time and time again looking to recapture the memories I've made, and to forge new ones. It is a rare game in that with no shred of a doubt, these hours I have spent were not wasted. I heartily recommend that you purchase this game.  Fortune and glory kid..."


# while (True):
# 	if input_str == "exit":
# 		break
# 	# input_str = str(input("Enter Example:"))
# 	input_list = 	input_pipline.pipeline(input_str)
# 	for sentence in input_list:
# 		temp_sentence_tf = tf.transform([sentence])
# 		print("~~~")
# 		print("Sentence: {}\nCategory: {} ".format(sentence,polarity_lsvc.predict_proba(temp_sentence_tf)))
# 		print("~~~")
# 	break


