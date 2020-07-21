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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import confusion_matrix
def merge(list1, list2): # https://www.geeksforgeeks.org/python-merge-two-lists-into-list-of-tuples/ 
	merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
	return merged_list 
#comment start

# #----------------------------MAIN--------------------------------------------------

#CLASSIFICATION
source_str = "./cleaned"

complete_frame = []
review_class = 1
#						-- Import Files --

for file in os.listdir(source_str):               #file traversal
    file_name = file.split(".")[0]
    print('Working on {}'.format(file))
    if file_name == "Audio":
        audio_data = pd.read_csv('{}/{}'.format(source_str,file),index_col=None)				#Load original dataframe
    elif file_name == "Graphics":
        graphics_data = pd.read_csv('{}/{}'.format(source_str,file),index_col=None)				#Load original dataframe
    elif file_name == "Gameplay":
        gameplay_data = pd.read_csv('{}/{}'.format(source_str,file),index_col=None)				#Load original dataframe
    else:            
        multi_data = pd.read_csv('{}/{}'.format(source_str,file),index_col=None)				#Load original dataframeport_file,ignore_index=True)	
#						------------------

#						-- Drop Invalid Rows --
audio_data      = audio_data.drop(audio_data[audio_data['isValid']==0].index)
graphics_data   = graphics_data.drop(graphics_data[graphics_data['isValid']==0].index)
gameplay_data   = gameplay_data.drop(gameplay_data[gameplay_data['isValid']==0].index)
multi_data      = multi_data.drop(multi_data[(multi_data.is_audio== -2) & (multi_data.is_graphics == -2) & (multi_data.is_gameplay == -2)].index)

def classification_list(mrow):
    output_list = []
    if mrow['is_audio'] == 1:
        output_list.append('audio')
    if mrow['is_graphics'] == 1:
        output_list.append('graphics')
    if mrow['is_gameplay'] == 1:
        output_list.append('gameplay')
    if len(output_list) == 0:
        print(mrow)
    return output_list
def polarity_list(mrow):
    output_list = [-2,-2,-2]
    # print(mrow)
    if mrow['is_audio'] == 1:
        output_list[0] = mrow['audio_polarity']
    if mrow['is_gameplay'] == 1:
        output_list[1] = mrow['gameplay_polarity']
    if mrow['is_graphics'] == 1:
        output_list[2] = mrow['graphics_polarity']
    # print(output_list)
    return output_list
# #OLD
# input_columns = ['gameId','AccountName','review','classifications']
# input_compilation = pd.DataFrame([],columns=input_columns)
# input_compilation = input_compilation.append(pd.DataFrame({"gameId":audio_data['gameId'],
#                                         'AccountName':audio_data['AccountName'],
#                                         'review':audio_data['review'],
#                                         'classifications': [['audio'] for i in range(0,audio_data.shape[0])]
#                                         }))
# input_compilation = input_compilation.append(pd.DataFrame({"gameId":graphics_data['gameId'],
#                                         'AccountName':graphics_data['AccountName'],
#                                         'review':graphics_data['review'],
#                                         'classifications': [['graphics'] for i in range(0,graphics_data.shape[0])]
#                                         }))
# input_compilation = input_compilation.append(pd.DataFrame({"gameId":gameplay_data['gameId'],
#                                         'AccountName':gameplay_data['AccountName'],
#                                         'review':gameplay_data['review'],
#                                         'classifications': [['gameplay'] for i in range(0,gameplay_data.shape[0])]
#                                         }))


# md_classification_list = []
# for i in range(0,multi_data.shape[0]):
#     md_classification_list.append(classification_list(multi_data.iloc[i]))


# input_compilation = input_compilation.append(pd.DataFrame({"gameId":multi_data['gameId'],
#                                         'AccountName':multi_data['AccountName'],
#                                         'review':multi_data['review'],
#                                         'classifications': md_classification_list
#                                         }))
# input_compilation = input_compilation.reset_index()
# input_compilation = input_compilation.drop(columns=['index'])

#------------------------------------------------------------------------------------------------------------------------------
#NEW
input_columns = ['gameId','AccountName','review','classifications','polarity']
input_compilation = pd.DataFrame([],columns=input_columns)
input_compilation = input_compilation.append(pd.DataFrame({"gameId":audio_data['gameId'],
                                        'AccountName':audio_data['AccountName'],
                                        'review':audio_data['review'],
                                        'classifications': [['audio'] for i in range(0,audio_data.shape[0])],
                                        'polarity':list([i,-2,-2] for i in audio_data["Polarity"])
                                        }))
input_compilation = input_compilation.append(pd.DataFrame({"gameId":graphics_data['gameId'],
                                        'AccountName':graphics_data['AccountName'],
                                        'review':graphics_data['review'],
                                        'classifications': [['graphics'] for i in range(0,graphics_data.shape[0])],
                                        'polarity':list([-2,-2,i] for i in graphics_data["Polarity"])
                                        }))
input_compilation = input_compilation.append(pd.DataFrame({"gameId":gameplay_data['gameId'],
                                        'AccountName':gameplay_data['AccountName'],
                                        'review':gameplay_data['review'],
                                        'classifications': [['gameplay'] for i in range(0,gameplay_data.shape[0])],
                                        'polarity':list([-2,i,-2] for i in gameplay_data["Polarity"])
                                        }))

# print(input_compilation.head(10))


md_classification_list = []
md_polarity_list = []
for i in range(0,multi_data.shape[0]):
    md_classification_list.append(classification_list(multi_data.iloc[i]))
    md_polarity_list.append(polarity_list(multi_data.iloc[i]))

input_compilation = input_compilation.append(pd.DataFrame({"gameId":multi_data['gameId'],
                                        'AccountName':multi_data['AccountName'],
                                        'review':multi_data['review'],
                                        'classifications': md_classification_list,
                                        'polarity':md_polarity_list
                                        }))
input_compilation = input_compilation.reset_index()
input_compilation = input_compilation.drop(columns=['index'])

# print(input_compilation.tail(10))

#------------------------------------------------------------------------------------------------------------------------------
# print(input_compilation.classifications)


# for i,j in zip(multi_data['is_audio'],multi_data['audio_polarity']): print (i,j)



multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(input_compilation.classifications)
review_output = multilabel_binarizer.transform(input_compilation.classifications)
review_output = list([i,j] for i,j in zip(review_output,input_compilation['polarity']))
# print(multilabel_binarizer.classes_)
# review_output = input_compilation.classifications


review_list = input_compilation['review'].tolist()


# #							stem data using PorterStemmer
tokenizer_words = TweetTokenizer()
stemmer = PorterStemmer()
timer = 0
stemmed_review_list = []
print("No Stemming")
for sentence in review_list:

	stemmed_review_list.append(sentence.lower())          #For no stemming comment everything above 

	#-------------------------------------------------------- Stemming only
	# tokenized_sentence = tokenizer_words.tokenize(sentence.lower()) 
	# stemmed_sentence_list = [stemmer.stem(i) for i in tokenized_sentence]
	# stemmed_sentence = " ".join(stemmed_sentence_list)
	# stemmed_review_list.append(stemmed_sentence)
	#-------------------------------------------------------- 

	#-------------------------------------------------------- Stemming + POS Tagging
	# tokenized_sentence = tokenizer_words.tokenize(sentence.lower()) 
	# stemmed_sentence_list = nltk.pos_tag(tokenized_sentence)
	# stemmed_pos_sentence_list = []
	# for i, j  in stemmed_sentence_list:
	# 	stemmed_pos_sentence_list.append("{}_{}".format(i,j))
	# # # print(stemmed_sentence_list)
	# stemmed_sentence = " ".join(stemmed_pos_sentence_list)
	# stemmed_review_list.append(stemmed_sentence)
	# #-------------------------------------------------------- 


						# --------Training - Test Split------------

data_train, data_test, label_train, label_test = train_test_split(stemmed_review_list, review_output, test_size = 0.3, random_state = 7)
# # print(len(data_train))
# # print(len(data_test))
# #					--------------------------------------------
# for i,j in zip(data_train,label_train):
#     print(j[0])
#     print(i)

# #						------------Vectorization----------------
category_tf = TfidfVectorizer(ngram_range=(1,1))
category_tf.fit(data_train)

data_train_tf	= category_tf.transform(data_train)
data_test_tf	= category_tf.transform(data_test)


# print(len(data_train))
# print(len(data_test))
# # print(tf.get_feature_names())
# #						----------------------------------------------
# #					------------ Model---------------------

# #					----------------Category Classification------------------------
#TEMP
# print(label_train.tolist())
# category_lsvc = CalibratedClassifierCV(LinearSVC(multi_class='ovr'))
# category_lsvc = MultinomialNB()
# print(data_train_tf.shape)
# print(label_train.shape)
#

category_lsvc = OneVsRestClassifier(CalibratedClassifierCV(LinearSVC()))
category_lsvc.fit(data_train_tf,list(i[0] for i in label_train))
category_prediction = category_lsvc.predict(data_test_tf)
category_probability = category_lsvc.predict_proba(data_test_tf)


#Setting Threshold





def is_over_threshold(threshold,input_list):
    label_types = np.array([0,0,0])
    if input_list[0] > threshold:
        label_types[0] = 1      
    if input_list[1] > threshold:
        label_types[1] = 1
    if input_list[2] > threshold:    
       label_types[2] = 1
    # if len(label_types) == 0:
    #     label_types.append(0)
    return label_types

# print(np.sum(test_list[0]))
# print(len(test_list[0]))



def compute_sub_accuracy(label,output):
    test_list = np.hsplit(label,3)
    output_list = np.hsplit(np.array(output),3)

    # print(output_list)
    # label_1_total = np.sum(test_list[0])
    # label_2_total = np.sum(test_list[1])
    # label_3_total = np.sum(test_list[2])

    for i in range(0,3):
        x_list = test_list[i]
        y_list = output_list[i]
        tn, fp, fn, tp = confusion_matrix(x_list,y_list).ravel()
        accuracy  = (tp + tn)/ (tp+tn+fp+fn)
        precision = (tp) / (tp + fp)
        recall    = (tp) / (tp + fn)        
        # print("Label {}".format(i+1))
        if (i+1) == 1: 
            print("Audio Accuracy: {} Precision: {}".format(round(accuracy,4),round(precision,4)))
        elif (i+1) == 2:
            print("Gameplay Accuracy: {} Precision: {}".format(round(accuracy,4),round(precision,4)))
        elif (i+1) == 3:
            print("Graphics Accuracy: {} Precision: {}".format(round(accuracy,4),round(precision,4)))
        # print("True Positive: {}".format(tp))
        # print("True Negative: {}".format(tn))
        # print("False Positive: {}".format(fp))    
        # print("False Negative: {}".format(fn))
    #     print("Total: {}".format(tp + tn + fp + fn))
    # print(label_1_total)
    # print(label_2_total)
    # print(label_3_total)
    # print(len(label))

def print_testing():
    category_label_test = np.array(list(i[0] for i in label_test))
    threshold_list = [.3,.4,.5,.6,.7]
    for temp_treshold in threshold_list:
        output_labels = []
        for i in category_probability: 
            output_labels.append(is_over_threshold(temp_treshold,i))
        # for x,y in zip(category_label_test,output_labels): print(x,y)
        
        # print("-------------------------")
        print("Treshold: {}".format(temp_treshold))
        print("Accuracy: {}".format(round(accuracy_score(category_label_test,output_labels),4)))
        print("Hamming Loss: {}".format(round(hamming_loss(category_label_test,output_labels),4)))
        compute_sub_accuracy(category_label_test,output_labels)

# print_testing()
var_threshold = .3

# data_test
category_label_test = np.array(list(i[0] for i in label_test))
pol_label_test = np.array(list(i[0] for i in label_test))
output_labels = []
for i in category_probability:
    output_labels.append(is_over_threshold(var_threshold,i))

test_audio_passed_text = []
test_audio_passed_pol = []

test_gameplay_passed_text = []
test_gameplay_passed_pol = []

test_graphics_passed_text = [] 
test_graphics_passed_pol = [] 
# for i in pol_label_test:
#     print(i)
# print("lol")
for i,j,k,l in zip(data_test,category_label_test,output_labels,pol_label_test):
    if(j[0] == k[0] == 1):
        test_audio_passed_text.append(i)
        test_audio_passed_pol.append(l[0])
    if(j[1] == k[1] == 1):
        test_gameplay_passed_text.append(i)
        test_gameplay_passed_pol.append(l[1])
    if(j[2] == k[2] == 1):
        test_graphics_passed_text.append(i)
        test_graphics_passed_pol.append(l[2])


##POLARITY TEST
        # if from source
# audio_data_2 = input_compilation[input_compilation.classifications.apply(lambda x: 'audio' in x)]
# graphics_data_2 = input_compilation[input_compilation.classifications.apply(lambda x: 'graphics' in x)]
# gameplay_data_2 = input_compilation[input_compilation.classifications.apply(lambda x: 'gameplay' in x)]


# #						------------Import Training Data----------------
audio_data_2_text = []
audio_data_2_polarity = []

graphics_data_2_text = []
graphics_data_2_polarity = []

gameplay_data_2_text = []
gameplay_data_2_polarity = []

for i,j in zip(data_train,label_train):

    temp_classifications = j[0]
    temp_polarity = j[1]

    if temp_classifications[0] == 1:
        audio_data_2_text.append(i)
        audio_data_2_polarity.append(temp_polarity[0])
    if temp_classifications[1] == 1:
        gameplay_data_2_text.append(i)
        gameplay_data_2_polarity.append(temp_polarity[1])
    if temp_classifications[2] == 1:
        graphics_data_2_text.append(i)
        graphics_data_2_polarity.append(temp_polarity[2])

# for i in graphics_data_2_polarity:
#     print(i)

#Printer
# def print_pair(temp_list,polarity):
#     for i,j in zip(temp_list,polarity):
#         print(j)
#         print(i)
#     print()
# print_pair(audio_data_2_text,audio_data_2_polarity)
# print_pair(gameplay_data_2_text,gameplay_data_2_polarity)
# print_pair(graphics_data_2_text,graphics_data_2_polarity)
# #					------------------

# #						------------Import Successful from category classification----------------


# #						------------Vectorization----------------

#Audio

pol_audio_tf = TfidfVectorizer(ngram_range=(1,1))
pol_audio_tf.fit(audio_data_2_text)
pol_audio_test_tf = pol_audio_tf.transform(audio_data_2_text)

#Gameplay
pol_gameplay_tf = TfidfVectorizer(ngram_range=(1,1))
pol_gameplay_tf.fit(gameplay_data_2_text)
pol_gameplay_test_tf = pol_gameplay_tf.transform(gameplay_data_2_text)

#Graphics
pol_graphics_tf = TfidfVectorizer(ngram_range=(1,1))
pol_graphics_tf.fit(graphics_data_2_text)
pol_graphics_test_tf = pol_graphics_tf.transform(graphics_data_2_text)

def count_output(temp_list):
    temp_count_pos = 0
    temp_count_neu = 0
    temp_count_neg = 0
    for i in temp_list:
        if i == 1:
            temp_count_pos = temp_count_pos + 1
        elif i == 0:
            temp_count_neu = temp_count_neu + 1
        elif i == -1:
            temp_count_neg = temp_count_neg + 1

    print("Positive: {}"    .format(temp_count_pos))
    print("Neutral: {}"     .format(temp_count_neu))
    print("Negative: {}"    .format(temp_count_neg))
    print()



# #					----------------Polarity Model------------------------
def pol_output(pred_output,label):
    category_output_dict = {}
    for i,j in zip(label,pred_output):
        temp_text = "{}_{}".format(i,j)
        if temp_text in category_output_dict:
            category_output_dict[temp_text] += 1
        else:
            category_output_dict[temp_text] = 1

    for key, value in sorted(category_output_dict.items()):
        print("{} {}".format(key,value))
        

# for i in test_audio_passed_pol:
#     print(i)

#Audio
pol_audio_lsvc = CalibratedClassifierCV(LinearSVC(multi_class='ovr'))
pol_audio_lsvc.fit(pol_audio_test_tf,audio_data_2_polarity)
print("Audio Polarity")
print(pol_audio_lsvc.score(pol_audio_tf.transform(test_audio_passed_text),test_audio_passed_pol))
pol_audio_output = pol_audio_lsvc.predict(pol_audio_tf.transform(test_audio_passed_text))
# pol_output(pol_audio_output,test_audio_passed_pol)
#Gameplay
pol_gameplay_lsvc = CalibratedClassifierCV(LinearSVC(multi_class='ovr'))
pol_gameplay_lsvc.fit(pol_gameplay_test_tf,gameplay_data_2_polarity)
print("Gameplay Polarity")
print(pol_gameplay_lsvc.score(pol_gameplay_tf.transform(test_gameplay_passed_text),test_gameplay_passed_pol))
pol_gameplay_output = pol_gameplay_lsvc.predict(pol_gameplay_tf.transform(test_gameplay_passed_text))
# pol_output(pol_gameplay_output,test_gameplay_passed_pol)

#Graphics
pol_graphics_lsvc = CalibratedClassifierCV(LinearSVC(multi_class='ovr'))
pol_graphics_lsvc.fit(pol_graphics_test_tf,graphics_data_2_polarity)
print("Graphics Polarity")
print(pol_graphics_lsvc.score(pol_graphics_tf.transform(test_graphics_passed_text),test_graphics_passed_pol))
pol_graphics_output = pol_graphics_lsvc.predict(pol_graphics_tf.transform(test_graphics_passed_text))
# pol_output(pol_graphics_output,test_graphics_passed_pol)



print("TRAIN")
train_list = np.hsplit(np.array(list(i[1] for i in label_train)),3)
count_output(train_list[0])
count_output(train_list[1])
count_output(train_list[2])
print()
print("TEST")
output_list = np.hsplit(np.array(list(i[1] for i in label_test)),3)
count_output(output_list[0])
count_output(output_list[1])
count_output(output_list[2])

# count_output(test_audio_passed_pol)
# print()
# count_output(test_gameplay_passed_pol)
# print()
# count_output(test_graphics_passed_pol)
# #					-------------------------------------------------
# # input_pipline = InputPipeline.InputPipeline()
# # input_str = "You feel an evil presence watching you... This is the sort of game that you boot up thinking you'll spend an hour or two on, only to one day wake up face down on your keyboard surrounded by bottles and plates wondering where the past 3 weeks went. 100 hours in and I've still not defeated the current 'final' boss, something that's a mixture of the games' excellent difficulty curve, and a melancholy realisation that once I defeat him, my time with the game will naturally come to an end. Primarily, Terraria is a sandbox game. You appear in a new world with some basic equipment and no real instruction, eventually you will build a small house to survive the monsters that surface during the night, discover some form of corruption eating away at the world, encounter new NPCs and face off against powerful Boss enemies. Despite all this, you are given little direction, these are merely facets of a larger game that allows players to do as they wish, encountering all the world has to offer at mostly their own pace. Part of the great design behind the title is in the difficulty curve I mentioned earlier, meaning challenges pitted against a player start easy and steadily increase at a rate easy to handle. Although traditional RPG elements are largely absent, a discrete levelling system is present in the form of Bosses that have been defeated, meaning if you don't progress through these enemies then the gameplay doesn't become more difficult. In fact, the first boss doesn't spawn until you're suitably equipped with armour and health. My hours in the game are split roughly equally between single and multiplayer. Single player stands up on its own perfectly fine, but in my opinion the game is much more rewarding gathering some friends and working together to tackle the bosses, as the feeling of reaching new goals is shared, and the ammount of 'grinding' for materials is spread across multiple people. It also helps to appreciate the brilliant soundtrack, as if you voice chat with your group then spontaneous humming along loudly is to be expected. Enemy design is exciting and unique, with floating eyes, possessed suits of armour, giant robotic worms, and a ninja suspended in a giant ball of slime all making appearances. That being said, whoever on the design team is responsible for 'hellbats' has earned a special sort of hatred from myself...All in all Terraria is an amazing experience from beginning to end, its tone is humourous at times whist still provoking a feeling of wonderment and mystery in its exploration, and fearful excitement at its combat. It's a game I will come back to time and time again looking to recapture the memories I've made, and to forge new ones. It is a rare game in that with no shred of a doubt, these hours I have spent were not wasted. I heartily recommend that you purchase this game.  Fortune and glory kid..."


# # while (True):
# # 	if input_str == "exit":
# # 		break
# # 	# input_str = str(input("Enter Example:"))
# # 	input_list = 	input_pipline.pipeline(input_str)
# # 	for sentence in input_list:
# # 		temp_sentence_tf = tf.transform([sentence])
# # 		print("~~~")
# # 		print("Sentence: {}\nCategory: {} ".format(sentence,polarity_lsvc.predict_proba(temp_sentence_tf)))
# # 		print("~~~")
# # 	break
