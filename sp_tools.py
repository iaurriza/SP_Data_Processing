
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import confusion_matrix

def print_unique_instances(temp_input):
    temp_list = list(str(i) for i in temp_input)
    cnt = Counter(temp_list)
    for i in cnt.most_common(300):
        print(i)
        
def classification_list(mrow):
    output_list = [0,0,0]
    if mrow['is_audio'] == 1:
        output_list[0] = 1
    if mrow['is_gameplay'] == 1:
        output_list[1] = 1
    if mrow['is_graphics'] == 1:
        output_list[2] = 1

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
    return output_list

def polarity_counter(temp_arr):
    temp_arr = temp_arr.tolist()
    temp_au_pos = 0
    temp_au_neu = 0
    temp_au_neg = 0

    temp_ga_pos = 0
    temp_ga_neu = 0
    temp_ga_neg = 0

    temp_gr_pos = 0 
    temp_gr_neu = 0 
    temp_gr_neg = 0
    
    for temp_row in temp_arr:
        #remove excess string
        temp_row = temp_row.replace("[","")
        temp_row = temp_row.replace("]","")
        temp_row = temp_row.split(",")
        i = int(temp_row[0])
        j = int(temp_row[1])
        k = int(temp_row[2])
        
        if i == 1:
            temp_au_pos += 1
        if i == 0:
            temp_au_neu += 1
        if i == -1:
            temp_au_neg += 1

        if j == 1:
            temp_ga_pos += 1
        if j == 0:
            temp_ga_neu += 1
        if j == -1:
            temp_ga_neg += 1

        if k == 1:
            temp_gr_pos += 1
        if k == 0:
            temp_gr_neu += 1
        if k == -1:
            temp_gr_neg += 1

    print("AUDIO: {} {} {}".format(temp_au_pos,temp_au_neu,temp_au_neg))
    print("GAMEPLAY: {} {} {}".format(temp_ga_pos,temp_ga_neu,temp_ga_neg))
    print("GRAPHICS: {} {} {}".format(temp_gr_pos,temp_gr_neu,temp_gr_neg))
#Functions for cell below
def is_over_threshold(threshold,input_list):
    label_types = np.array([0,0,0])
    if input_list[0] > threshold:
        label_types[0] = 1      
    if input_list[1] > threshold:
        label_types[1] = 1
    if input_list[2] > threshold:    
       label_types[2] = 1
    return label_types
def compute_sub_accuracy(label,output):
    test_list = np.hsplit(label,3)
    output_list = np.hsplit(np.array(output),3)
    print()
    print("Sub Accuracy")
    for i in range(0,3):
        x_list = test_list[i]
        y_list = output_list[i]
        tn, fp, fn, tp = confusion_matrix(x_list,y_list).ravel()
        accuracy  = (tp + tn)/ (tp+tn+fp+fn)
        precision = (tp) / (tp + fp)
        recall    = (tp) / (tp + fn)        
        if (i+1) == 1: 
            print("Audio    \t Accuracy: {} \tPrecision: {} \t Recall: {}".format(round(accuracy,4),round(precision,4),round(recall,4)))
        elif (i+1) == 2:
            print("Gameplay \t Accuracy: {} \tPrecision: {} \t Recall: {}".format(round(accuracy,4),round(precision,4),round(recall,4)))
        elif (i+1) == 3:
            print("Graphics \t Accuracy: {} \tPrecision: {} \t Recall: {}".format(round(accuracy,4),round(precision,4),round(recall,4)))


def print_testing(label_test,category_probability):
    category_label_test = label_test
    threshold_list = [.3,.4,.5,.6,.7]
    for temp_treshold in threshold_list:
        output_labels = []
        for i in category_probability: 
            output_labels.append(is_over_threshold(temp_treshold,i))
        
        # print("-------------------------")
        print("Treshold: \t{}".format(temp_treshold))
        print("Accuracy: \t{}".format(round(accuracy_score(category_label_test,output_labels),4)))
        print("Precision:\t {}".format(precision_score(category_label_test,output_labels,average="micro"),4))
        ## micro = global_tp/(global_tp+global_fp)
        ## macro = ave(audio_tp/(audio_tp/audio_fp)+ ... +...)
        print("Hamming Loss:\t {}".format(round(hamming_loss(category_label_test,output_labels),4)))
        compute_sub_accuracy(category_label_test,output_labels)
        print()
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
def print_len(*args):
    for temp_array in args: print(len(temp_array))