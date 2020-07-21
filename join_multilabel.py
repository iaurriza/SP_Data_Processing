#Author: Ian Michael Urriza
#Function: To split whole paragraphs into single sentences. Assume that data is already stored in DataFrame.


import pandas as pd
import csv
import re
import os
import nltk


source_str = './need_cleaning'
destination_str = './multilabel_list'
#----------------------------MAIN--------------------------------------------------
print('Start Program')
if not os.path.exists(source_str):				#Check if Input Exists
	exit()

if not os.path.exists(destination_str):			#Check if Output Exists
	os.mkdir(destination_str)

print('Start Loop')

data_columns = ["gameId","AccountName","Date","Hours_Played","isPositive","review","is_audio","audio_polarity","is_graphics","graphics_polarity","is_gameplay","gameplay_polarity"]
df_for_multi = pd.DataFrame(columns=data_columns)
print()
for file in os.listdir(source_str):               #file traversal
	print('Working on {}'.format(file))
	data = pd.read_csv('{}/{}'.format(source_str,file),header=0)				#Load original dataframe
	# print(data)
	print(data.shape)
	data = data.drop(data[data['isValid']==0].index)
	print(data.shape)
	data = data.drop(data[data['isSingle']==1].index)
	print(data.shape)
	data = data.reset_index(drop=True)
	print(data.shape)

	# print(data.head())

	height = data.shape[0]
	if file == "Audio.csv":
		print("Audio Size: {}".format(height))
		for row in range(0,height):
			copying_row = data.iloc[row].copy(deep=True)
			insert_row = [copying_row["gameId"],
							copying_row["AccountName"],
							copying_row["Date"],
							copying_row["Hours_Played"],
							copying_row["isPositive"],
							copying_row["review"],
							1,
							copying_row["Polarity"],
							-2,
							-2,
							-2,
							-2]
			new_insert_row = dict(zip(data_columns,insert_row))
			df_for_multi = df_for_multi.append(new_insert_row,ignore_index=True)	
	elif file =="Graphics.csv":
		for row in range(0,height):
			copying_row = data.iloc[row].copy(deep=True)
			insert_row = [copying_row["gameId"],
							copying_row["AccountName"],
							copying_row["Date"],
							copying_row["Hours_Played"],
							copying_row["isPositive"],
							copying_row["review"],
							-2,
							-2,
							1,
							copying_row["Polarity"],
							-2,
							-2]
			new_insert_row = dict(zip(data_columns,insert_row))
			df_for_multi = df_for_multi.append(new_insert_row,ignore_index=True)	
	elif file == "Gameplay.csv":
		for row in range(0,height):
			copying_row = data.iloc[row].copy(deep=True)
			insert_row = [copying_row["gameId"],
							copying_row["AccountName"],
							copying_row["Date"],
							copying_row["Hours_Played"],
							copying_row["isPositive"],
							copying_row["review"],
							-2,
							-2,
							-2,
							-2,
							1,
							copying_row["Polarity"]]
			new_insert_row = dict(zip(data_columns,insert_row))
			df_for_multi = df_for_multi.append(new_insert_row,ignore_index=True)	


df_for_multi = df_for_multi.sort_values(by=["gameId","AccountName"])
df_for_multi.to_csv("{}/{}.csv".format(destination_str,"multi"),index=False)
