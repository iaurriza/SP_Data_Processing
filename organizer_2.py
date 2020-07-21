import pandas as pd
import csv
import re
import os


def dropCol(df):
    del df['isSingle']
    del df['isValid']
    del df['Hours_Played']
    del df['noOfHelpful']
    del df['noOfFunny']
    del df['Date']

source_str = './cleaned'
destination_str = './cleaned_sorted'

print("START")

if not os.path.exists(destination_str):
	os.mkdir(destination_str)

data_columns = ["gameId","AccountName","isPositive","review","is_audio","audio_polarity","is_graphics","graphics_polarity","is_gameplay","gameplay_polarity"]
#add accountID if ever
compiled_data = pd.DataFrame(columns=data_columns)


# Import All Filess
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
        multi_data = pd.read_csv('{}/{}'.format(source_str,file),index_col=None)				#Load original dataframe


#Remove irrelevant data
audio_data = audio_data.drop(audio_data[audio_data['isValid']==0].index)
audio_data = audio_data.drop(audio_data[audio_data['isSingle']==0].index)
audio_data = audio_data.reset_index(drop=True)
dropCol(audio_data)
columns_list = audio_data.columns
print(columns_list)

graphics_data = graphics_data.drop(graphics_data[graphics_data['isValid']==0].index)
graphics_data = graphics_data.drop(graphics_data[graphics_data['isSingle']==0].index)
graphics_data = graphics_data.reset_index(drop=True)
dropCol(graphics_data)


gameplay_data = gameplay_data.drop(gameplay_data[gameplay_data['isValid']==0].index)
gameplay_data = gameplay_data.drop(gameplay_data[gameplay_data['isSingle']==0].index)
gameplay_data = gameplay_data.reset_index(drop=True)
dropCol(gameplay_data)
print(audio_data.shape)
print(graphics_data.shape)
print(gameplay_data.shape)


#Insert multi to specific DFs
height = multi_data.shape[0]
print(audio_data.columns)

for row in range(0,height):
    copying_row = multi_data.iloc[row].copy(deep=True)				#Copy Row	
    insert_row = [  copying_row["AccountName"],
                    copying_row["gameId"],
                    copying_row["isPositive"],
                    copying_row["review"]
                ]
    if copying_row["is_audio"] == 1:
        insert_row.append(copying_row["audio_polarity"])
        audio_data = audio_data.append(pd.Series(insert_row, index=audio_data.columns ),ignore_index=True)
        del insert_row[-1]
    if copying_row["is_graphics"] == 1:
        insert_row.append(copying_row["graphics_polarity"])
        graphics_data = graphics_data.append(pd.Series(insert_row, index=graphics_data.columns ),ignore_index=True)
        del insert_row[-1]
    if copying_row["is_gameplay"] == 1:
        insert_row.append(copying_row["gameplay_polarity"])
        gameplay_data = gameplay_data.append(pd.Series(insert_row, index=gameplay_data.columns ),ignore_index=True)
        del insert_row[-1]
	
# print(audio_data.shape)
# print(graphics_data.shape)
# print(gameplay_data.shape)

audio_data.to_csv("{}/Audio.csv".format(destination_str),index=False)
gameplay_data.to_csv("{}/Gameplay.csv".format(destination_str),index=False)
graphics_data.to_csv("{}/Graphics.csv".format(destination_str),index=False)
