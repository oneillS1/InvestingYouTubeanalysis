
""" This script explores the initial dataset created by the 3 webscraping scripts """

""" Part 1: Importing libraries needed """
import os
import pandas as pd

""" Part 2: Importing and appending the datasets """
folder_path ="C:/Users/Steve.HAHAHA/Desktop/Dissertation/Overall datasets/"
dataframes = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df['Source'] = filename
        dataframes.append(df)

# Concatenate the dataframes together
raw_video_data = pd.concat(dataframes)

# Save the combined dataframe to a new CSV file
#raw_video_data.to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/Final dataset(s) for analysis/raw data.csv", index=False)

""" Part 3: Initial data exploration 
- checking: no of videos in dataset, duplicates, no without transcript """
# print(raw_video_data.shape)
# print(raw_video_data.dtypes)
# print(raw_video_data.columns)

# Duplicates
number_of_duplicates = raw_video_data.duplicated().sum()
print('Number of duplicte rows: ', number_of_duplicates)

# No transcript
empty_str_count = raw_video_data["Transcript"].isna().sum()
print("Number of rows with no transcript:", empty_str_count)

print(raw_video_data.shape)

# Duplicate IDs (as there have been changes to the likes between searches and therefore the same videos don't come up as duplicates)
duplicates_count = raw_video_data.duplicated(subset=['id']).sum()
print("Number of duplicate ids:", duplicates_count)

## Dropping duplicates and no transcript videos
video_data = raw_video_data.drop_duplicates(subset=['id'])
video_data = video_data.dropna(subset=["Transcript"])
print(video_data.shape)

# Checking how many videos from each search method
print(video_data['Source'].value_counts())

""" Part 4: Saving cleaned dataset for analysis """
video_data.to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/Final dataset(s) for analysis/cleaned data.csv", index=False)