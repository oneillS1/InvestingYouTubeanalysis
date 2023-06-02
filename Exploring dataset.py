
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
        dataframes.append(df)

# Concatenate the dataframes together
raw_video_data = pd.concat(dataframes)

# Save the combined dataframe to a new CSV file
raw_video_data.to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/Final dataset(s) for analysis/raw data.csv", index=False)

""" Part 3: Initial data exploration 
- checking: no of videos in dataset, duplicates, no without transcript """

