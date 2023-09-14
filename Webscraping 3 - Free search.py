
""" This is the webscraping of videos found by searching YouTube with keywords.

Webscraping 1 and 2 search within channels. This script does the free search of YouTube."""

""" Part 1: Importing packages from other scripts """

""" 1 a) Loading libraries """
# Running the script which loads the packages for the project
import subprocess
import sys

python_executable = sys.executable
subprocess.call([python_executable, "Installing & Loading packages.py"])

import os
import pandas as pd

""" 1 b) Importing necessary functions written for the webscraping """
from YouTube_scraping_functions import find_channel_ids, search_videos_by_keyword_in_channel, search_videos_by_keyword
from YouTube_scraping_functions import extract_metadata, extract_multiVideo_metadata
from YouTube_scraping_functions import append_metadata_to_csv

""" Part 2: Connecting to YouTube API """
# Using multiple API keys gets around daily restrictions on calls to the API
api_key = "AIzaSyBvqa2-cEtjDKTCZ47qQcVJQqY4wKk5kek"
api_key_2 = "AIzaSyCkMzXm8C6lhQJCJUJbWM57bPF3Bi5jO3U"
api_key_3 = "AIzaSyAOLyZ77wU_n_wBg22jcN_RF-QMXvg7azg"
api_key_4 = "AIzaSyDCxeqgw_SekMX_tnuJ2keIXIydChvMOZQ"
api_key_5 = "AIzaSyDJaz06nOcCQMOsi9UrC6Y9LfZ2r3oIBGc"
api_key_6 = "AIzaSyAR6W-smS3JzjJwlJ5KafSwi8f2k-kIVHY"
api_key_7 = "AIzaSyDVlVnYgR4cL8zx4bHjPJygJqVLUGqcR0w"
api_key_8 = "AIzaSyAl8cA_xOIBvt4m0fhQDmGKfyrklDZpKLU"

""" Part 3: Finding relevant video ids by keyword search """
video_ids_search1 = search_videos_by_keyword("investment tips", 1200)
video_ids_search2 = search_videos_by_keyword("investment advice", 1200)
video_ids_search3 = search_videos_by_keyword("stock tips", 1200)
video_ids_search4 = search_videos_by_keyword("#notfinancialadvice", 1200)
video_ids_search5 = search_videos_by_keyword("stocks to buy", 1200)
video_ids_search6 = search_videos_by_keyword("crypto to buy", 1200)
video_ids_search7 = search_videos_by_keyword("crypto tips", 1200)
video_ids_search8 = search_videos_by_keyword("next big cryptocurrency", 1200)

""" Part 4: Scraping the data """
metadata_search1 = extract_multiVideo_metadata(video_ids_search1, api_key)
metadata_search2 = extract_multiVideo_metadata(video_ids_search2, api_key)
metadata_search3 = extract_multiVideo_metadata(video_ids_search3, api_key_4)
metadata_search4 = extract_multiVideo_metadata(video_ids_search4, api_key_4)
metadata_search5 = extract_multiVideo_metadata(video_ids_search5, api_key_5)
metadata_search6 = extract_multiVideo_metadata(video_ids_search6, api_key_5)
metadata_search7 = extract_multiVideo_metadata(video_ids_search7, api_key_6)
metadata_search8 = extract_multiVideo_metadata(video_ids_search8, api_key_6)

""" Part 5: Appending to a csv """
csv_fle_search1 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_free_search/search1.csv"
csv_fle_search2 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_free_search/search2.csv"
csv_fle_search3 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_free_search/search3.csv"
csv_fle_search4 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_free_search/search4.csv"
csv_fle_search5 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_free_search/search5.csv"
csv_fle_search6 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_free_search/search6.csv"
csv_fle_search7 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_free_search/search7.csv"
csv_fle_search8 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_free_search/search8.csv"

df_search1 = append_metadata_to_csv(metadata_search1, csv_fle_search1)
df_search2 = append_metadata_to_csv(metadata_search2, csv_fle_search2)
df_search3 = append_metadata_to_csv(metadata_search3, csv_fle_search3)
df_search4 = append_metadata_to_csv(metadata_search4, csv_fle_search4)
df_search5 = append_metadata_to_csv(metadata_search5, csv_fle_search5)
df_search6 = append_metadata_to_csv(metadata_search6, csv_fle_search6)
df_search7 = append_metadata_to_csv(metadata_search7, csv_fle_search7)
df_search8 = append_metadata_to_csv(metadata_search8, csv_fle_search8)

""" 5 b) Combining all dataframes and saving as overall csv 
    (again coding in this way to reduce times connecting to YouTube API - not the most efficient way but gets around the YouTube restrictions) """
folder_path ="C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_free_search/"
dataframes = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"): # folder is manually created so only the .csv files from this script are in it
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenating the free search dataframes together
overall_video_data = pd.concat(dataframes)

# All free search videos together saved in csv
overall_video_data.to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/Overall datasets/overall_free_search.csv", index=False)

