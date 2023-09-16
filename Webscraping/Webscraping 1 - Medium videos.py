
"""
This script is for the Webscraping section of the 'Analysis of Investing on YouTube' project.
This script scrapes the medium length videos (4-15 mins)

*NB: This file is used in conjunction with 'YouTube_scraping_functions.py' & 'Hard_coding.py' & 'Installing & Loading packages.py'
Related files 'webscraping short videos.py' and 'description of logic followed.py'

"""

""" Part 1: Importing packages & also from other scripts """

""" 1 a) Loading libraries """
# Running the script which loads the packages for the project
import subprocess
import sys

python_executable = sys.executable
subprocess.call([python_executable, "Installing & Loading packages.py"])

import os
import pandas as pd

""" 1 b) Import hard coding elements - usage explained in the script (largely to avoid YouTube API daily restrictions """

""" 1 c) Importing necessary functions written for the webscraping """
from YouTube_scraping_functions import find_channel_ids, search_videos_by_keyword_in_channel
from YouTube_scraping_functions import extract_multiVideo_metadata
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

""" Part 3: Finding relevant videos """
    # Channel usernames identified from sector knowledge, google searches
## Crypto
channel_names_crypto = ["@BitBoyCryptoChannel", "@CryptoBanterGroup", "@TheCryptoLark", "@CoinBureau", "@DataDash",
                        "@IvanOnTech", "@CryptoJebb", "@AltcoinDaily""@CryptoZombie", "@CryptoLove", "@TheModernInvestor",
                        "@CryptoCapitalVenture", "@CryptoDaily", "@CryptosRUs", "@NuggetsNews", "@CryptoCrowOfficial",
                        "@cryptobeadles3949", "@ReadySet", "@TheMoon", "@JamesCryptoGuru", "@Jungernaut"]

## Stocks
channel_names_stocks = ["@RickyGutierrezz", "@thetradingchannel", "@DaytradeWarrior", "@StockMoe", "@CharlieChang", "@AlexBeckersChannel",
            "@ZipTrader", "@InformedTrades", "@claytrader", "@TradeIdeas", "@FinancialEducation", "@MotleyFool", "@RyanScribner",
            "@InvestingwithTom", "@LearntoInvest", "@themebfabershow1017", "@MeetKevin", "@GrahamStephan",
            "@AndreiJikh", "@NateOBrien", "@wealthhacker", "@wolfofdubai", "@StockswithJosh", "@jeremylefebvremakesmoney7934"]


""" 3 b) Finding channel ids from the usernames """
# Crypto
channel_ids_username_crypto, channel_ids_crypto = find_channel_ids(channel_names_crypto, api_key)

# Stocks
channel_ids_username_stocks, channel_ids_stocks = find_channel_ids(channel_names_stocks, api_key)

# To reduce the times I call the YouTube API (as there are daily restrictions) I hardcode the output of the channel ids function for subsequent runs.
# The output of the two functions above is contained in the 'Hard_coding.py' file & imported here as 'channnel_ids_crypto' and 'channel_ids_stocks'.
# Alternatively I could just run the code above too.

""" 3 c) Finding relevant video ids within channels (by channel id and keyword search) """
# Crypto
   # Keyword: Crypto tips
video_ids_by_channel_crypto1, video_ids_crypto1 = search_videos_by_keyword_in_channel(channel_ids_crypto, "crypto tips", max_results=10, api_key = api_key)
   # Keyword: advice
video_ids_by_channel_crypto2, video_ids_crypto2 = search_videos_by_keyword_in_channel(channel_ids_crypto, "advice", max_results=10, api_key = api_key)
   # Keyword: #notfinancialadvice
video_ids_by_channel_crypto3, video_ids_crypto3 = search_videos_by_keyword_in_channel(channel_ids_crypto, "#notfinancialadvice", max_results=10, api_key = api_key_2)
   # Keyword: what to buy
video_ids_by_channel_crypto4, video_ids_crypto4 = search_videos_by_keyword_in_channel(channel_ids_crypto, "what to buy", max_results=10, api_key = api_key_2)

# Similarly I have hard coded the outputs of the above (which is a list of video ids) so as to run subsequent code without reconnecting to the YouTube API
# The output of is contained in the 'Hard_coding.py' file & imported here as 'video_ids_crypto' with the 1-4 as suffix.
# Alternatively I could just run the code above too but it will call YouTube API a lot so use different API keys

# Stocks
   # Keyword: Stock tips
video_ids_by_channel_stocks1, video_ids_stocks1 = search_videos_by_keyword_in_channel(channel_ids_stocks, "stock tips", max_results=10, api_key = api_key_2)
   # Keyword: investment advice
video_ids_by_channel_stocks2, video_ids_stocks2 = search_videos_by_keyword_in_channel(channel_ids_stocks, "investment advice", max_results=10, api_key = api_key)
   # Keyword: #notfinancialadvice
video_ids_by_channel_stocks3, video_ids_stocks3 = search_videos_by_keyword_in_channel(channel_ids_stocks, "#notfinancialadvice", max_results=10, api_key = api_key)
   # Keyword: what to buy
video_ids_by_channel_stocks4, video_ids_stocks4 = search_videos_by_keyword_in_channel(channel_ids_stocks, "what to buy", max_results=10, api_key = api_key)

# Similarly I have hard coded the outputs of the above (which is a list of video ids) so as to run subsequent code without reconnecting to the YouTube API
# The output of is contained in the 'Hard_coding.py' file & imported here as 'video_ids_stocks' with the 1-4 as suffix.
# Alternatively I could just run the code above too but it will call YouTube API a lot so use different API keys

""" Part 4: Scraping the data """
# Crypto
metadata_crypto1 = extract_multiVideo_metadata(video_ids_crypto1, api_key_2)
metadata_crypto2 = extract_multiVideo_metadata(video_ids_crypto2, api_key_3)
metadata_crypto3 = extract_multiVideo_metadata(video_ids_crypto3, api_key_3)
metadata_crypto4 = extract_multiVideo_metadata(video_ids_crypto4, api_key_4)

# Stocks
metadata_stocks = extract_multiVideo_metadata(video_ids_stocks1, api_key_5)
metadata_stocks2 = extract_multiVideo_metadata(video_ids_stocks2, api_key_6)
metadata_stocks3 = extract_multiVideo_metadata(video_ids_stocks3, api_key_7)
metadata_stocks4 = extract_multiVideo_metadata(video_ids_stocks4, api_key_8)

""" Part 5: Creating the dataset """
## Crypto
csv_fle_crypto1 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/crypto1.csv"
csv_fle_crypto2 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/crypto2.csv"
csv_fle_crypto3 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/crypto3.csv"
csv_fle_crypto4 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/crypto4.csv"

df_crypto = append_metadata_to_csv(metadata_crypto1, csv_fle_crypto1)
df_crypto2 = append_metadata_to_csv(metadata_crypto2, csv_fle_crypto2)
df_crypto3 = append_metadata_to_csv(metadata_crypto3, csv_fle_crypto3)
df_crypto4 = append_metadata_to_csv(metadata_crypto4, csv_fle_crypto4)

#print(df_crypto.head())

## Stocks
csv_fle_stocks1 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/stocks1.csv"
csv_fle_stocks2 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/stocks2.csv"
csv_fle_stocks3 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/stocks3.csv"
csv_fle_stocks4 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/stocks4.csv"

df_stocks = append_metadata_to_csv(metadata_stocks, csv_fle_stocks1)
df_stocks2 = append_metadata_to_csv(metadata_stocks2, csv_fle_stocks2)
df_stocks3 = append_metadata_to_csv(metadata_stocks3, csv_fle_stocks3)
df_stocks4 = append_metadata_to_csv(metadata_stocks4, csv_fle_stocks4)
# print(df_stocks.head())

""" 5 b) Combining all dataframes and saving as overall csv 
    (again coding in this way to reduce times connecting to YouTube API - not the most efficient way but gets around their restrictions) """
folder_path ="C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/"
dataframes = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenating medium dataframes
overall_video_data = pd.concat(dataframes)

# All medium videos together saved in csv
overall_video_data.to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/Overall datasets/overall_long_video_data.csv", index=False)



""" Additional helpful code """
    # Timing of functions
#start_time = time.time()
#end_time = time.time()
#print("Elapsed time:", end_time - start_time, " seconds")