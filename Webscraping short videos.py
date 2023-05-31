
"""
This script is for the Webscraping section of the 'Analysis of Investing on YouTube' project.
"""

# Running the script which loads the packages for the project
import subprocess
import sys

python_executable = sys.executable
# subprocess.call([python_executable, "Installing & Loading packages.py"])

""" Loading libraries - may move to 'Installing and Loading packages.py' """
import requests
import csv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from youtube_transcript_api._errors import NoTranscriptFound
from youtubesearchpython import VideosSearch, ChannelsSearch
import time
from googleapiclient.discovery import build
import pandas as pd
import os


""" Part 1: Necessary functions """

""" 1 a) Function to search YouTube for relevant video ids"""
def search_videos_by_keyword(keyword, max_results=100):
    videos_search = VideosSearch(keyword, limit=max_results)

    video_ids = []
    for video in videos_search.result()['result']:
        video_ids.append(video['id'])

    return video_ids

def search_videos_by_keyword_in_channel(channel_ids, keyword, max_results=100, api_key=None):
    video_ids_by_channel = {}
    all_video_ids = []

    youtube = build('youtube', 'v3', developerKey=api_key)

    for channel_id in channel_ids:
        video_ids = []

        search_response = youtube.search().list(
            part='id',
            q=keyword,
            channelId=channel_id,
            type='video',
            maxResults=max_results,
            videoDuration='short'
        ).execute()

        for item in search_response['items']:
            video_id = item['id']['videoId']
            video_ids.append(video_id)
            all_video_ids.append(video_id)

        if channel_id in video_ids_by_channel:
            video_ids_by_channel[channel_id].extend(video_ids)
        else:
            video_ids_by_channel[channel_id] = video_ids

    return video_ids_by_channel, all_video_ids


def find_channel_ids(channel_usernames, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)

    channel_ids_username = []
    channel_ids = []

    for channel_username in channel_usernames:
        username = channel_username.lstrip('@')  # Remove the "@" symbol if present

        search_response = youtube.search().list(
            part='snippet',
            q=username,
            type='channel',
            maxResults=1,
            fields='items(id,snippet(title,channelId))'
        ).execute()

        if 'items' in search_response and len(search_response['items']) > 0:
            item = search_response['items'][0]
            channel_id = item['id']['channelId']
            channel_name = item['snippet']['title']
            channel_ids_username.append((channel_name, channel_id))
            channel_ids.append(channel_id)
        else:
            channel_ids_username.append((username, None))

    return channel_ids_username, channel_ids

""" 1 b) Function(s) for extracting relevant data from videos """
def extract_metadata(video_id, api_key):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    video_info = data["items"][0]

    # Video metadata
    metadata = {}
    fields = ['channelId', 'publishedAt', 'tags', 'id', 'title', 'description', 'likeCount', 'viewCount', 'commentCount', 'Transcript']

    for field in fields:
        try:
            if field == 'Transcript':
                try:
                    metadata[field] = YouTubeTranscriptApi.get_transcript(video_id)
                except (TranscriptsDisabled, NoTranscriptFound):
                    metadata[field] = None
            elif field in ['likeCount', 'viewCount', 'commentCount']:
                metadata[field] = video_info["statistics"][field]
            else:
                metadata[field] = video_info["snippet"][field] if field.lower() != 'id' else video_info[
                    'id']
        except KeyError:
            pass

    return metadata

def extract_multiVideo_metadata(video_ids, api_key):
    metadata = []
    for video_id in video_ids:
        data = extract_metadata(video_id, api_key)
        metadata.append(data)

    return metadata

""" 1 c) Function(s) to add data from videos to csv """
def append_metadata_to_csv(metadata, csv_file):
    var_names = ['channelId', 'publishedAt', 'tags', 'id', 'title', 'description', 'likeCount', 'viewCount', 'commentCount', 'Transcript']

    # Create a DataFrame with the metadata
    df = pd.DataFrame(metadata, columns=var_names)

    # Replace any newline characters in the 'Transcript' column
    df['Transcript'] = df['Transcript'].apply(lambda x: ' '.join(segment['text'] for segment in x) if x is not None else None)

    # Replace any remaining newline characters in other fields
    df = df.replace('\n', ' ', regex=True)

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    return df

""" Part 2: Connecting to YouTube API """
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


    # Finding channel ids
## Crypto
#channel_ids_username_crypto, channel_ids_crypto = find_channel_ids(channel_names_crypto, api_key)

## Stocks
#channel_ids_username_stocks, channel_ids_stocks = find_channel_ids(channel_names_stocks, api_key)

# To reduce the times I call the YouTube API (as there are daily restrictions) I will hardcode the output of the channel ids function for subsequent runs
channel_ids_crypto = ['UCjemQfjaXAzA-95RKoy9n_g', 'UCN9Nj4tjXbVTLYWN0EKly_Q', 'UCl2oCaw8hdR_kbqyqd2klIA', 'UCqK_GSMbpiV8spgD3ZGloSw', 'UCCatR7nWbYrkVXdxXb4cGXw',
                      'UCrYmtJBtLdtm2ov84ulV-yg', 'UCviqt5aaucA1jP3qFmorZLQ', 'UCCE0Z5w0jcYipFPnve0R8Rg', 'UCu7Sre5A1NMV8J3s2FhluCw', 'UC-5HLi3buMzdxjdTdic3Aig',
                      'UCnMku7J_UtwlcSfZlIuQ3Kw', 'UC67AEEecqFEc92nVvcqKdhA', 'UCI7M65p3A-D3P4v5qW8POxQ', 'UCLo66QVfEod0nNM_GzKNxmQ', 'UCwsRWmIL5XKqFtdytBfeX0g',
                      'UCMbOW2DpyVfqYRP-KIaurJw', 'UCj1pAhsJCqMMcmAeQJgOc9w', 'UCc4Rz_T9Sb1w5rqqo9pL1Og', 'UCtR_Pn4XLjLU53LhRqYrJTQ', 'UCvwEcDj42J1gbh5R8D4EqQg']
channel_ids_stocks = ['UCtlAFoYl2aWb6pMiHCctQHA', 'UCGL9ubdGcvZh_dvSV2z1hoQ', 'UCBayuhgYpKNbhJxfExYkPfA', 'UCoMzWLaPjDJBbipihD694pQ', 'UCTQuKo8v0PaPCvk5aRxc_Yg',
                         'UCKQvGU-qtjEthINeViNbn6A', 'UC0BGhWsIbV7Dm-lsvhdlMbA', 'UCxyD6SQNbe8sh-jqo8eTfEg', 'UCn2JOViAAWssWefchGpTNvw', 'UCZy3pEdGnWLDsqGu56axxcQ',
                         'UCnMn36GT_H0X-w5_ckLtlgQ', 'UCpRQuynBX9Qy9tPrcswpPag', 'UC3mjMoJuFnjYRBLon_6njbQ', 'UCG9FGwgAx9-RKq1smF1lc8w', 'UCSglJMvX-zSgv3PEJIE_inw',
                         'UCKvWzzrVUA_DSCoKXL6GU2w', 'UCUvvj5lwue7PspotMDjk5UA', 'UCV6KDgJskWaEckne5aPA0aQ', 'UCGy7SkBjcIAgTiwkXEtPnYg',
                         'UCO3tlaeZ6Z0ZN5frMZI3-uQ', 'UCkNgKCu9062P0CPyVoBI5sQ', 'UCzBuoeqN94gdGcx5g4CgYNA', 'UCNttUtm9vloDjo-a2jqkoNQ', 'UC12lnsYNt8_VthTNOuOGTmQ']

    # Finding video ids within channels (by channel id and keyword search)
## Crypto
    # Keyword: Crypto
# video_ids_by_channel_crypto1, video_ids_crypto1 = search_videos_by_keyword_in_channel(channel_ids_crypto, "crypto tips", max_results=10, api_key = api_key)
#     # Keyword: Crypto
# video_ids_by_channel_crypto2, video_ids_crypto2 = search_videos_by_keyword_in_channel(channel_ids_crypto, "advice", max_results=10, api_key = api_key)
#     # Keyword: Crypto
# video_ids_by_channel_crypto3, video_ids_crypto3 = search_videos_by_keyword_in_channel(channel_ids_crypto, "#notfinancialadvice", max_results=10, api_key = api_key_2)
#     # Keyword: Crypto
# video_ids_by_channel_crypto4, video_ids_crypto4 = search_videos_by_keyword_in_channel(channel_ids_crypto, "what to buy", max_results=10, api_key = api_key_2)

#video_ids_crypto = video_ids_crypto1 + video_ids_crypto2 + video_ids_crypto3 + video_ids_crypto4

## Stocks
    # Keyword: Stocks
# video_ids_by_channel_stocks1, video_ids_stocks1 = search_videos_by_keyword_in_channel(channel_ids_stocks, "stock tips", max_results=10, api_key = api_key_5)
#     # Keyword: Stocks
# video_ids_by_channel_stocks2, video_ids_stocks2 = search_videos_by_keyword_in_channel(channel_ids_stocks, "investment advice", max_results=10, api_key = api_key_5)
#     # Keyword: Stocks
# video_ids_by_channel_stocks3, video_ids_stocks3 = search_videos_by_keyword_in_channel(channel_ids_stocks, "#notfinancialadvice", max_results=10, api_key = api_key_6)
#     # Keyword: Stocks
# video_ids_by_channel_stocks4, video_ids_stocks4 = search_videos_by_keyword_in_channel(channel_ids_stocks, "what to buy", max_results=10, api_key = api_key_6)

#video_ids_stocks = video_ids_stocks1 + video_ids_stocks2 + video_ids_stocks3 + video_ids_stocks4


""" Part 4: Scraping the data """
## Crypto
# metadata_crypto1 = extract_multiVideo_metadata(video_ids_crypto1, api_key_3)
# metadata_crypto2 = extract_multiVideo_metadata(video_ids_crypto2, api_key_3)
# metadata_crypto3 = extract_multiVideo_metadata(video_ids_crypto3, api_key_4)
# metadata_crypto4 = extract_multiVideo_metadata(video_ids_crypto4, api_key_4)

## Stocks
# metadata_stocks = extract_multiVideo_metadata(video_ids_stocks1, api_key_7)
# metadata_stocks2 = extract_multiVideo_metadata(video_ids_stocks2, api_key_7)
# metadata_stocks3 = extract_multiVideo_metadata(video_ids_stocks3, api_key_8)
# metadata_stocks4 = extract_multiVideo_metadata(video_ids_stocks4, api_key_8)

## Overall
# video_ids = video_ids_crypto + video_ids_stocks
# metadata_overall = extract_multiVideo_metadata(video_ids, api_key)

""" Part 5: Creating the dataset """
## Crypto
csv_fle_crypto1 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_short_videos/crypto1_short.csv"
csv_fle_crypto2 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_short_videos/crypto2_short.csv"
csv_fle_crypto3 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_short_videos/crypto3_short.csv"
csv_fle_crypto4 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_short_videos/crypto4_short.csv"

# df_crypto_short = append_metadata_to_csv(metadata_crypto1, csv_fle_crypto1)
# df_crypto2_short = append_metadata_to_csv(metadata_crypto2, csv_fle_crypto2)
# df_crypto3_short = append_metadata_to_csv(metadata_crypto3, csv_fle_crypto3)
# df_crypto4_short = append_metadata_to_csv(metadata_crypto4, csv_fle_crypto4)

#print(df_crypto.head())

## Stocks
csv_fle_stocks1 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_short_videos/stocks1_short.csv"
csv_fle_stocks2 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_short_videos/stocks2_short.csv"
csv_fle_stocks3 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_short_videos/stocks3_short.csv"
csv_fle_stocks4 = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_short_videos/stocks4_short.csv"

# df_stocks_short = append_metadata_to_csv(metadata_stocks, csv_fle_stocks1)
# df_stocks2_short = append_metadata_to_csv(metadata_stocks2, csv_fle_stocks2)
# df_stocks3_short = append_metadata_to_csv(metadata_stocks3, csv_fle_stocks3)
# df_stocks4_short = append_metadata_to_csv(metadata_stocks4, csv_fle_stocks4)
# print(df_stocks.head())

## Overall
# csv_fle_overall = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/overall_video_data.csv"
# metadata_overall = metadata_stocks + metadata_stocks2 + metadata_stocks3 + metadata_stocks4 + metadata_crypto + metadata_crypto2 + metadata_crypto3 + metadata_crypto4
# df_overall = append_metadata_to_csv(metadata_overall, csv_fle_overall)

# Combining all dataframes and saving as overall csv (again coding in this way to reduce times connecting to YouTube API - not the most efficient way but gets around their restrictions)
folder_path ="C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_short_videos/"
dataframes = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenate the dataframes together
overall_video_data = pd.concat(dataframes)

# Save the combined dataframe to a new CSV file
overall_video_data.to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data_short_videos/overall_short_video_data.csv", index=False)