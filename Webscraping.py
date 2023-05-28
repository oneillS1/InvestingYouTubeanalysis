
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
            videoDuration='medium'
        ).execute()
        print(search_response)

        for item in search_response['items']:
            print(item['id'])
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

        response = youtube.search().list(part='snippet', q=username, type='channel').execute()
        #print(response)

        if 'items' in response and len(response['items']) > 0:
            channel_id = response['items'][0]['snippet']['channelId']

            channel_response = youtube.channels().list(part='snippet', id=channel_id).execute()
            #print(channel_response)

            if 'items' in channel_response and len(channel_response['items']) > 0:
                channel_name = channel_response['items'][0]['snippet']['title']
                channel_ids_username.append((channel_name, channel_id))
                channel_ids.append(channel_id)
            else:
                channel_ids_username.append((username, channel_id))
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

    #print(metadata)

    return metadata

def extract_multiVideo_metadata(video_ids, api_key):
    metadata = []
    for video_id in video_ids:
        data = extract_metadata(video_id, api_key)
        metadata.append(data)
    #print(metadata)
    return metadata

""" 1 c) Function(s) to add data from videos to csv """
def append_metadata_to_csv2(metadata, csv_file):
    var_names = ['channelId', 'publishedAt', 'tags', 'id', 'title', 'description', 'likeCount', 'viewCount', 'commentCount', 'Transcript']

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = var_names)

        if csvfile.tell() == 0:
            writer.writeheader()

        for data in metadata:
            if 'Transcript' in data:
                transcript = data['Transcript']
                if transcript is not None:
                    # Concatenate the 'text' values from each segment
                    transcript_text = ' '.join(segment["text"] for segment in transcript)
                    data['Transcript'] = transcript_text

            print(data)
            writer.writerow(data)

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
api_key = "AIzaSyBRpuSMO306VzZkUGCNt06zIk7deIJk0Ec"

""" Part 3: Finding relevant videos """
    # Channel usernames identified from sector knowledge, google searches
## Crypto
channel_names_crypto = ["@BitBoyCryptoChannel", "@SheldonEvansx", "@CryptoBanterGroup", "@TheCryptoLark"
                 "@CoinBureau", "@DataDash", "@IvanOnTech", "@CryptoJebb", "@AltcoinDaily", "@CryptoZombie", "@CryptoLove", "@TheModernInvestor",
                 "@CryptoCapitalVenture", "@CryptoDaily", "@CryptosRUs", "@NuggetsNews", "@CryptoCrowOfficial",
                 "@cryptobeadles3949", "@ReadySet", "@TheMoon", "@JamesCryptoGuru", "@Jungernaut"]
## Stocks
channel_names_stocks = ["@RickyGutierrezz", "@thetradingchannel", "@DaytradeWarrior", "@StockMoe", "@CharlieChang", "@AlexBeckersChannel",
            "@ZipTrader", "@InformedTrades", "@claytrader", "@TradeIdeas", "@FinancialEducation", "@MotleyFool", "@RyanScribner",
            "@InvestingwithTom", "@LearntoInvest", "@themebfabershow1017", "@MeetKevin", "@GrahamStephan",
            "@AndreiJikh", "@NateOBrien", "@wealthhacker", "@wolfofdubai", "@StockswithJosh", "@jeremylefebvremakesmoney7934"]


    # Finding channel ids
## Crypto
channel_ids_username_crypto, channel_ids_crypto = find_channel_ids(channel_names_crypto, api_key)
## Stocks
channel_ids_username_stocks, channel_ids_stocks = find_channel_ids(channel_names_stocks, api_key)

    # Finding video ids within channels (by channel id and keyword search)
## Crypto
    # Keyword: Crypto
video_ids_by_channel_crypto1, video_ids_crypto1 = search_videos_by_keyword_in_channel(channel_ids_crypto, "crypto", max_results=10, api_key = api_key)
    # Keyword: Crypto
video_ids_by_channel_crypto2, video_ids_crypto2 = search_videos_by_keyword_in_channel(channel_ids_crypto, "crypto tips", max_results=10, api_key = api_key)
    # Keyword: Crypto
video_ids_by_channel_crypto3, video_ids_crypto3 = search_videos_by_keyword_in_channel(channel_ids_crypto, "#notfinancialadvice", max_results=10, api_key = api_key)
    # Keyword: Crypto
video_ids_by_channel_crypto4, video_ids_crypto4 = search_videos_by_keyword_in_channel(channel_ids_crypto, "what to buy", max_results=10, api_key = api_key)

video_ids_crypto = video_ids_crypto1 + video_ids_crypto2 + video_ids_crypto3 + video_ids_crypto4

## Stocks
    # Keyword: Stocks
video_ids_by_channel_stocks1, video_ids_stocks1 = search_videos_by_keyword_in_channel(channel_ids_stocks, "crypto", max_results=10, api_key = api_key)
    # Keyword: Stocks
video_ids_by_channel_stocks2, video_ids_stocks2 = search_videos_by_keyword_in_channel(channel_ids_stocks, "crypto tips", max_results=10, api_key = api_key)
    # Keyword: Stocks
video_ids_by_channel_stocks3, video_ids_stocks3 = search_videos_by_keyword_in_channel(channel_ids_stocks, "#notfinancialadvice", max_results=10, api_key = api_key)
    # Keyword: Stocks
video_ids_by_channel_stocks4, video_ids_stocks4 = search_videos_by_keyword_in_channel(channel_ids_stocks, "what to buy", max_results=10, api_key = api_key)

video_ids_stocks = video_ids_stocks1 + video_ids_stocks2 + video_ids_stocks3 + video_ids_stocks4

""" Part 4: Scraping the data """
## Crypto
metadata_crypto = extract_multiVideo_metadata(video_ids_crypto, api_key)

## Stocks
metadata_stocks = extract_multiVideo_metadata(video_ids_stocks, api_key)

## Overall
# video_ids = video_ids_crypto + video_ids_stocks
# metadata_overall = extract_multiVideo_metadata(video_ids, api_key)

""" Part 5: Creating the dataset """
## Crypto
csv_fle_crypto = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/crypto_test.csv"
df_crypto = append_metadata_to_csv(metadata_crypto, csv_fle_crypto)

print(df_crypto.head())

## Stocks
csv_fle_stocks = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/stocks_test.csv"
df_stocks = append_metadata_to_csv(metadata_stocks, csv_fle_stocks)

print(df_stocks.head())

## Overall
# csv_fle_stocks = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/overall_test.csv"
# df_stocks = append_metadata_to_csv(metadata_overall, csv_fle_overall)

""" Additional helpful code """
    # Timing of functions
#start_time = time.time()
#end_time = time.time()
#print("Elapsed time:", end_time - start_time, " seconds")