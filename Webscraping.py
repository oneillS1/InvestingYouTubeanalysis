
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

api_key = "AIzaSyBRpuSMO306VzZkUGCNt06zIk7deIJk0Ec"
channel_names = ["@BitBoyCryptoChannel"]
start_time = time.time()
channel_ids_username, channel_ids = find_channel_ids(channel_names, api_key)
end_time = time.time()
print(channel_ids_username, channel_ids)
print("Elapsed time:", end_time - start_time, " seconds")

start_time = time.time()
video_ids_by_channel, video_ids = search_videos_by_keyword_in_channel(channel_ids, "crypto", max_results=10, api_key = api_key)
end_time = time.time()
print(video_ids_by_channel)
print(video_ids)
print("Elapsed time:", end_time - start_time, " seconds")



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

# metadata = extract_multiVideo_metadata(video_ids, api_key)
# append_metadata_to_csv(metadata, csv_fle)

""" Part 3: Finding relevant videos """
# Usage
keyword = 'crypto advice'  # Replace with your desired keyword
# start_time = time.time()
# video_ids2 = search_videos_by_keyword(keyword)
# end_time = time.time()
# print("Elapsed time:", end_time - start_time, " seconds")

# Print the video IDs
# print(video_ids2)


""" Part 4: Scraping the data """
metadata = extract_multiVideo_metadata(video_ids, api_key)

""" Part 5: Creating the dataset """
csv_fle = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/data_test.csv"
append_metadata_to_csv(metadata, csv_fle)