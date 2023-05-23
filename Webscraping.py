
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
from youtubesearchpython import VideosSearch, ChannelsSearch
import time


""" Part 1: Necessary functions """

""" 1 a) Function to search YouTube for relevant video ids"""
def search_videos_by_keyword(keyword, max_results=100):
    videos_search = VideosSearch(keyword, limit=max_results)

    video_ids = []
    for video in videos_search.result()['result']:
        video_ids.append(video['id'])

    return video_ids

def search_videos_by_keyword_in_channel(channel_ids, keyword, max_results=100):
    video_ids_by_channel = {}

    for channel_id in channel_ids:
        channels_search = ChannelsSearch(channel_id, limit=1)

        if len(channels_search.result()['result']) == 0:
            video_ids_by_channel[channel_id] = "Not Found"
            continue

        channel_username = channels_search.result()['result'][0]['username']

        videos_search = VideosSearch(f'{keyword} inchannel:{channel_username}', limit=max_results)

        video_ids = []
        for video in videos_search.result()['result']:
            video_ids.append(video['id'])

        video_ids_by_channel[channel_id] = video_ids

    return video_ids_by_channel

def find_channel_ids2(channel_names, api_key):
    api_url = f"https://www.googleapis.com/youtube/v3/search"

    channel_ids = []

    for channel_name in channel_names:
        username = channel_name.lstrip('@')  # Remove the "@" symbol if present

        params = {
            "part": "id",
            "q": username,
            "type": "channel",
            "key": api_key
        }

        response = requests.get(api_url, params=params)
        data = response.json()
        print(data)

        if "items" in data and len(data["items"]) > 0:
            print('here')
            channel_id = data["items"][0]["id"]
            print(channel_id)
            channel_ids.append(channel_id)
        else:
            channel_ids.append(None)

    return channel_ids

from googleapiclient.discovery import build

def find_channel_ids(channel_usernames, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)

    channel_ids_username = []
    channel_ids = []

    for channel_username in channel_usernames:
        username = channel_username.lstrip('@')  # Remove the "@" symbol if present

        response = youtube.search().list(part='snippet', q=username, type='channel').execute()
        print(response)

        if 'items' in response and len(response['items']) > 0:
            channel_id = response['items'][0]['snippet']['channelId']

            channel_response = youtube.channels().list(part='snippet', id=channel_id).execute()
            print(channel_response)

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
channel_names = ["@stephenoneill3309", "@skysports", "@SkySportsF1", "@skysportspremierleague"]
channel_ids_username, channel_ids = find_channel_ids(channel_names, api_key)
print(channel_ids_username, channel_ids)

# UCn6Ra0_U_0yr2o-JZUHFFxQ
#video_ids3 = search_videos_by_keyword_in_channel(channel_ids, "crypto", max_results=100)
#print(video_ids3)

""" 1 b) Function(s) for extracting relevant data from videos """
def extract_metadata2(video_id, api_key):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    video_info = data["items"][0]

    # Video metadata
    channel = video_info["snippet"]["channelTitle"]
    date = video_info["snippet"]["publishedAt"]
    tags = video_info["snippet"]["tags"]
    id = video_info['id']
    title = video_info["snippet"]["title"]
    description = video_info["snippet"]["description"]
    #video_length = video_info["contentDetails"]["duration"]
    likes = video_info["statistics"]["likeCount"]
    views = video_info["statistics"]["viewCount"]
    comments = video_info["statistics"]["commentCount"]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    # Reshape to be a dictionary
    metadata = {
        'Channel': channel,
        'Date': date,
        'Tags': tags,
        'Video id': id,
        'Title': title,
        'Description': description,
        'Likes': likes,
        'Views': views,
        'Comments': comments,
        'Transcript': transcript
    }

    return metadata


def extract_metadata(video_id, api_key):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    video_info = data["items"][0]
    print(video_info)

    # Video metadata
    metadata = {}
    fields = ['channelId', 'publishedAt', 'tags', 'id', 'title', 'description', 'likeCount', 'viewCount', 'commentCount', 'Transcript']

    for field in fields:
        try:
            if field == 'Transcript':
                metadata[field] = YouTubeTranscriptApi.get_transcript(video_id)
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

    with open(csv_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = var_names)

        if csvfile.tell() == 0:
            writer.writeheader()

        for data in metadata:
            writer.writerow(data)


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
#video_ids = ['3irypwLDtTs', 'SbMtY-aJjFA']
#metadata = extract_multiVideo_metadata(video_ids, api_key)

""" Part 5: Creating the dataset """
#csv_fle = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/data_test.csv"
#append_metadata_to_csv(metadata, csv_fle)