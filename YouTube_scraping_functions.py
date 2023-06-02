
""" The functions needed to run the Webscraping element of the YouTube Investing analysis project """
import requests
import csv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from youtube_transcript_api._errors import NoTranscriptFound
from youtubesearchpython import VideosSearch, ChannelsSearch
from googleapiclient.discovery import build
import pandas as pd


""" 1: Function to search YouTube for relevant video ids"""
def search_videos_by_keyword(keyword, max_results=100):
    videos_search = VideosSearch(keyword, limit=max_results)

    video_ids = []
    for video in videos_search.result()['result']:
        video_ids.append(video['id'])

    return video_ids

""" 2: Function to search a channel on YouTube by keyword and return video ids (4 - 15 min videos) """
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

        for item in search_response['items']:
            video_id = item['id']['videoId']
            video_ids.append(video_id)
            all_video_ids.append(video_id)

        if channel_id in video_ids_by_channel:
            video_ids_by_channel[channel_id].extend(video_ids)
        else:
            video_ids_by_channel[channel_id] = video_ids

    return video_ids_by_channel, all_video_ids

""" 2 b) Find short videos (<4 mins) by channel id and keyword search """
def search_videos_by_keyword_in_channel_short(channel_ids, keyword, max_results=100, api_key=None):
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



""" 3: Function to find YouTube channel ids given the usernames of the channel """
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

""" 4: Function for extracting relevant data from YouTube video """
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

""" 5: Function to extract metadata from multiple videos (uses function 4) """
def extract_multiVideo_metadata(video_ids, api_key):
    metadata = []
    for video_id in video_ids:
        data = extract_metadata(video_id, api_key)
        metadata.append(data)

    return metadata

""" 6: Function(s) to add data from videos to csv """
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
