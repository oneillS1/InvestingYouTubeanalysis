
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
from youtubesearchpython import VideosSearch
import time


""" Part 1: Necessary functions """

""" 1 a) Function to search YouTube for relevant video ids"""
def search_videos_by_keyword(keyword, max_results=100):
    videos_search = VideosSearch(keyword, limit=max_results)

    video_ids = []
    for video in videos_search.result()['result']:
        video_ids.append(video['id'])

    return video_ids


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

"""
Criteria for inclusion:
    Relevance:
        Channel found via the search method(s)
            1. Search by keywords on YouTube - using #notfinancialadvice, crypto to buy, stocks to buy, what to invest in 
            2. Snowball method from Google search of most important investing influencers on YouTube
        Video itself covers investing (any product)
        Video or channel over X views (X TO BE DECIDED ON VIEWING THE LIST OF VIDEOS)
    
    Pragmatics - must meet the following criteria:
        Transcript available on YouTube
        Between 2 - 10 minute videos (may also do shorts if possible)
"""
# Usage
keyword = 'crypto advice'  # Replace with your desired keyword
# start_time = time.time()
# video_ids2 = search_videos_by_keyword(keyword)
# end_time = time.time()
# print("Elapsed time:", end_time - start_time, " seconds")

# Print the video IDs
# print(video_ids2)


""" Part 3: Scraping the data """

"""
What data to include? 
    Channel
    Date
    Title
    Description
    Video length
    Number of likes & dislikes
    Number of comments
    
    Transcript text
        Autogenerated / Creator written
"""
video_ids = ['3irypwLDtTs', 'SbMtY-aJjFA']
csv_fle = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/data_test.csv"

metadata = extract_multiVideo_metadata(video_ids, api_key)
append_metadata_to_csv(metadata, csv_fle)



""" Part 4: Creating the dataset """

