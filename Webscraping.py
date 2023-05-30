
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

        for item in search_response['items']:
            video_id = item['id']['videoId']
            video_ids.append(video_id)
            all_video_ids.append(video_id)

        if channel_id in video_ids_by_channel:
            video_ids_by_channel[channel_id].extend(video_ids)
        else:
            video_ids_by_channel[channel_id] = video_ids

    return video_ids_by_channel, all_video_ids

def find_channel_ids2(channel_usernames, api_key):
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
# api_key_old = "AIzaSyBRpuSMO306VzZkUGCNt06zIk7deIJk0Ec"
api_key = "AIzaSyBvqa2-cEtjDKTCZ47qQcVJQqY4wKk5kek"
api_key_2 = "AIzaSyCkMzXm8C6lhQJCJUJbWM57bPF3Bi5jO3U"
api_key_3 = "AIzaSyAOLyZ77wU_n_wBg22jcN_RF-QMXvg7azg"
api_key_4 = "AIzaSyDCxeqgw_SekMX_tnuJ2keIXIydChvMOZQ"

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
# channel_ids_username_crypto, channel_ids_crypto = find_channel_ids(channel_names_crypto, api_key)

## Stocks
# channel_ids_username_stocks, channel_ids_stocks = find_channel_ids(channel_names_stocks, api_key)

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
#video_ids_by_channel_crypto1, video_ids_crypto1 = search_videos_by_keyword_in_channel(channel_ids_crypto, "crypto tips", max_results=10, api_key = api_key)
    # Keyword: Crypto
#video_ids_by_channel_crypto2, video_ids_crypto2 = search_videos_by_keyword_in_channel(channel_ids_crypto, "advice", max_results=10, api_key = api_key)
    # Keyword: Crypto
#video_ids_by_channel_crypto3, video_ids_crypto3 = search_videos_by_keyword_in_channel(channel_ids_crypto, "#notfinancialadvice", max_results=10, api_key = api_key_2)
    # Keyword: Crypto
#video_ids_by_channel_crypto4, video_ids_crypto4 = search_videos_by_keyword_in_channel(channel_ids_crypto, "what to buy", max_results=10, api_key = api_key_2)

# Similarly I will hard code the outputs of the above so as to run the below without reconnecting to the YouTube API
video_ids_crypto1 = ['kFHdQFt8-n4', 'TMpULJwCLZk', '4PgIySbSm4g', 'GjSG-dk4m8c', '0jLv0brvAwc', 'Zc5dUsBTbpI', 'hAb_OTm-mGc', '7OAKcoCNaCU',
                     'AaNfHskwLdg', '4SJNH-UwCWg', '28vMbDcW_jA', 'm36eAoRIAXg', 'bPCZfLsd7fU', 'YX-RZKjGV-M', 'WAImJuW-3iA', '04TTf55Z0lE',
                     'zSGCtoO4VJU', 'ws-jEZa3CCQ', 'aCYItUDYQ4Y', '4FJ9RXHKWgc', '0DKVpnL3hUc', 'hcWhgDME2kU', 'qIY_6yPv0k0', 'KnoQdWZ_n18',
                     'V5wE0ML_WMc', 'Af4cKK_T2F0', 'I3sh-vfSbt0', '-qRpW20AB8k', 'Qf7qx-B3NHA', '72hPK8zUnn0', 'F4YFtNHfGrw', 'EADLE7kO4tA',
                     'BrRP-KxUnoU', 'lRIJIxfS5Ck', 'YPt41FQS1XE', 'tNsc4tdXnCE', 'amLBQZfIw3M', 'si5kq7TFjjg', 'yDXB1jafqg4', 'i83OuE-aXOM',
                     'dB3RL3DzvMU', 'fbiR8tVbjTQ', 'HhYsGjxYjbs', 'zBjbyew6oyc', 'embu6Vy-M78', 'ByvURTZLSf8', 'bzs8-7OqWSU', 'tXBrK8jSA9U',
                     '8PNWFdry6WA', 'p4UXdvAwmgg', '8fzV9bO56ow', 'QZQIt9ONwSk', 'Ir_Xx7GQmOs', 'l79506sWDIg', 'IGhqslEwuOI', 'gXQpX2kObJg',
                     'i5NTKh-MNH8', 'RI4n6tQims4', 'EMA84FdiBvY', 'U556vsDPFrs', 'hkUoj9IZ1bA', 'Aadox_2RbdE', 'o1f8lG5s7sg', 'bxJTklJpw8I',
                     'dHR-krnrh-Q', 'ziyaJLxVfmc', 'kjwGCajj7ZY', 'RP0Do5pX5gw', '6JHd9iCdJ2E', 'Lb-Ciq8QpFk', 'nOLYdBEB5Pg', 'S9urRPApxy4',
                     'A4eRXTsNx6I', 'lBNoQeTaGaM', '956VkPvtA7Q', 'F_dPFJVSH2M', '0rpMLAWe-OY', 'oH8vmDnxIK0', 'jqQ9ErbsV7w', 'LnGAWFtNr48',
                     'QpIz1XsIXdA', '0hFYA43boIQ', 'cXTaYTrRuv8', 'IwL85i9SURs', '6ajkRnW6Mm4', 'P-d4SAqiv3c', '15zQenUrMF4', 'cWlcillUtdw',
                     'LUqPcxi8ZYU', '_rZFSzbAqlw', 'Y0HHt72fogA', '7GD_D8Wnq-U', 'r3BbzmzA_30', 'g8AzyXxD_JI', '1mI1UFq7kJE', 'JhQXHAjnfjQ',
                     'i9ehhiOVnrA', '8m79DftD2s8', 'YYyCadWCLH0', 'vp4yqpLhJ5Q', 'jRb-mKiQ3Yc', 'DFUs98kOHQY', 'tyci0fbMEZc', 'kXwtonUJcFU',
                     'n1ttKgGdEGE', 'VOi4j57_clQ', 'U0qkxhdcqPE', 'YrnOxhI3lF0', 'iC4uhrvxSe0', 'lJRd1XwaIOg', 'atD3ik7fnI8', 'rrp8Undxcik',
                     'QxOcmOLkLx0', '-oDRqE4gKFU', 'wv4BzFeDF14', 'XW7FGmgS-9s', 'KDGfjaGehSo', 'o4a8l3NVH1I', 'JBTR45kRe9I', 'HoqpgER2ZGE',
                     'Gq3rem1jd-Y', '60_OaOOZaAo', 'Q-X4MgoKGPg', '1mnn2r1Ysv8', 'FCBgRc-v_mA', 'C8p8XoKzd4w', 'ZXHotricTTw', '2We6Z-v2GzA',
                     'JjCuxckpJ9E', '41Ggxvj0B8s', 'ZMghrnDpIRk', 'RabdrZT87oE', 'sncdDPGHmu4', 'eTa_w-KA5qQ', 'Kq6y7y57V7o', '3W04ra6vYmc',
                     '_qr6sNEAQsY', 'A_84dYLz_E0', 'nQRfsjJUzyk', 'zQMgwgDNNbE', 'QRpode16wnI', 'qDNQ-WTkwv0', 'uKp0G74aivM', 'iyL06tWsp6E',
                     'murb5ga-56g', '3n5HDH5TwF4', 'AbRFl9ZXyqc', 'aJ46nM45NL8', '-hJCcNsiBVQ', 'Y8lMZBy7ydo', 'DxZu6kUa5FM', '90hq6GIxJTw',
                      'Ui4RFGPIGDg', 'CpIW-Mtg5X4', '9ozbvy8AxuU', '2BntbmUnLaM', 'M0dix6hsf0M', 'dMjiR9B-ts4', 'PnFOPT4JWp0', 'EZl0ntQwsoI']
video_ids_crypto2 = ['kFHdQFt8-n4', 'g-Z-21qfqf8', 'Rm6TuE2pyjg', 'F_qkF_gSzv4', '5Xm_BbSZxUw', 'TMpULJwCLZk', '0Rs7o6oAMyQ', 'Ww5hAXoPiPg',
                     'WFrFp7ITsys', 'Mu9XNE_Uim0', 'OPCxzupYLzA', '-YG1o-xhX4k', 'jOOVrLrzZ7A', 'ws-jEZa3CCQ', '_5JULMp2Hls', '77o1LcX8T40',
                     '_1Sb7tmUXBI', 'JlAdCox7cI8', 'y2IWGE_HIFU', 'zqq8MjdjoIk', 'hcWhgDME2kU', 'KnoQdWZ_n18', '5C4LTWq_gTI', 'W7fumWmhiaA',
                     'VXnYYxI6xYA', 'AOUCI22GEW8', '4aQZYL6e3_0', 'vXS_lHLYnqA', 'ssVaf7-mfso', '6qBczZue7s8', 'lRIJIxfS5Ck', 'EADLE7kO4tA',
                     'F4YFtNHfGrw', 'oyFMzFfxTdo', 'LoWOO3GtUBY', '1lwN6iIIDps', 'BrRP-KxUnoU', '2GktMxR3wak', 'PDui3ReLeTE', 'If9r3IX-B6E',
                     'EO-wSgwn7u0', 'fmuwT2L3u4E', 'mDN94E2MRNM', 'cpT5Z-Nlc2E', '6acmEYT3adY', 'GOyBgum-cg4', 'iAhFhdSbtgg', 'b4tg5XiI5l4',
                     'kShqEyNib2k', 'XU-LKhm2WUU', '-2svLnsky0A', 'L4kldDbR6mI', 'iNwMSJza4Wc', 'tvRbVh4JCP4', 'l79506sWDIg', '8fzV9bO56ow',
                     'T_3dMKCFXf0', 'iMGJnaFLz-w', '_7Rph_Hm5lg', 'MlGa0fXIE-o', 'XaTtnusUmZ4', 'bxJTklJpw8I', 'xNrE2SqhECw', 'cq8DsOrfaiY',
                     'Nj96VsBgqbU', '2JXLkXI5b1I', 'o1f8lG5s7sg', 'gjyig2EVLL0', 'UUivm1DMdX4', 'gX70v3slPuA', '-1MlAXARobc', 'cY8FMSuLvVQ',
                     'KKw3rCCpf44', 'tNvgP2DFqaU', 'LpX93GvEvG4', 'zSTz8275_5s', 'eQgzpj8qauw', 'chRpx_yZSLw', 'm6kai8pwPz0', '_24BJuNpyzc',
                     '-jaOq0e8Xl4', 'eolCjGDRMCg', '1ZTn0sTooD4', 'PingosQVCAU', 'zrKmhJNzuS4', 'WrDcdfPHmkY', 'PQzTBvoQ8lU', '67iKCqo2RCY',
                     '2L3vvT4fHTM', 'zK2shli5UdM', 'f23uEgU_O8E', 'g41rqWduT6Q', 'PcA0mjL_oVg', 'ZPBhCda1SmU', 'KHHwJVrP68E', 'VDdPuo3Y-xk',
                     'DRiUSQ5lmtg', 'bImMlD5ucug', '76z0a8wrWYk', '3Ajh7-y59o0', 'Qguf03l0KaU', 'wf9zHLiDWRk', 'YYtgVvSnz_A', 'Xzd3X79pjAo',
                     'xb3ELyUB1Og', 'L8A6wMrCwXc', 'B5Dw5y7uKK8', 'ygDoXFDmLlQ', '1xKD8oRktt0', 'bxx0nL5doFQ', 'LqFyWxRnwiw', 'Q61Zl-99_Q4',
                     'fHb8RU_1qqk', 'I8cgRhNLzVA', 'gVjh-hVfcc8', 'HVkEnck2Vck', 'Bm4UKKrRvC0', 'dInqu3p7J0o', 'Tw4ot68cVzA', 'StJ3fDXWtAU',
                     'JtqtIw4CCug', 'F12MPTNNuo4', 'ANgnWlNbQjg', 'qntIHEXxG7M', 'Ist6VjzLVm4', 'SEwklu4t9Rg', 'QpZrOE2VDKQ', 'SLZ_3bdyO2M',
                     'AeSh3weZXJ8', 'xNZlo2EUfwQ', 'nNPK3yjEavw', '4kBxSzSK2Ts', 'gUVR6AzMtqI', 'UiByfkI2rjY', 'qDmwBAjwRSc', 'ZMghrnDpIRk',
                     'hvQXsewHSkE', 'PQPi_ewwe1I', 'hxsajHMvJoY', 'JGbO6zep5Bg', 'ihwI9nr2VeA', 'QRpode16wnI', 'IqOUtR6WcOk', '80zosJeVhU4',
                     'iyL06tWsp6E', '0bJ0ZEHd6ho', 'murb5ga-56g', 'GU01L6I0BsY', '0qw1mI1baDY', 'UZpJIre_LOc', '_RkyQFB-ETg', '1Boq66u_W6E',
                     '1I7Cg64gDS4', 'dEbbTTdWupY', '8YAMt5Tt24Q', 'F1uos4QIWNM', 'DxZu6kUa5FM', 'pG2zv-XowYs', 'rR7gKAZFtcI', 'PlVV8V6CFRI']
video_ids_crypto3 = ['WFrFp7ITsys', 'ZPH2ekOI5BA', '4-EgKFBmwKY', 'ofIm-w4OGCs', 'pA7kuLQN4AM', 'iVrrYhWSyx8', 'x2GPeDZOEyo', 'g-Z-21qfqf8',
                     'Uz56waLwUfM', 'q6HfyuZ4wrQ', 'jOOVrLrzZ7A', '-YG1o-xhX4k', 'OPCxzupYLzA', 'HtNDxhcbYLo', '5BQ7zaJC3B0', 'lJcPsVg3cUQ',
                     'YX-RZKjGV-M', '8ihE2q0QkU8', 'nAXJu2rlIxc', 'hEQPC9kLuLc', 'hcWhgDME2kU', 'MpOzD65quBE', '0DKVpnL3hUc', 'T8NKiePdkqo',
                     'pHKvTjNgqB4', '4X4R1luTsho', 'SbMtY-aJjFA', 'xobu78ax3GQ', '57G4L44Bc7A', 'UrVbekMqhG4', 'F4YFtNHfGrw', 'BrRP-KxUnoU',
                     '0Og5-OJ1x1s', 'If9r3IX-B6E', '1lwN6iIIDps', 'PDui3ReLeTE', 'lRIJIxfS5Ck', 'ETZscebFb58', 'gpnLjfnZ2qM', 'oyFMzFfxTdo',
                     'fmuwT2L3u4E', 'EO-wSgwn7u0', 'X0AjxdE2ZsA', '0dph0dUMWEo', 'LrouAGf00Sk', 'FOIy_kPlu5M', '9uwOjXd1OH0', '6HwF3ejSpIU',
                     'iAmbpMXDh94', 'SZPGD0rOIRI', 'tvRbVh4JCP4', '-2svLnsky0A', 'L4kldDbR6mI', 'iNwMSJza4Wc', 'skVvlt7TicI', '8fzV9bO56ow',
                     'MlGa0fXIE-o', '_7Rph_Hm5lg', 'l79506sWDIg', 'iMGJnaFLz-w', 'Nj96VsBgqbU', 'h5ZDjK1ad8o', 'hA0jafKpRkc', 'o1f8lG5s7sg',
                     '2JXLkXI5b1I', 'cq8DsOrfaiY', 'p0YyRIj17_M', 'roXACxVWjIk', 'u9VDqRGSEpE', 'FSRRKFuoD-8', '-1MlAXARobc', 'cY8FMSuLvVQ',
                     '-qhTxdKHlbY', 'WJl2SodWhqc', 'cLYCNFV69s4', 'xA-A6JynmOE', 'bn8X4HPvhQQ', 'RDn0UfDb2GY', 'ceob8tvCVaw', 'BDDAvsV7-l4',
                     '1ZTn0sTooD4', 'PingosQVCAU', 'WrDcdfPHmkY', '-jaOq0e8Xl4', 'JLWwdUSKYbQ', 'zK2shli5UdM', 'gnEVwG_M0ac', '0hFYA43boIQ',
                     'rIzuDm3n0DI', 'HmXYJbLXPXg', 'f23uEgU_O8E', 'g41rqWduT6Q', 'PcA0mjL_oVg', 'VDdPuo3Y-xk', 'DRiUSQ5lmtg', 'KHHwJVrP68E',
                     'AKI64v-7Ko8', 'ZPBhCda1SmU', '76z0a8wrWYk', 'WghckXE4hQ0', 'u8blZPDgwk4', 'wf9zHLiDWRk', 'd8vwSALYQaM', '-ERngk9mCxM',
                     'RwpsBA0dmio', 'U0qkxhdcqPE', '_w8UlTO97eY', '4hdgGfPkTkE', 'RPXMc36_d48', 'eJ5WgwcE7Kw', 'LqFyWxRnwiw', 'Q61Zl-99_Q4',
                     'Kl3nH4qqsis', 'rVc3XN5BdPo', 'b7h7yGm2G3A', 'Ojrkuq77HFE', '6OQMgHSESpA', 'Sq2AIgRN1Lw', 'exIwwkKxQJs', '6JX2eN9eYl8',
                     'ANgnWlNbQjg', 'qGc8-jDnk0c', 'Ist6VjzLVm4', '7_2Bmgqt95M', 'QpZrOE2VDKQ', 'XwyxQjBTqIs', 'gq9T2vLiDig', 'C37yyjTLSio',
                     'Z0NMPUv6fcM', 'hgt_FVi63Vg', 'gUVR6AzMtqI', 'nNPK3yjEavw', '4kBxSzSK2Ts', 'UiByfkI2rjY', 'qDmwBAjwRSc', 'OGd3uz4AO7s',
                     'ZMghrnDpIRk', 'hvQXsewHSkE', 'PQPi_ewwe1I', '6kdIQFe0lv0', 'ihwI9nr2VeA', 'GU01L6I0BsY', '0bJ0ZEHd6ho', 'lH3ONqLW4-E',
                     'qCATotC1v-A', 'iyL06tWsp6E', 'IqOUtR6WcOk', 'QRpode16wnI', 'uKp0G74aivM', 'qDNQ-WTkwv0', 'EsnpRGvb4K0', 'qp4Aalv-Av4',
                     '8Xr-dS8sdSE', 'z4ra-cLXGZ4', 'WMUyhTfppNY', '2OAK2pYOMxw', 'FlwhiNckwRQ', 'LR9Ign86zZw', 'z3rZxTPw2VM', 'SzlO6x6imr4']
video_ids_crypto4 = ['L4869xRVM3E', 'D51Uf078d0w', '50tYOPvqA5c', 'C814v3HT1To', 'BcGd0DyRm58', 'LO5XrVbt-fc', 'oSx1pKSrAXk', 'NTBWt6AlDX4',
                     'pnxVSFk_jZc', 'dfhCgICxscY', 'HtNDxhcbYLo', 'jOOVrLrzZ7A', 'WAImJuW-3iA', '3EieOrx5r5E', 'fCWcfYoDPFk', 'oVEfVR_KxTk',
                     'oRSsr6VnOW4', '8Eosysbqiac', 'XNR-NNUh38w', 'KWaiAI6ZFrE', 'eNoLBYDf2wc', '0DKVpnL3hUc', 'SbMtY-aJjFA', 'NiuI9o09DWI',
                     'NZNuXWsdfCY', 'dHy86IAAFXU', 'hcWhgDME2kU', 'BpX1CTbwX6E', 'RhfKaRAhIRc', '4jeaBIMnp1U', 'UK6LKszd73M', 'Prkv45GfGUg',
                     'lRIJIxfS5Ck', 'v7LyUzKWGfQ', 'F4YFtNHfGrw', 'EADLE7kO4tA', 'UaGiOdogC7o', 'XSJnX9oe_is', 'rmDPFHdE448', 'si5kq7TFjjg',
                     'DJnv6tbbK3A', 'qXWsTozY3k0', 'OVlQ8Pgcq8w', 'uIIR5XG0Luc', 'dLLUIeHH-3A', 'nFU019GGTxY', 'fbiR8tVbjTQ', 'mDN94E2MRNM',
                     'NJ3oHc_lzws', 'tvIYRGdZwdg', 'IRfUsVmpsfs', 'l79506sWDIg', 'iMGJnaFLz-w', 'L4kldDbR6mI', '8fzV9bO56ow', 'iNwMSJza4Wc',
                     'QZQIt9ONwSk', 'kePhyClKtU0', 'skVvlt7TicI', 'dgpz5fckgec', '0liZ8jG3yUY', 'o1f8lG5s7sg', 'BLdRGPDOqJw', 'T_v4pW9o7a4',
                     '5CySDSRCWhY', 'Yn4sKVdJiUU', 'R_8_ESg2Q3Y', '-ZP59yjtV5M', 'ueDPXW4dSaw', 'RSEJUFS4S4w', 'LxqIKK0NRaI', 'cY8FMSuLvVQ',
                     'psa3mZFj4Gg', 's1TOtBPFipk', 'cG1cSY-fjEs', 'AkxAdYKBUEg', '7ttJPvqKBDg', 'YjFOj1M7ZOo', 'cvPUX_K2NTs', 'ec6tfiwBQYU',
                     'JLWwdUSKYbQ', 'uxxf4h-hwIs', '1P2u-PbXr_Q', 'cXTaYTrRuv8', '5gs8vwQnInI', 'WTTmbjj0Z0c', 'pMBLUCxzTUM', '8UiAGkeqNwQ',
                     'Uby6wu8I0e4', 'g-CG3aOXRIY', 'PcA0mjL_oVg', '7GD_D8Wnq-U', '76z0a8wrWYk', 'TwY9CZahsxU', 'HuF17vCq70E', 'jXbMAk6dsk0',
                     '1mI1UFq7kJE', '-uIgcOKmEhs', 'VDdPuo3Y-xk', 'ZPBhCda1SmU', '2Qv9lFfLYDI', 'xcRVqogMCRg', 'SXspRnb-xJo', 'oA2L_22Y6z0',
                     'CQRZKYjO6uw', 'JY3l4VwcNrw', 'wmZQGy1GWQ4', 'm501gRu_nik', 'eCDsiwnPGMk', 'g55JAdwVs2Q', 'fHb8RU_1qqk', 'Q61Zl-99_Q4',
                     '_zl0R5tMH5U', 'Kl3nH4qqsis', 'Av9ypMylguE', 'zF-FZfl1d60', 'KBzQgKxcHbE', 'rKMkYsQuB-s', 'wv4BzFeDF14', '9W0uBdZkON4',
                     'Ir8oIRgs-Zs', 'wrcsRKlCX_M', 'YQiJ7mMvHow', 'goVx267uuqw', 'gq9T2vLiDig', 'LSkEWrCT-eo', 'Z0NMPUv6fcM', 'n6P6o-say-0',
                     'mSCE7HX45_A', 'Ist6VjzLVm4', 'gUVR6AzMtqI', '4kBxSzSK2Ts', 'LKtA-x3J12Q', 'lI7q_9ZvDGQ', 'XzUE4pUguAw', 'S3p1oKAeZvY',
                     'ZMghrnDpIRk', 'vlvMYOhM-LE', 'nNPK3yjEavw', 'ZCoetv12AT0', 'G34fTJHXMTk', '3n5HDH5TwF4', 'lH3ONqLW4-E', 'QRpode16wnI',
                     'IqOUtR6WcOk', 'fKWLZHg6m_I', 'VIzHL1RNc6E', 'TrKRC7Y56uo', 'uKp0G74aivM', 'KTJcL6WJfy4', '406jWUWCAWk', 'on8OzKfeu58', 'm396VLNXmnU',
                     'LzcL9q8y88o', 'txzxXFKe6DU', 'aNr2cx9d6s0', 'Az251B6logo', 'IPld39yx0tY', 'IxfgrEaO50c', '8LNyH1ITSdY', 'RSMnV9f5pwg']

#video_ids_crypto = video_ids_crypto1 + video_ids_crypto2 + video_ids_crypto3 + video_ids_crypto4

## Stocks
    # Keyword: Stocks
#video_ids_by_channel_stocks1, video_ids_stocks1 = search_videos_by_keyword_in_channel(channel_ids_stocks, "stock tips", max_results=10, api_key = api_key_2)
    # Keyword: Stocks
#video_ids_by_channel_stocks2, video_ids_stocks2 = search_videos_by_keyword_in_channel(channel_ids_stocks, "investment advice", max_results=10, api_key = api_key)
    # Keyword: Stocks
video_ids_by_channel_stocks3, video_ids_stocks3 = search_videos_by_keyword_in_channel(channel_ids_stocks, "#notfinancialadvice", max_results=10, api_key = api_key)
print(video_ids_by_channel_stocks3)
print(video_ids_stocks3)
print("hello")
    # Keyword: Stocks
#video_ids_by_channel_stocks4, video_ids_stocks4 = search_videos_by_keyword_in_channel(channel_ids_stocks, "what to buy", max_results=10, api_key = api_key)


video_ids_stocks1 = ['dV00JxVDBdk', 'hjm4t4F-1AY', 'x6EyI_hLtJg', '78LpNG_rbI8', 'CWcD7JF8ebw', 'RBaxlBu8fMM', 'k-MQaDfd250', 'MFoL_ljkTNo',
                     '9hHst4KVwkI', 'nQAEmbkqiOA', 'OIXvIvDl2fM', 'dyV1NYbk_nA', 'cjXQPIb_xtM', 'J1JmNaqLzNM', 'tdjVVsVtXi0', 'PLfJkb5Ydbw',
                     'bXOXc49IbCI', 's8wC6U7QJmQ', 'fHXPu-EDb6M', 'jmoOrgTP5XQ', '5X_ZcifasBg', 'uJy9XIJq1cQ', '1axXKjgCe84', '32QhRORiqGk',
                     '8h1xWOkgrpI', 'JMcaRfFThmg', 'I3FFdGMNe58', 'xn_Nxa_0JBU', '9greKFrOY94', 'N3E5eYS8d-I', '-E91e3al4gg', 'm00G4m1XEVI',
                     'D60pgI8hVm8', 'yeCwcJuesTc', '3OvpXQGoTHY', 'isjHU3MzO5U', 'BbSlcEUr1Ow', 'OpmMhTUJMnY', 'wqC3mWP7zAw', 'wWeECooKbbc',
                     'uWmPzNsgDZs', 'j38ZXGcZ9u4', 'KdZNiVFlQIs', 'Y6HfXpqtvdI', 'fYaqitj2Ics', 'gxlhsuWLIoI', '8yTK6Tok7EU', 'ysv2BwHEy8o',
                     'pNIZw_toUIo', 'SHwDePphnCo', 'bsuespwe_uY', 'IRavvVM6AVs', 'WwotytnkYRQ', 'rxU6ze-NluI', 'jLcnYd6Ki0E', 'A5DAoRLDN0w',
                     '32w1xnOqSPo', 'hfmloFUxJ1Q', '8J_mscS7MnY', 'iQPU1Pyic4U', 'hEzDxnSUlIA', 'pVjDriSvN-U', 'H1XbtWdhZ5E', 'rEnGNbcySj0',
                     'ANGeuB0fQcs', 'WMRDuAdk7q4', 'P3oXSKZXfXA', '2VQp6-alQMg', 'VpCuDb1H-nc', '-kHLDxOqguQ', 'rtgIVfezB-U', 'Cc_sauhqOLs',
                     'oc4rLzU4OKo', 'CRAMVf35SCs', 'ikh2xbbHvqY', '864d8f0mBAY', 'fMhMT0hz3EA', 'N9LvWy1IGPY', 'Rn2BkivJpl8', 'ExsfiMK6bhE',
                     'SrQiOg0TV20', '1kD66RMWimo', 'EPEud3AVAGQ', 'jye4aTRjhnw', 'LhEzhflVw9g', 'ECgwpqVYB78', 'GdRXeJxbT6M', '4LZilvA1u0Y',
                     'sP_XnMjJxY8', 'm6NgMXUVlq8', 'izViFORf7SQ', 'rKbhTSzBGRI', 'rr3p4os5SyQ', 'KOi44KdrlxA', 'd57G0R5aldU', 'TRF_ryFg06A',
                     'J_EAo5U9izg', '4LqvOL7kMvY', '7PPwcamumT4', 'hx1CzMH0cHs', 'jqsP5O3udv0', 'dWWXHs8gLKU', 'DRmDZ_rqD-A', 'WPc9_iFzScs',
                     'twgm08WOhnM', 'FAvldninROA', 'Vr4m5X4FMfE', 'Oew3VnE-UD8', 'iV6rkpJveBg', 'JahHRqoZaHg', 'haUujRwCUl4', 'uCdK7YNijX8',
                     'jjTmsbz_ydE', 'tlGqzT843k0', 'ka2TZM94EtM', 'hmNfjjEhcNY', 'A1Ntf1f-bzg', 'STMK8L199pI', 'URES59zcNyg', 'vPTfaahHkGI',
                     'ZeLXFbOmprc', 'a_OTAxe9xFk', 'rPmnMKZfx0g', 'iuM1DOysNd4', 'lvlMlf_5yOs', 'qfRAlbWvThA', 'IQMl9-MxcoE', 'Bp8KCAQ6J50',
                     'C9g8E8QTu1A', 'jGRDSoOuqiY', '3WGPa3BnlbQ', 'nb8zn_25dPU', 'AAIrhCsWNSk', 'OP9sSyULSww', 'Wsg3N82YJs4', '28aZp3n6mmg',
                     'WNj4hHxWj2g', 'Cy-jss4E52A', '05dHr2LWjsU', 'keikMNOJ5ho', '_dp1bPeO2dY', '_gUMPQZS7b4', 'ntX9bHqygrU', 'KJgUTAbTXOk',
                     '6DXbZa-lfr8', 'rAWq82IEgkc', 'fUD55CfhH-Q', 'Vvby9pR6Q7I', 'TWIH22YAcrk', '-nSODrSo7nY', 'mHtMxxBsdR8', 'VErctXiMaWI',
                     'P-XB0ZdE3aQ', 'uFdtO4bVtI0', '8Jq1DjR1RdM', 'kHlP1-H0nsM', 'L76Kd19NZT8', 'Uxzqfb6unA0', 'f26X-L9P4ig', 'f5NZZ-H98I4',
                     'R0pLsJOjGXM', 'P8upKEsCyU4', '5_3Ra-Q6vK4', 'VmdncOfNWas', 'Mc3Jxx23hbo', 'HBy-xP5aQyA', 'tD3BpmsPPjE', 'JjdDUanjGiY',
                     'PnSC2BY4e7c', '8JOo4L2w43Q', 'udHzPjxq2lg', 'vSCyfXdl7VE', 'nY234RoQeHw', 'aUbBfolWaYU', 'hWwTzqCmdI4', 'GgXNedPCej8',
                     'okeg4GhPxjE', '2I_GZebHd8Y', 'KGnNKnLL_b0', 'o_8HxR6K5gY', 'ApMMSacmwXI', 'lFyeLeP44kY', 'HvjSRskN_kU', 'gQqgJJKYYZk',
                     '5rN22nIRh0M', 'Mxu85SCT9lQ', 'ihf5cn5JP8A', 'vVv-RKRuJzk', 'juXE9-vbSyk', 'IXsb3aoXbVk', 'NskNfAlD3wU', 'qze8YAjya-E',
                     '87Q6l9jtumQ', 'ZocOzdVvNvE', 'HOeL03zCcs8', 'PW2jGN6tW1o', 'nzU6n0da4as', 'RbyCGfw-L6U', 'MPj17ot8qH0', 'LIG1kSplrYM',
                     'GuE_MpyMlx4', 'e4W9RzjP4t4', 'vypTRO85GhI', 'V5wxOaenRRs', '86PFzsJRp9U', 'P_Kde4dAcNY', 'xRPBoOVZp7A', 'IoJLJ2D2Ul0',
                     'ELArJhtGh40', '9kedsUW0ypk', 'YU5VVAfd2j0', 'Bs8vu4CEARU', 'pdRjwXC7moA', 'gSLuzEWIIXg', 'c-psoassiY8', 'fOJns7JUY4o', 'SYfVzR5cB7I',
                     'jACDruSwi1A', '-Fy1KMt6miI', 'RzSSI_XptKk', 'pQxCiu_CxMk', 'ZoLKXrAIa_g', 'rxpOQkDozdQ', '9DKAuxO25ss', 'uEyJidRH9Go']
video_ids_stocks2 = ['iJFgMkB_ARw', '78LpNG_rbI8', 'CwTQH_clH6Y', 'dV00JxVDBdk', 'x6EyI_hLtJg', 'aP0Sy9MOmFs', '-p2p2U_L4MA', 'HlnalLo0SjE',
                     'Uf4XG3UTBaY', 'CWcD7JF8ebw', 'AXogsfz4gqE', 'PosvpZRMIjA', '8GWVoduDOGM', 'oDdMPi-S_u0', 'Yr-C3Gn1Jhc', 'fHXPu-EDb6M',
                     'Ziv9DUKSrHs', 'r1vxSFJQpPg', 'prRpgXVMmWY', 'V1B3uhXfxLA', '5X_ZcifasBg', 'y3USmdoXKt8', 'AfTzD9VHdi4', 'g8e9ul0Sxxc',
                     'Fe4BaF8gCOQ', 'nm-rhysU96k', '32QhRORiqGk', 'DgV6r0yKyJw', 'ZGap7kb14gI', 'NWeH-uOyTyI', 'J87kGYGDN6A', 'D60pgI8hVm8',
                     'isjHU3MzO5U', 'yeCwcJuesTc', '6auT5dNCT9I', '3OvpXQGoTHY', '4A2xq0MaIhI', 'alsU4rXLN6c', 'l1DE5WEie-A', 'U4a7mWvOaso',
                     '8yTK6Tok7EU', 'uWmPzNsgDZs', 'kBuKX2orkgA', 'j38ZXGcZ9u4', 'CAE2_cpB-ic', 'YYHzkVVSpAo', 'pNIZw_toUIo', 'fw-qUooWejA',
                     'vyUNHmluPlg', 'koe_55782q8', 'rxU6ze-NluI', 'eVgFzmC2lAI', 'IRavvVM6AVs', '2aCRnWpvl5A', 'bku0ailQfl8', 'HpRws1GLmPw',
                     'm2TOm6b-J_o', 'rUFAEslz84k', 'bsuespwe_uY', '5w6GLvaAdCQ', 'hEzDxnSUlIA', 'mHTmYXGMalE', 'wt2wwoCfPSc', 'pVjDriSvN-U',
                     'STjYPcVmkoA', 'zbTn6HFaZok', 'CTWUmR3g2VQ', 'HhIbxORFyCU', '4G6cZrtVmuQ', 'j5pQOb5TXy0', 'dCgqHKLyeCA', 'eLRzzN9qUZ0',
                     'bvHanrTJ6JQ', 'oq_Uu1nWFnI', 'mjNj3rX4wCo', 'aPPnj8NR3vQ', 'jTmPD-ksahU', 'VpCuDb1H-nc', 'Qqzb5NxOGm0', 'df9HYIwy1Ro',
                     'GdRXeJxbT6M', 'm2db-PpRsq8', 'ZjIyk5Op93I', 'Rn2BkivJpl8', 'h65R-XkAHjg', 'FPOpBDKjtYk', 'nWo8a6Pdv-g', 'LCiHGuTIuLk',
                     '1f7ZCdsciDs', 'Z8ycYGNFFZA', '_N6QFnkS7Ns', 'U7SnnM9FxQ0', 'rr3p4os5SyQ', 'm6NgMXUVlq8', 'HycMHz5Qp04', 'SPa1zF2Nptc',
                     'z9bgRtuBRJg', 'AdTDrN21J8I', 'Iltkz-96CZc', '9Y00mp4TbIA', 'dC9JeAko-Lo', 'XLj44NGugUQ', 'nIu1DgllITg', 'KkDOKHG0Hak',
                     'CSyg75XR1oE', '0a-ZG2tmCI0', 'Vr4m5X4FMfE', 'U78GKrcH0V8', 'PsIQ1DaHRl4', '_cHa-NoQLtc', 'IV9OrXDiouc', 'JahHRqoZaHg',
                     '9FAVM8NR9zE', 'tlGqzT843k0', 'omuMbgHjH7k', 'Ft6bQI2M37s', '2jCPhSZM2Qc', 'AmogriLhOFE', 'IFja5oYLdww', 'D6dV3pH01Ys',
                     '0C-_N-5VlR8', 'URES59zcNyg', 'XHxEgtJ-i2c', 'JY1k0D6O8C0', 'IQMl9-MxcoE', 'PnTR6iAQFnA', 'lvlMlf_5yOs', 'w-lFAKuXMfk',
                     'fbyzQKKEszs', '7poZYLWWkkw', 'M591pP9tJIk', 'HdHnrFH79T0', 'slprFkr0S7k', 'nb8zn_25dPU', 'C9g8E8QTu1A', 'wl1q-iMZf34',
                     'jeW41cEQVCM', 'XmUMcRVN5O8', 'Cy-jss4E52A', 'SM7_7Ht-MGI', '05dHr2LWjsU', 'TGlo4NHeMTM', 'ntX9bHqygrU', 'P9gRYjwcDQw',
                     'CvZ9HBkS27k', 'CVeNuRzhq9U', '_gUMPQZS7b4', 'fUD55CfhH-Q', '_pXH9hS0orY', 'T0yKl98PCCM', 'LMTV6Z58hu0', 'JdwgN5H3MBw',
                     'aXW2MqrqZU0', 'w45SXUzZ3ro', 'mHtMxxBsdR8', 'b-OkhwQO1wU', 'T9nNMMfQchc', 'dO9jKv3ryuk', '6aHbIaQAxXg', 'b380H8wqUnE',
                     '8Jq1DjR1RdM', 'L76Kd19NZT8', 'eVxeoKf0f7w', 'nfueJfuqq8c', 'Gii-o98VwIs', 'sOw_xpfCnxI', 'orbKC1BWuMc', 'h31pnkNK7Bs',
                     'WPxP2heADjQ', 'oBGkjWKDfes', 'P8upKEsCyU4', 'VmdncOfNWas', 'j1xAthFAd-0', 'EKIhc3tS0os', 'QLtvIXc6PWQ', 'TPX_OTvGk9k',
                     'Mc3Jxx23hbo', 'BougQAMldsM', 'qDkOE1RITLw', '7C6roAxktuI', 'vSCyfXdl7VE', 'nY234RoQeHw', 'aUbBfolWaYU', 'hWwTzqCmdI4',
                     'ryQsJm1a5qc', 'fMkOtRGz5EI', 'X6uQomdxDlI', 'GgXNedPCej8', 'b2-wP0UEERc', 'pnIZLvP-7N4', 'HvjSRskN_kU', 'L8W1tbEVyOM',
                     '8c-gU1A_Bho', 'IXsb3aoXbVk', 'NskNfAlD3wU', 'lFyeLeP44kY', '5rN22nIRh0M', '_Nb6OKOM_nw', 'CRt135W9Imc', 'tYfuEF9IYuA',
                     'tS01dA9LaqA', '87Q6l9jtumQ', 'qze8YAjya-E', 'RbyCGfw-L6U', 'Mbd7koFnlaY', 'PW2jGN6tW1o', 'kaPOVR1W5zI', 'aQ26xfoT-3E',
                     'sm9m1pwatcI', 'DU5zuwuXWOk', 'WGf_RE5Th_U', 'JvR9aC_y6Ts', '_bheTToKZIg', 'ojoVbLvtwUI', 'YTALNVYZ_Y4', 'Rf-yQYoP8mY',
                     'H7JxVfBV9Q4', 'sLwF7MynC9E', 'Sov0IscJq9Y', 'QWcnLw3pVdQ', 'IICdGHn9tSo', 'vqlSKj97OhM', 'a3JV5FaRods', 'yz4DYxbfHmk',
                     'lgdVN2jc09A', 'r-epU06B2eY', '_Dl_DOLeEH4', 'sLvGrqqkEyw', 'NwjvubapZ54', 'EuPed8AwvnI',
                     'ZpAuGkoM5fs', 'LfM3A9_xoho', 'XQFf2T3G88M', 'rxpOQkDozdQ', 'ZFqSblxDzU8', '3FBcP9Ui2HQ', 'hhmdYKsH2b0', 'r8SUuudrP5E']
video_ids_stocks3 = ['dV00JxVDBdk', 'HlnalLo0SjE', 'CCjWpVGoVyQ', 'aP0Sy9MOmFs', 'sBn3rk2r19s', 'CwTQH_clH6Y', 'Qb_Em_x5Pf8', '-p2p2U_L4MA',
                     'h9cfkJ2npec', '9cuPzYAy6-c', 'psGxIB7H20o', 'prRpgXVMmWY', 'PosvpZRMIjA', '9Zyde4d1zo8', 'lNExnsbvSrA', 'TSxquI6ac3c',
                     'N-l9a4BmU4Q', 'hU7YZMwzZPY', 'Yr-C3Gn1Jhc', 'v3bk2MdJ9lw', 'ifAOlaKmuO8', 'g8e9ul0Sxxc', 'y3USmdoXKt8', '5X_ZcifasBg',
                     'AJwHtsHPgJw', 'yht8S-4TmZU', 'N3E5eYS8d-I', 'SrJ2vSYSFGc', '9greKFrOY94', '32QhRORiqGk', 'J87kGYGDN6A', 'isjHU3MzO5U',
                     'd32cvhffgco', '6auT5dNCT9I', 'l1DE5WEie-A', 'gBPn4wEKKrw', '-I33SHN44PU', '3OvpXQGoTHY', 'U4a7mWvOaso', 'yeCwcJuesTc',
                     '8yTK6Tok7EU', 'JW9J93hrDNE', 'fw-qUooWejA', 'uRq1HdL352E', 'zyA_aaSRrKE', 'pNIZw_toUIo', '5VpCuLGeUyM', 'ZWghr1C-jYw',
                     'ysv2BwHEy8o', 'gTrq65yuHeM', 'eVgFzmC2lAI', 'IRavvVM6AVs', 'rxU6ze-NluI', 'A8vFUo_Yjg4', 'rUFAEslz84k', '32w1xnOqSPo',
                     '5GvNYH4KhZQ', 'J23HvCzKM3k', 'p5wSRjaQsnE', '-N0ccyDr-ng', 'STjYPcVmkoA', '3irypwLDtTs', 'hEzDxnSUlIA', 'VYKXrL-rSOs',
                     'RV-CcaDDi48', 'mHTmYXGMalE', 'dvZ1mkBc79E', '7yhKpXKRPwY', '89pCfpw5OF0', 'izlDHnM5NDw', 'rjB_NBimSxU', 'dCgqHKLyeCA',
                     'mJdwwOjf6LI', 'aPPnj8NR3vQ', 'Mxdq8C4nyxE', '864d8f0mBAY', 'My9drgMEuro', 'mYwiUCg_ZiQ', 'mjNj3rX4wCo', '25WiPIndXtA',
                     'kJkxX7ltsRQ', 'RxtiePtxfzY', '7z0qM459Opc', 'HWl_H1p0QJY', 't1j5RVaMxaU', 'jITJURSEpds', 'gmPmVpQ20NE', 'NYJgS9UpudE',
                     'wXyOhbH01C0', 'yIZP92kGFi4', '_N6QFnkS7Ns', 'rr3p4os5SyQ', 'z9bgRtuBRJg', 'HycMHz5Qp04', 'sJK0o90NWHY', 'ZuGdtdsyxGA',
                     'SPa1zF2Nptc', 'XLj44NGugUQ', 'nIu1DgllITg', '_cHa-NoQLtc', '0a-ZG2tmCI0', 'pltfunL1UQM', 'kQFWr23DIJg', 'DPuQJNhwCOg',
                     'KkDOKHG0Hak', 'FAvldninROA', 'KY4k6SACiuQ', 'D6dV3pH01Ys', 'AmogriLhOFE', 'FkRQJ_OXQ-4', 'IbRyElTtWT8', 'x5cxA0s5JOg',
                     '_fHN4ZSFUYw', 'RA7Vb329aSU', 'JahHRqoZaHg', 'resUIggMjDY', 'xW1KvnMgaLA', 'XHxEgtJ-i2c', '0C-_N-5VlR8', 'J6PcJC7vXo8',
                     'UmEC9qQq0AQ', 'T2MZzQqdQiA', 'fFe3cSVnlmo', 'oWCN6ux_Yf0', 'URES59zcNyg', 't8hDFLksrHE', 'fGu7oWy6Cy8', 'slprFkr0S7k',
                     'wl1q-iMZf34', 'M591pP9tJIk', 'HdHnrFH79T0', 'oC7xkxTyNbY', 'V8TdCPGw0Sw', 'dx1Yay7lw08', 'XmUMcRVN5O8', 'imkhwaK-c0g',
                     'didMRUT06Jk', 'CvZ9HBkS27k', '_gUMPQZS7b4', 'CVeNuRzhq9U', 'bj8dp5Afd58', '_pXH9hS0orY', 'YnQbsuWC3F0', 'T0yKl98PCCM',
                     '24-j8QrbHv0', 'CqXtTAIllTY', 'bAWCs4rO8KE', 'LMTV6Z58hu0', 'JdwgN5H3MBw', 'T9nNMMfQchc', '6aHbIaQAxXg', 'mHtMxxBsdR8',
                     'dO9jKv3ryuk', 'aXW2MqrqZU0', 'b-OkhwQO1wU', 'Ds3voY0SOLw', 'X7OO85279C0', 'UtthZD7Ckck', 'D9CaxgnWAhQ', 'Gii-o98VwIs',
                     'KFOsNmektpk', 'h31pnkNK7Bs', 'eBWJwQG-J4c', 'fRdsJnr10xU', 'VxbEF_xhkz4', '-Ugc0dhjp_0', 'yhjC79Cb5Rs', 'TPX_OTvGk9k',
                     'j1xAthFAd-0', 'jIu6YiBsvOw', 'qDkOE1RITLw', 'P8upKEsCyU4', 'sd9w8Zc-QrU', 'VmdncOfNWas', 'aza0yR6uDgs', 't5w5mk1_K3A',
                     'QLtvIXc6PWQ', 'aUbBfolWaYU', 'nY234RoQeHw', 'b2-wP0UEERc', 'yrcf2CWrJqk', 'hWwTzqCmdI4', 'kQTI73v84m4', 'vSCyfXdl7VE',
                     'jiSel4EEonM', 'GgXNedPCej8', 'ryQsJm1a5qc', 'lYquwNlkcUU', 'zXPW0HgMdJI', 'cO_VL9fo4rc', 'At-cde5dhGk', 'L8W1tbEVyOM',
                     'Yi80jotgLWU', 'GII33cpSBnE', 'juXE9-vbSyk', 'Sn4w2g5yJyI', 'CRt135W9Imc', 'u4PLpO_S5po', '7GrB2b0pS6w', 'qze8YAjya-E',
                     'KSBket6fWuQ', '4DhCKlWWAjE', '87Q6l9jtumQ', 'l9Nu2e9znJ4', 'nzU6n0da4as', '3VWc1v_JQs4', 'tS01dA9LaqA', 'Rf-yQYoP8mY',
                     'E516JZbNFKE', 'M7g2j7YXEdU', 'afB0Q6kB-40', 'DqejEplBQ38', 'YTALNVYZ_Y4', 'aDmOxuJE78A', 'Fi26SKWTKUc', 'CItnQXGj3fM',
                     'cEATlnPfK4Q', 'IICdGHn9tSo', 'vqlSKj97OhM', 'a3JV5FaRods', 'sLvGrqqkEyw', '_Dl_DOLeEH4', 'lgdVN2jc09A', 'EuPed8AwvnI', 'NwjvubapZ54', 'akFGfF0pacg',
                     'BwyeJbvmP9w', 'ZpAuGkoM5fs', 'LfM3A9_xoho', 'XQFf2T3G88M', '3FBcP9Ui2HQ', 'ZFqSblxDzU8', 'hhmdYKsH2b0', 'r8SUuudrP5E']
# video_ids_stocks4 =





#video_ids_stocks = video_ids_stocks1 + video_ids_stocks2 + video_ids_stocks3 + video_ids_stocks4










""" Part 4: Scraping the data """
## Crypto
#metadata_crypto = extract_multiVideo_metadata(video_ids_crypto, api_key)

## Stocks
# metadata_stocks = extract_multiVideo_metadata(video_ids_stocks, api_key)

## Overall
# video_ids = video_ids_crypto + video_ids_stocks
# metadata_overall = extract_multiVideo_metadata(video_ids, api_key)

""" Part 5: Creating the dataset """
## Crypto
csv_fle_crypto = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/crypto_test.csv"
#df_crypto = append_metadata_to_csv(metadata_crypto, csv_fle_crypto)

#print(df_crypto.head())

## Stocks
#csv_fle_stocks = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/stocks_test.csv"
# df_stocks = append_metadata_to_csv(metadata_stocks, csv_fle_stocks)

# print(df_stocks.head())

## Overall
# csv_fle_stocks = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Data/overall_test.csv"
# df_stocks = append_metadata_to_csv(metadata_overall, csv_fle_overall)

""" Additional helpful code """
    # Timing of functions
#start_time = time.time()
#end_time = time.time()
#print("Elapsed time:", end_time - start_time, " seconds")