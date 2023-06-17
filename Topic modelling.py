
""" Topic modelling of the YouTube videos

- start with shorter videos
- then longer ones """


""" 1. Importing the necessary packages """
import pandas as pd

""" 2. Reading in the files """
long_video_path = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Overall datasets/overall_long_video_data.csv"
short_video_path = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Overall datasets/overall_short_video_data.csv"
freeSearch_video_path = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Overall datasets/overall_free_search.csv"

long_video_df = pd.read_csv(long_video_path)
short_video_df = pd.read_csv(short_video_path)
freeSearch_video_df = pd.read_csv(freeSearch_video_path)


