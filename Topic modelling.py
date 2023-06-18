
""" Topic modelling of the YouTube videos

- start with shorter videos
- then longer ones """


""" 1. Importing the necessary packages """
import pandas as pd
import numpy as np
import nltk
#nltk.download('stopwords')
#nltk.download('omw-1.4')
#nltk.download('wordnet')
#wn = nltk.WordNetLemmatizer()
from bertopic import BERTopic
from umap import UMAP

""" 2. Reading in the files """
video_data_path = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Final dataset(s) for analysis/cleaned data.csv"
video_data = pd.read_csv(video_data_path)

long_video_df = video_data[video_data['Source'] == "overall_long_video_data.csv"]
short_video_df = video_data[video_data['Source'] == "overall_short_video_data.csv"]
freeSearch_video_df = video_data[video_data['Source'] == "overall_free_search.csv"]

test_short_video_df = short_video_df[short_video_df['Transcript'] != ''].head(200)[['Transcript']]
print(test_short_video_df.head())

""" 3. Topic modelling """
# Test on 200 short videos
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=100)

topic_model_test = BERTopic(umap_model=umap_model, language="english", calculate_probabilities=True)
topics, probabilities = topic_model_test.fit_transform(test_short_video_df['Transcript'])
