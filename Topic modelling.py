
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
wn = nltk.WordNetLemmatizer()
from bertopic import BERTopic
from umap import UMAP
from matplotlib import pyplot as plt

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth = desired_width)
pd.set_option('display.max_columns', 10)


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
stopwords = nltk.corpus.stopwords.words('english')
remove_from_stopwords = ["should", "shouldn't", "must", "mustn't", "need", "needn't", "would", "wouldn't"]
for word in remove_from_stopwords:
    if word in stopwords:
        stopwords.remove(word)
print(len(stopwords))

# Remove stopwords
test_short_video_df['transcript_without_stopwords'] = test_short_video_df['Transcript'].apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in stopwords]))
# # Lemmatization
test_short_video_df['transcript_lemmatized'] = test_short_video_df['transcript_without_stopwords'].apply(lambda x: ' '.join([wn.lemmatize(w) for w in x.split() if w not in stopwords]))
# # Take a look at the data
print(test_short_video_df.head())

# Model building
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=100)

topic_model_test = BERTopic(umap_model=umap_model, language="english", calculate_probabilities=True)
topics, probabilities = topic_model_test.fit_transform(test_short_video_df['transcript_lemmatized'])

topic_model_test.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/test_model1")
print('done')
print('\n')

topic_model_test1 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/test_model1")

print(topic_model_test1.get_topic_info())
print('\n')
print(topic_model_test1.get_topic(0))
print('\n')
barchart = topic_model_test1.visualize_barchart(top_n_topics = 5)
plt.savefig("barchart.png")
print('\n')
print(topic_model_test1.visualize_term_rank())

print('\n')
print(topic_model_test1.visualize_topics())
print('\n')
print(topic_model_test1.visualize_hierarchy(top_n_topics=5))
print('\n')
print('here')
