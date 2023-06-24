
""" Topic modelling of the YouTube videos

- start with shorter videos
- then longer ones """


""" 1. Importing the necessary packages """
import pandas as pd
import numpy as np
import nltk
import spacy
#nltk.download('stopwords')
#nltk.download('omw-1.4')
#nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
#nltk.download('punkt')
from bertopic import BERTopic
from umap import UMAP
from matplotlib import pyplot as plt
import plotly.io as pio
import re
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# printing output settings
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
all_video_df = video_data

test_short_video_df = short_video_df[short_video_df['Transcript'] != ''][['Transcript']]
test_long_video_df = long_video_df[long_video_df['Transcript'] != ''][['Transcript']]
test_all_video_df = all_video_df[all_video_df['Transcript'] != ''][['Transcript', 'id']]
print(test_all_video_df.head())

""" 3. Topic modelling """

""" 3 a) Data preprocessing """



""" 3 a) 1. Removing / Altering poor auto transcribed videos """
# s p 500 = s&p500
# [Music][Applause]


""" 3 a) 2. Chunking so that sentence embedding works on full transcript """
# Load english Spacy model
nlp = spacy.load('en_core_web_sm')

# Process the transcripts and track their corresponding 'id'
transcripts = test_all_video_df['Transcript'].tolist()
ids = test_all_video_df['id'].tolist()

# Initialize lists to store the extracted sentences and their corresponding 'id'
sentences = []
sentence_ids = []

# Iterate over each transcript and its corresponding 'id' and split into sentences
# for transcript, id in zip(transcripts, ids):
#     # Process the transcript
#     doc = nlp(transcript)
#
#     # Extract sentences and append them to the lists
#     sentences.extend([sent.text for sent in doc.sents])
#     sentence_ids.extend([id] * len(list(doc.sents)))
#
# # Create a new dataframe with the extracted sentences and their corresponding 'id'
# new_df = pd.DataFrame({'Sentences': sentences, 'ID': sentence_ids})
# new_df.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/output2.csv', index=False)

transcript_chunks = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/output2.csv"
transcript_chunks_df = pd.read_csv(transcript_chunks)

# Combining the chunks as the spacy model splits them too much. Although better than any other model tried thus far
# so I am using it and combining chunks to get chunks of roughly 200 words
combined_chunks = []
combined_ids = []
current_chunk = ""
current_id = transcript_chunks_df.iloc[0]['ID']

for index, row in transcript_chunks_df.iterrows():
    chunk = row['Sentences']
    chunk_length = len(chunk.split())  # Assuming tokens are separated by whitespace

    # If the current chunk length exceeds the desired limit, append the current chunk and ID to the combined lists
    if len(current_chunk.split()) + chunk_length > 200:
        combined_chunks.append(current_chunk)
        combined_ids.append(current_id)
        current_chunk = chunk
        current_id = row['ID']
    else:
        # If the current chunk length is within the desired limit, append it to the current chunk
        current_chunk += " " + chunk

        # If it's the first chunk for a given ID, store the ID
        if current_id == "":
            current_id = row['ID']

    # If the current chunk is the last one for a given ID, append it to the combined lists
    if index == len(transcript_chunks_df) - 1 or row['ID'] != transcript_chunks_df.iloc[index + 1]['ID']:
        combined_chunks.append(current_chunk)
        combined_ids.append(current_id)
        current_chunk = ""
        current_id = ""

# Create a new DataFrame with the combined chunks and their corresponding IDs
transcript_chunks_combined_df = pd.DataFrame({'ID': combined_ids, 'combined_sentence': combined_chunks})
transcript_chunks_combined_df.replace("", np.nan, inplace=True)
transcript_chunks_combined_df.dropna(how='all', inplace=True)
transcript_chunks_combined_df.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/transcript_chunks_combined_df.csv', index=False)

chunk_counts = transcript_chunks_combined_df['ID'].value_counts()
id_chunks_count_df = pd.DataFrame({'ID': chunk_counts.index, 'chunk_count': chunk_counts.values})
id_chunks_count_df.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/id_chunks_count.csv', index=False)

all_video_df_chunkcount = all_video_df.merge(id_chunks_count_df, left_on='id', right_on='ID', how='left')
all_video_df_chunkcount.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/all_video_df_chunks_count.csv', index=False)


""" 3 b) Document embedding"""
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings_1 = sentence_model.encode(transcripts)
# np.save('embeddings_1.npy', embeddings_1)
#
# sentence_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
# embeddings_2 = sentence_model.encode(transcripts)
# np.save('embeddings_2.npy', embeddings_2)
#
# """ 3 c) Creating the topic model """
# # Defining sub-models
# vectorizer = CountVectorizer(stop_words="english")
# umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
# hdbscan_model = HDBSCAN(min_cluster_size=20, min_samples=2, metric='euclidean', cluster_selection_method='eom')
#
#
# # Model building
# topic_model_test = BERTopic(
#     umap_model=umap_model,
#     language="english",
#     hdbscan_model=hdbscan_model,
#     vectorizer_model=vectorizer
# )
#topic_model_test.fit(docs, embeddings_1)

# topics, probabilities = topic_model_test.fit_transform(test_long_video_df['transcript_lemmatized'])
#
# topic_model_test.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/test_model1")
#
# topic_model_test1 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/test_model1")
#
# print(topic_model_test1.get_topic_info())
# print(topic_model_test1.get_topic(0))
#
# barchart = topic_model_test1.visualize_barchart(top_n_topics = 10)
# pio.write_image(barchart, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Bertopic figures/barchart.png")
#
# term_rank = topic_model_test1.visualize_term_rank()
# pio.write_image(term_rank, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Bertopic figures/term_rank.png")
#
# vis_topics = topic_model_test1.visualize_topics()
# pio.write_image(vis_topics, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Bertopic figures/vis_topics.png")
#
# hierarchy = topic_model_test1.visualize_hierarchy(top_n_topics=10)
# pio.write_image(hierarchy, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Bertopic figures/hierarchy.png")
# print('here')
