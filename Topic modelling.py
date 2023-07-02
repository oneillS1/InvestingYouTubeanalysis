
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
import time
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import ZeroShotClassification
from gensim.models.coherencemodel import CoherenceModel
import bokeh
from bokeh.io import export_png

# printing output settings
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth = desired_width)
pd.set_option('display.max_columns', 10)


""" 2. Reading in the files """
# video_data_path = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Final dataset(s) for analysis/cleaned data.csv"
# video_data = pd.read_csv(video_data_path)
#
# test_all_video_df = video_data[video_data['Transcript'] != ''][['Transcript', 'id']]

""" 3. Topic modelling """

""" 3 a) Data preprocessing """

""" 3 a) 1. Removing / Altering poor auto transcribed videos """
# # s p 500 = s&p500
# test_all_video_df['transcript_processed'] = test_all_video_df['Transcript'].str.replace('s p 500', 'S&P500', case = False)

""" 3 a) 2. Chunking so that sentence embedding works on full transcript """
# # Load english Spacy model
# nlp = spacy.load('en_core_web_sm')
#
# # Process the transcripts and track their corresponding 'id'
# transcripts = test_all_video_df['transcript_processed'].tolist()
# ids = test_all_video_df['id'].tolist()
#
# # Initialize lists to store the extracted sentences and their corresponding 'id'
# sentences = []
# sentence_ids = []
#
# # Iterate over each transcript and its corresponding 'id' and split into sentences
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
# new_df.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/sentences_ids_df.csv', index=False)
#
# transcript_chunks = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/sentences_ids_df.csv"
# transcript_chunks_df = pd.read_csv(transcript_chunks)
#
# # Combining the chunks as the spacy model splits them too much. Although better than any other model tried thus far
# # so I am using it and combining chunks to get chunks of roughly 200 words
# combined_chunks = []
# combined_ids = []
# current_chunk = ""
# current_id = transcript_chunks_df.iloc[0]['ID']
#
# for index, row in transcript_chunks_df.iterrows():
#     chunk = row['Sentences']
#     chunk_length = len(chunk.split())  # Assuming tokens are separated by whitespace
#
#     # If the current chunk length exceeds the desired limit, append the current chunk and ID to the combined lists
#     if len(current_chunk.split()) + chunk_length > 200:
#         combined_chunks.append(current_chunk)
#         combined_ids.append(current_id)
#         current_chunk = chunk
#         current_id = row['ID']
#     else:
#         # If the current chunk length is within the desired limit, append it to the current chunk
#         current_chunk += " " + chunk
#
#         # If it's the first chunk for a given ID, store the ID
#         if current_id == "":
#             current_id = row['ID']
#
#     # If the current chunk is the last one for a given ID, append it to the combined lists
#     if index == len(transcript_chunks_df) - 1 or row['ID'] != transcript_chunks_df.iloc[index + 1]['ID']:
#         combined_chunks.append(current_chunk)
#         combined_ids.append(current_id)
#         current_chunk = ""
#         current_id = ""
#
# # Create a new DataFrame with the combined chunks and their corresponding IDs
# transcript_chunks_combined_df = pd.DataFrame({'ID': combined_ids, 'combined_sentence': combined_chunks})
# transcript_chunks_combined_df.replace("", np.nan, inplace=True)
# transcript_chunks_combined_df.dropna(how='all', inplace=True)
# transcript_chunks_combined_df.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/transcript_chunks_combined_df.csv', index=False)
#
# chunk_counts = transcript_chunks_combined_df['ID'].value_counts()
# id_chunks_count_df = pd.DataFrame({'ID': chunk_counts.index, 'chunk_count': chunk_counts.values})
# id_chunks_count_df.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/id_chunks_count.csv', index=False)
#
# video_data_chunkcount = video_data.merge(id_chunks_count_df, left_on='id', right_on='ID', how='left')
# video_data_chunkcount.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/video_data_chunks_count.csv', index=False)


""" 3 b) Document embedding """
transcript_chunks_combined_df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/transcript_chunks_combined_df.csv')
transcripts = transcript_chunks_combined_df['combined_sentence'].tolist()

# start_time = time.time()
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings_1 = sentence_model.encode(transcripts)
# end_time = time.time()
# print("Embeddings 1 time:", end_time - start_time, " seconds")
# np.save('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/embeddings_1.npy', embeddings_1)
embeddings_1 = np.load('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/embeddings_1.npy')

# start_time = time.time()
# sentence_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
# embeddings_2 = sentence_model.encode(transcripts)
# end_time = time.time()
# print("Embeddings 2 time:", end_time - start_time, " seconds")
# np.save('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/embeddings_2.npy', embeddings_2)
embeddings_2 = np.load('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/embeddings_2.npy')

# start_time = time.time()
# sentence_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
# embeddings_3 = sentence_model.encode(transcripts)
# end_time = time.time()
# print("Embeddings 3 time:", end_time - start_time, " seconds")
# np.save('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/embeddings_3.npy', embeddings_3)
embeddings_3 = np.load('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/embeddings_3.npy')

""" 3 c) Creating the topic model """
# # Defining sub-models
#
# # Altering the stopwords to be omitted due to intricacies of language used in investing and YouTube videos generally
# vectorizer = CountVectorizer(stop_words="english")
# stopwords = vectorizer.get_stop_words()
# modified_stopwords = list(stopwords)
# words_to_remove = ["should", "must", "interest", "never"]
# words_to_add = ["im", "Im", "uh", "hi", "youre", "you're", 'um', 'okay', 'yes', 'yeah', 'just', 'really', 'theyre', 'thank', 'Music', 'Applause', 'foreign', 'music', 'hmm', 'ill']
# modified_stopwords.extend(words_to_add)
# for word in words_to_remove:
#     modified_stopwords.remove(word)
# vectorizer.set_params(stop_words=modified_stopwords)
#
# umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=100)
# hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=2, metric='euclidean', cluster_selection_method='eom')
# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
# # candidate_topics = ["crypto", "ETF", "advice", "strategy", "stock picks", "countries"]
# #representation_model_2 = ZeroShotClassification(candidate_topics, model="facebook/bart-large-mnli")
# representation_model_1 = MaximalMarginalRelevance(diversity=0.5)
#
# # Model building
# # Topic model 1
# topic_model_1 = BERTopic(
#     umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=100),
#     language="english",
#     hdbscan_model=hdbscan_model,
#     vectorizer_model=vectorizer,
#     ctfidf_model=ctfidf_model,
#     representation_model=representation_model_1,
#     verbose = True
# )
# start_time = time.time()
# topic_model_1 = topic_model_1.fit(transcripts, embeddings_1)
# end_time = time.time()
# print("Topic model 1 time:", end_time - start_time, " seconds")
#
# # Topic model 2
# topic_model_2 = BERTopic(
#     umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
#     language="english",
#     hdbscan_model=hdbscan_model,
#     vectorizer_model=vectorizer,
#     ctfidf_model=ctfidf_model,
#     representation_model=representation_model_1
# )
# start_time = time.time()
# topic_model_2 = topic_model_2.fit(transcripts, embeddings_2)
# end_time = time.time()
# print("Topic model 2 time:", end_time - start_time, " seconds")
#
# #Topic model 3
# topic_model_3 = BERTopic(
#     umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=14),
#     language="english",
#     hdbscan_model=hdbscan_model,
#     vectorizer_model=vectorizer,
#     ctfidf_model=ctfidf_model,
#     representation_model=representation_model_1
# )
# start_time = time.time()
# topic_model_3 = topic_model_3.fit(transcripts, embeddings_3)
# end_time = time.time()
# print("Topic model 3 time:", end_time - start_time, " seconds")

# # topic_model_1.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1")
# topic_model_1 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1")
# # topic_model_2.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_2")
# topic_model_2 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_2")
# # topic_model_3.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_3")
# topic_model_3 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_3")

# ## Visualisation
# # print("TM1")
# # print(topic_model_1.get_topic_info())
# # print("TM2")
# # print(topic_model_2.get_topic_info())

# vectorizer_model = CountVectorizer(stop_words=modified_stopwords, ngram_range=(1, 3), min_df=10)
#
# topic_model_1.update_topics(transcripts, vectorizer_model = vectorizer_model)
# topic_model_1.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1_countVec")
topic_model_1 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1_countVec")

# topic_model_2.update_topics(transcripts, vectorizer_model = vectorizer_model)
# topic_model_2.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_2_countVec")
topic_model_2 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_2_countVec")

# topic_model_3.update_topics(transcripts, vectorizer_model = vectorizer_model)
# topic_model_3.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_3_countVec")
topic_model_3 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_3_countVec")

# #topic_model_1_countVec.reduce_topics(transcripts, nr_topics=50)
# #topic_model_2_countVec.reduce_topics(transcripts, nr_topics=50)

## Visualisation
print("Updated with Count Vec TM1")
print(topic_model_1.get_topic_info().head(20))
# print("Updated with Count Vec TM2")
# print(topic_model_2.get_topic_info().head(12))
# print("Updated with Count Vec TM3")
# print(topic_model_3.get_topic_info().head(12))

# topic_model_1.get_topic_info().to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topics_list_TM1.csv")
#topic_model_2.get_topic_info().to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topics_list_TM2.csv")
#topic_model_3.get_topic_info().to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topics_list_TM3.csv")


""" 3 d) Visualising"""
# Barchart
# vis_barchart_1 = topic_model_1.visualize_barchart(top_n_topics = 12, n_words=5, width=500, height=500)
# pio.write_image(vis_barchart_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_barchart_1.png")
# vis_barchart_2 = topic_model_2.visualize_barchart(top_n_topics = 12, n_words=5, width=500, height=500)
# pio.write_image(vis_barchart_2, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM2/vis_barchart_2.png")
# vis_barchart_3 = topic_model_3.visualize_barchart(top_n_topics = 12, n_words=5, width=500, height=500)
# pio.write_image(vis_barchart_3, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM3/vis_barchart_3.png")

# term_rank = topic_model_1.visualize_term_rank()
# pio.write_image(term_rank, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Bertopic figures/term_rank.png")
#
# Visualise topics in 2D space
# vis_topics_1 = topic_model_1.visualize_topics(top_n_topics = 12)
# pio.write_image(vis_topics_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_topics_1.png")
# topic_model_1.visualize_topics().write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_topics_1_html.html")
# topic_model_1.visualize_topics(top_n_topics = 12).write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_topics_1_top12_html.html")
#
# vis_topics_2 = topic_model_2.visualize_topics(top_n_topics = 12)
# pio.write_image(vis_topics_2, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM2/vis_topics_2.png")
# topic_model_2.visualize_topics().write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM2/vis_topics_2_html.html")
# topic_model_2.visualize_topics(top_n_topics = 12).write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM2/vis_topics_2_top12_html.html")
#
# vis_topics_3 = topic_model_3.visualize_topics(top_n_topics = 12)
# pio.write_image(vis_topics_3, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM3/vis_topics_3.png")
# topic_model_3.visualize_topics().write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM3/vis_topics_3_html.html")
# topic_model_3.visualize_topics(top_n_topics = 12).write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM3/vis_topics_3_top12_html.html")
#
# # Topic hierarchy
# hierarchy_topics_1 = topic_model_1.visualize_hierarchy(top_n_topics=12)
# pio.write_image(hierarchy_topics_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/hierarchy_topics_1.png")
# hierarchy_topics_2 = topic_model_2.visualize_hierarchy(top_n_topics=12)
# pio.write_image(hierarchy_topics_2, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM2/hierarchy_topics_2.png")
# hierarchy_topics_3 = topic_model_3.visualize_hierarchy(top_n_topics=12)
# pio.write_image(hierarchy_topics_3, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM3/hierarchy_topics_3.png")

""" 3 e) Evaluation """


""" 4 Editing and visualising the chosen topic model """
# On inspection of the visualisations and the list of topics, topic model 1 is chosen to investigate further
""" Editing """
# Topic reduction
# topic_model_1.reduce_topics(transcripts, nr_topics=30)
# topics = topic_model_1.topics_
# Manually merging of topics



""" Visualising """
# Labels of topics
topic_labels_1 = topic_model_1.generate_topic_labels(nr_words=4,
                                                  topic_prefix=False,
                                                  word_length=15,
                                                  separator=", ")
topic_model_1.set_topic_labels(topic_labels_1)

# Defining the index of topics of particular interest
topics_of_interest_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                        14, 24, 26, 36, 38, 43, 57]

""" 4 a Visualise topics"""
# Barchart - top 12
vis_barchart_1 = topic_model_1.visualize_barchart(top_n_topics = 12, n_words=5, width=300, height=300)
pio.write_image(vis_barchart_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_barchart_1.png")
vis_barchart_1.write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_barchart_1.html")
topic_model_1.visualize_topics().write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_topics_1_html.html")
topic_model_1.visualize_topics(top_n_topics = 12).write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_topics_1_top12_html.html")

# Barchart - topics of interest
vis_barchart_1_toi = topic_model_1.visualize_barchart(topics=topics_of_interest_1, n_words=5, width=300, height=300)
pio.write_image(vis_barchart_1_toi, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_barchart_1_toi.png")
vis_barchart_1_toi.write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_barchart_1_toi.html")
topic_model_1.visualize_topics(topics=topics_of_interest_1).write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_topics_1__toi_html.html")


""" 4 b Visualise documents """
tm1_visualisation_toi = topic_model_1.visualize_documents(
    transcripts,
    embeddings=embeddings_1,
    hide_annotations=False,
    topics=topics_of_interest_1,
    custom_labels=True
).write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/tm1_visualisation_toi.html")
#pio.write_image(tm1_visualisation_toi, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/visualization.png")

""" 4 c Visualise topic hierarchy & topic tree"""
# Can ID which topics could be merged from these two figures
hierarchical_topics = topic_model_1.hierarchical_topics(transcripts)
hierarchy_topics_1 = topic_model_1.visualize_hierarchy()
hierarchy_topics_1.write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/hierarchy_topics_1.html")
pio.write_image(hierarchy_topics_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/hierarchy_topics_1_b.png")

topic_tree_1 = topic_model_1.get_topic_tree(hierarchical_topics)
# print(topic_tree_1)

# Visualise hierarchy of specific topics
hierarchy_topics_1 = topic_model_1.visualize_hierarchy(top_n_topics=12)
pio.write_image(hierarchy_topics_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/hierarchy_topics_1.png")

hierarchy_topics_1 = topic_model_1.visualize_hierarchy(topics=topics_of_interest_1)
pio.write_image(hierarchy_topics_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/hierarchy_topics_1_toi.png")

""" 4 d Visualise topic similarity """
topic_similarity_heatmap_1 = topic_model_1.visualize_heatmap()
pio.write_image(topic_similarity_heatmap_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/topic_similarity_heatmap_1.png")

""" 4 e Visualise hierarchical documents """
vis_hierarchical_docs = topic_model_1.visualize_hierarchical_documents(transcripts, hierarchical_topics, embeddings = embeddings_1)
vis_hierarchical_docs.write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_hierarchy_docs_1.html")

""" 5 Using the model """
