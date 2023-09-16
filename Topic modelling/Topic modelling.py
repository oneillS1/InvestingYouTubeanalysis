
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
video_data_path = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Final dataset(s) for analysis/cleaned data.csv"
video_data = pd.read_csv(video_data_path)

test_all_video_df = video_data[video_data['Transcript'] != ''][['Transcript', 'id']] # just need id and transcript for topic modelling

""" 3. Topic modelling """

""" 3 a) Data preprocessing """

""" 3 a) 1. Removing / Altering poor auto transcribed videos """
# s p 500 = s&p500
test_all_video_df['transcript_processed'] = test_all_video_df['Transcript'].str.replace('s p 500', 'S&P500', case = False)

""" 3 a) 2. Chunking so that sentence embedding works on full transcript """
# Load english version of the spacy model which is used to split transcripts into appropriate length chunks for embedding
nlp = spacy.load('en_core_web_sm')

# Create the list of transcripts to be embedded and also track their corresponding 'id'
transcripts = test_all_video_df['transcript_processed'].tolist()
ids = test_all_video_df['id'].tolist()

# Initialize lists for the chunks and their corresponding 'id'
sentences = []
sentence_ids = []

# Iterate over each transcript and its corresponding 'id' and split them into chunks
for transcript, id in zip(transcripts, ids):
    # Process the transcript
    doc = nlp(transcript) # applying the spacy model

    # Extract sentences from the chunks created from the spacy model
    sentences.extend([sent.text for sent in doc.sents])
    sentence_ids.extend([id] * len(list(doc.sents)))

# Create a new dataframe with the extracted sentences and their corresponding 'id'
chunks_df = pd.DataFrame({'Sentences': sentences, 'ID': sentence_ids})
chunks_df.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/sentences_ids_df.csv', index=False)

# Loading this dataset in (can start from here on re-running)
transcript_chunks = "C:/Users/Steve.HAHAHA/Desktop/Dissertation/sentences_ids_df.csv"
transcript_chunks_df = pd.read_csv(transcript_chunks)

# On inspection the chunks are very short i.e., some are only a few words. This loses too much context.
# As a result I will combine the chunks as the spacy model splits them too much. Although Spacy model is better than any other model tried thus far
# so I am using it and combining chunks to get chunks of roughly 200 words
combined_chunks = []
combined_ids = []
current_chunk = ""
current_id = transcript_chunks_df.iloc[0]['ID']

for index, row in transcript_chunks_df.iterrows():
    chunk = row['Sentences']
    chunk_length = len(chunk.split())  # This assumes tokens are separated by whitespace which is usually true

    # If the current chunk length exceeds the desired limit I set, then I append the current chunk and ID to the combined lists
    if len(current_chunk.split()) + chunk_length > 200: # setting 200 as the upper limit of the number of words
        combined_chunks.append(current_chunk)
        combined_ids.append(current_id)
        current_chunk = chunk
        current_id = row['ID']
    else:
        # If the current chunk length is within the desired limit, then combine it with the next chunk
        current_chunk += " " + chunk

        #  store the ID if its the first time it appears
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

transcript_chunks_combined_df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/transcript_chunks_combined_df.csv')
#
# chunk_counts = transcript_chunks_combined_df['ID'].value_counts()
# id_chunks_count_df = pd.DataFrame({'ID': chunk_counts.index, 'chunk_count': chunk_counts.values})
# id_chunks_count_df.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/id_chunks_count2.csv', index=False)
#
# video_data_chunkcount = video_data.merge(id_chunks_count_df, left_on='id', right_on='ID', how='left')
# video_data_chunkcount.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/video_data_chunks_count.csv', index=False)

""" 3 b) Document embedding """
transcript_chunks_combined_df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/transcript_chunks_combined_df.csv')
transcripts = transcript_chunks_combined_df['combined_sentence'].tolist() # needed for the embedding models, error otherwise

# Embed with models 1, 2, 3 and save them. Timed to see which is quickest. Commented out for now as they take a while to run
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
# Defining sub-models

# Altering the stopwords to be omitted due to intricacies of language used in investing and YouTube videos generally, in particular speech
# Uses countvectorizer as base list of stop words which I then edit with adding and removing words
vectorizer = CountVectorizer(stop_words="english")
stopwords = vectorizer.get_stop_words()
modified_stopwords = list(stopwords)
words_to_remove = ["should", "must", "interest", "never"]
words_to_add = ["im", "Im", "uh", "hi", "youre", "you're", 'um', 'okay', 'yes', 'yeah', 'just', 'really', 'theyre', 'thank', 'Music', 'Applause', 'foreign', 'music', 'hmm', 'ill']
modified_stopwords.extend(words_to_add)
for word in words_to_remove:
    modified_stopwords.remove(word)
vectorizer.set_params(stop_words=modified_stopwords)

# BERTopic consists of submodels which are defined here (justifications for choices in report)
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=100)
hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=2, metric='euclidean', cluster_selection_method='eom')
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
representation_model_1 = MaximalMarginalRelevance(diversity=0.5)

# Model building - for models 1,2,3 using each embeddings created
# Topic model 1
topic_model_1 = BERTopic(
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=100),
    language="english",
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model_1,
    verbose = True
)
start_time = time.time()
topic_model_1 = topic_model_1.fit(transcripts, embeddings_1)
end_time = time.time()
print("Topic model 1 time:", end_time - start_time, " seconds")

# Topic model 2
topic_model_2 = BERTopic(
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
    language="english",
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model_1
)
start_time = time.time()
topic_model_2 = topic_model_2.fit(transcripts, embeddings_2)
end_time = time.time()
print("Topic model 2 time:", end_time - start_time, " seconds")

# Topic model 3
topic_model_3 = BERTopic(
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=14),
    language="english",
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model_1
)
start_time = time.time()
topic_model_3 = topic_model_3.fit(transcripts, embeddings_3)
end_time = time.time()
print("Topic model 3 time:", end_time - start_time, " seconds")

# SAVING AND LOADING AGAIN - USE THE LOADING ON RE-RUNNING TO SAVE TIME
topic_model_1.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1")
topic_model_1 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1")
topic_model_2.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_2")
topic_model_2 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_2")
topic_model_3.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_3")
topic_model_3 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_3")

## Visualisation
# print("TM1")
# print(topic_model_1.get_topic_info())
# print("TM2")
# print(topic_model_2.get_topic_info())

# Updating the model by removing the stopwords and allowing for 3 word phrases
vectorizer_model = CountVectorizer(stop_words=modified_stopwords, ngram_range=(1, 3), min_df=10)

topic_model_1.update_topics(transcripts, vectorizer_model = vectorizer_model)
topic_model_1.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1_countVec")
topic_model_1 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1_countVec")

topic_model_2.update_topics(transcripts, vectorizer_model = vectorizer_model)
topic_model_2.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_2_countVec")
topic_model_2 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_2_countVec")

topic_model_3.update_topics(transcripts, vectorizer_model = vectorizer_model)
topic_model_3.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_3_countVec")
topic_model_3 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_3_countVec")

# BELOW CODE NOT USED BUT RAN WHILE INSPECTING THE MODEL
# # #topic_model_1_countVec.reduce_topics(transcripts, nr_topics=50)
# # #topic_model_2_countVec.reduce_topics(transcripts, nr_topics=50)
#
# ## Visualisation of the models (checking primarily)
# print("Updated with Count Vec TM1")
# print(topic_model_1.get_topic_info().head(20))
# # print("Updated with Count Vec TM2")
# # print(topic_model_2.get_topic_info().head(12))
# # print("Updated with Count Vec TM3")
# # print(topic_model_3.get_topic_info().head(12))

topic_model_1.get_topic_info().to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topics_list_TM1.csv")
topic_model_2.get_topic_info().to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topics_list_TM2.csv")
topic_model_3.get_topic_info().to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topics_list_TM3.csv")


""" 3 d) Visualising the 3 models and saving the figures (all now on github too) """
# Barchart
vis_barchart_1 = topic_model_1.visualize_barchart(top_n_topics = 12, n_words=5, width=500, height=500)
pio.write_image(vis_barchart_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_barchart_1.png")
vis_barchart_2 = topic_model_2.visualize_barchart(top_n_topics = 12, n_words=5, width=500, height=500)
pio.write_image(vis_barchart_2, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM2/vis_barchart_2.png")
vis_barchart_3 = topic_model_3.visualize_barchart(top_n_topics = 12, n_words=5, width=500, height=500)
pio.write_image(vis_barchart_3, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM3/vis_barchart_3.png")

term_rank = topic_model_1.visualize_term_rank()
pio.write_image(term_rank, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/Bertopic figures/term_rank.png")

# Visualise topics in 2D space
vis_topics_1 = topic_model_1.visualize_topics(top_n_topics = 12)
pio.write_image(vis_topics_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_topics_1.png")
topic_model_1.visualize_topics().write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_topics_1_html.html")
topic_model_1.visualize_topics(top_n_topics = 12).write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_topics_1_top12_html.html")

vis_topics_2 = topic_model_2.visualize_topics(top_n_topics = 12)
pio.write_image(vis_topics_2, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM2/vis_topics_2.png")
topic_model_2.visualize_topics().write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM2/vis_topics_2_html.html")
topic_model_2.visualize_topics(top_n_topics = 12).write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM2/vis_topics_2_top12_html.html")

vis_topics_3 = topic_model_3.visualize_topics(top_n_topics = 12)
pio.write_image(vis_topics_3, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM3/vis_topics_3.png")
topic_model_3.visualize_topics().write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM3/vis_topics_3_html.html")
topic_model_3.visualize_topics(top_n_topics = 12).write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM3/vis_topics_3_top12_html.html")

# Topic hierarchy
hierarchy_topics_1 = topic_model_1.visualize_hierarchy(top_n_topics=12)
pio.write_image(hierarchy_topics_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/hierarchy_topics_1.png")
hierarchy_topics_2 = topic_model_2.visualize_hierarchy(top_n_topics=12)
pio.write_image(hierarchy_topics_2, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM2/hierarchy_topics_2.png")
hierarchy_topics_3 = topic_model_3.visualize_hierarchy(top_n_topics=12)
pio.write_image(hierarchy_topics_3, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM3/hierarchy_topics_3.png")


""" 4 Editing and visualising the chosen topic model """
# On inspection of the visualisations and the list of topics, topic model 1 is chosen to investigate further
topic_model_1 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1_countVec")

# Commented out code for reducing topic number to 20 and 50 (as written about in report). Code below was altered to run and save the same visualisations for
# these 2 as well. All outputs available on Github
# topic_model_1 = topic_model_1.reduce_topics(transcripts, nr_topics=50)
# topic_model_1.get_topic_info().to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topics_list_TM1_50.csv")
# topic_model_1 = topic_model_1.reduce_topics(transcripts, nr_topics=20)
#
## Defining the index of topics of particular interest
topics_of_interest_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        14, 24, 26, 36, 43, 54, 59, 60]

# """ 4 b: Visualise topics"""
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
topic_model_1.visualize_topics(topics=topics_of_interest_1).write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_topics_1_toi_html.html")

""" 4 b) Visualise documents """
tm1_visualisation_toi = topic_model_1.visualize_documents(
    transcripts,
    embeddings=embeddings_1,
    hide_annotations=False,
    # topics=topics_of_interest_1,
    custom_labels=True
).write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/tm1_visualisation_toi.html")
#pio.write_image(tm1_visualisation_toi, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/visualization.png")

""" 4 c) Visualise topic hierarchy & topic tree"""
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

# hierarchy_topics_1 = topic_model_1.visualize_hierarchy(topics=topics_of_interest_1)
# pio.write_image(hierarchy_topics_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/hierarchy_topics_1_toi.png")

""" 4 d) Visualise topic similarity """
topic_similarity_heatmap_1 = topic_model_1.visualize_heatmap()
pio.write_image(topic_similarity_heatmap_1, "C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/topic_similarity_heatmap_1.png")

""" 4 e) Visualise hierarchical documents """
vis_hierarchical_docs = topic_model_1.visualize_hierarchical_documents(transcripts, hierarchical_topics, embeddings = embeddings_1)
vis_hierarchical_docs.write_html("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/Model figures/TM1/vis_hierarchy_docs_1.html")

""" 5 Evaluating the topic model using silhouette and UMass """
from sklearn.metrics import silhouette_score
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import joblib

topic_model = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1_countVec")
transcripts = transcript_chunks_combined_df['combined_sentence'].to_list()
#
# I do the metrics for the full topic model, and then the one with 50 topics and 20 topics, by altering topic_model in below code to topic_model_smaller
# Rerun the code below then with topic_model_smaller rather than topic model.
# topic_model_smaller = topic_model.reduce_topics(transcripts, nr_topics=50)
# topic_model_smaller = topic_model.reduce_topics(transcripts, nr_topics=20)

topics, _ = topic_model.fit_transform(transcripts)
joblib.dump(topics, 'C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topics_1.pkl')
joblib.dump(_, 'C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/transformed_data_1.pkl')

# topics_smaller, _smaller = topic_model_smaller.fit_transform(transcripts)
# joblib.dump(topics_smaller, 'C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topics_smaller.pkl')
# joblib.dump(_smaller, 'C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/transformed_data_smaller.pkl')

topics = joblib.load('C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topics_1.pkl')
_ = joblib.load('C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/transformed_data_1.pkl')

# Create a dataframe ot transcript chunks, an ID and the topics from model 1
chunks = pd.DataFrame({"Chunk": transcripts,
                          "ID": range(len(transcripts)), # ID from 0 onwards for each chunks
                          "Topic": topics})
# ALtering the df to show the chunks assigned to each topic in one list
chunks_per_topic = chunks.groupby(['Topic'], as_index=False)
chunks_per_topic = chunks_per_topic.agg({'Chunk': ' '.join}) # grouping the documents in the same topic
cleaned_chunks = topic_model._preprocess_text(chunks_per_topic.Chunk.values) # preprocessing the trasncripts (needed to assess topic coherence)

# Extract vectorizer and analyzer from BERTopic
vectorizer = topic_model.vectorizer_model
analyzer = vectorizer.build_analyzer() # analyser needed to tokenise the grouped transcripts must be the same as in the topic model hence these two lines

# Extract the 4 input features for the topic coherence model (tokenised trasncripts, dicctionary, bag of words corupus and words associated with each topic)
# (from Grootendorst on GitHub - https://github.com/MaartenGr/BERTopic/issues/90)
tokenised_transcripts = [analyzer(doc) for doc in cleaned_chunks]
dictionary_idFreq = corpora.Dictionary(tokenised_transcripts) # create a gensim dictionary from the tokenised transcripts - each transcript chunk rep as word id word frequency pairs
corpus = [dictionary_idFreq.doc2bow(token) for token in tokenised_transcripts] # neeeded to create a bag of words representation of the full corpus
# topic_words = [[words for words, _ in topic_model.get_topic(topic)]
#                for topic in range(len(set(topics))-1)]
# Getting the words associated with each topic
topic_words = []
for topic in range(len(set(topics)) - 1):
    words_for_current_topic = [words for words, _ in topic_model.get_topic(topic)]
    topic_words.append(words_for_current_topic)

# Instantiating a CoherenceModel from gensim which allows for finding umass score
if __name__ == '__main__':
    coherence_model_umass = CoherenceModel(topics=topic_words,
                           texts=tokenised_transcripts,
                           corpus=corpus,
                           dictionary=dictionary_idFreq,
                           coherence='u_mass')
    coherence_umass = coherence_model_umass.get_coherence()
    print(coherence_umass)

# Calculating silhuuette Score from the labels used in hbdscan
labels = topic_model.hdbscan_model.labels_
silhouette_avg = silhouette_score(embeddings_1, labels)

# Write up metrics (add suffix to title for 20 and 50 topic version of the .txt files - after changing the topic model)
output_file = "topic_modelling_metrics_model_1.txt"
with open(output_file, "w") as txtfile:
    txtfile.write("Topic Modelling Metrics - Model 1\n")
    txtfile.write("-------------------------------\n")
    txtfile.write(f"Coherence (UMass): {coherence_umass:.4f}\n")
    txtfile.write(f"Silhouette Score: {silhouette_avg:.4f}\n")

# """ 6. Applied context: Example of search topics """
## To use find_topics(), which finds the topics most similar to a search term, the embeddings have to be passed directly to the model
## The below recreates the topic model above just with the embeddings directly in it.
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
topic_model_1 = BERTopic(
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=100),
    language="english",
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model_1,
    verbose = True,
    embedding_model = sentence_model
)

start_time = time.time()
topics_1_test, probs_1_test = topic_model_1.fit_transform(transcripts, embeddings_1)
end_time = time.time()
print("Topic model 1 transform time:", end_time - start_time, " seconds")
vectorizer_model = CountVectorizer(stop_words=modified_stopwords, ngram_range=(1, 3), min_df=10)

topic_model_1.update_topics(transcripts, vectorizer_model = vectorizer_model)
topic_model_1.save("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1_test")


topic_model_1 = BERTopic.load("C:/Users/Steve.HAHAHA/Desktop/Dissertation/BERTopic models/topic_model_1_test")
## Find the topics most similar to the terms 'crypto advice', 'stock tips' and print the 1st one's most common words
similar_topics, similarity = topic_model_1.find_topics("Electric vehicles", top_n=5)
print(topic_model_1.get_topic(similar_topics[0]))
