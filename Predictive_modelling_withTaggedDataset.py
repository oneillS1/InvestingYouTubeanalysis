
""" Predictive modelling:

Prediction tasks at 2 levels.
- Level 1: Transcript chunk: Inputs - transcript chunk embedding ; Predict - Advice flag
- Level 2: Video level: Inputs - metadata, transcript chunk embeddings as new variables; Predict - Advice flag
 """

""" Importing necessary packages """
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
import numpy as np
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import ast



""" Level 1 """
""" 1 a. Getting transcript chunks that are tagged """
# df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/dataset_for_tagging_tagged.csv')
# print(df[['Source', 'Advice']].value_counts())
# print(df.shape)
# # As I have only tagged a subset of the dataset originally planned for tagging, I have to keep the tagged for building the model
# df_tagged = df.dropna(subset=['Advice'], axis=0)
# print(df_tagged.shape)
# print(df_tagged[['Source', 'Advice']].value_counts())
# print(df_tagged.columns)
# df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm.csv', index=False)

""" 1 b. Adding the embeddings to use as inputs """
#df_tagged = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm.csv')
# transcripts_tagged_pm = df_tagged['combined_sentence'].tolist()
#
# start_time = time.time()
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings_1_tagged_pm = sentence_model.encode(transcripts_tagged_pm)
# end_time = time.time()
# print("Embeddings 1 time:", end_time - start_time, " seconds")
# np.save('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Embeddings/embeddings_1_tagged_pm.npy', embeddings_1_tagged_pm)
#
# df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv', index=False)

# print(df_tagged.head())
# print(df_tagged.shape)
# print(df_tagged.dtypes)
# print(df_tagged.columns)
# df_tagged['embeddings'] = df_tagged['combined_sentence'].apply(sentence_model.encode)
# print(df_tagged.head())
# print(df_tagged.shape)
# print(df_tagged.dtypes)
# print(df_tagged.columns)
# df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv', index=False)

""" 1 c. Splitting into train, validate, test datasets """
# df_tagged_pm1 = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv')
#
# df_tagged_pm1['embeddings'] = df_tagged_pm1['embeddings'].apply(lambda x: [float(val) for val in x[1:-1].split()])
#
# X = np.array(df_tagged_pm1['embeddings'].to_list())
# y = np.array(df_tagged_pm1['Advice'].to_list())
#
# # Split the data into training and testing sets (for k-fold cross-validation)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# """ 1 d. Train models on the training dataset """
# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models_summary, _ = clf.fit(X_train, X_test, y_train, y_test)
#
# print(models_summary)
# models_summary.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/LC_models_summary_pm_embedding_1.csv', index=False)
#


""" Level 2 """
df_tagged = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv')
df_tagged['embeddings2'] = df_tagged['embeddings'].apply(lambda x: [float(val) for val in x[1:-1].split()])
df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/checking_embedding.csv', index=False)

""" 2 a. Creating dataset with all information on each video (including embeddings of chunks) """
# # Step 1: Pivot the dataframe to convert 'embedding' and 'advice' into columns
# df_tagged_pivot = df_tagged.pivot_table(index='ID', columns=df_tagged.groupby('ID').cumcount(),
#                           values=['embeddings', 'Advice', 'combined_sentence'], aggfunc='first')
#
# df_tagged_pivot.columns = [f"{col}_{idx}" for col, idx in df_tagged_pivot.columns]
# df_tagged_pivot = df_tagged_pivot.reset_index()
# df_tagged_pivot.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/df_tagged_pivot.csv', index=False)
# print(df_tagged_pivot.head())
#
# video_data_chunks_count = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/video_data_chunks_count.csv')
# metadata_embeddings_tagged_df_pm = pd.merge(video_data_chunks_count, df_tagged_pivot, on='ID', how='inner')
# metadata_embeddings_tagged_df_pm.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/metadata_embeddings_tagged_df_pm.csv', index=False)
#
# """ 2 b. Embed the title and description to use in a predictive model """
# metadata_embeddings_tagged_df_pm['title_embedding'] = metadata_embeddings_tagged_df_pm['title'].apply(sentence_model.encode)
# #metadata_embeddings_tagged_df_pm['descr_embedding'] = metadata_embeddings_tagged_df_pm['description'].apply(sentence_model.encode)
#
# #metadata_embeddings_tagged_df_pm['title_embedding'] = metadata_embeddings_tagged_df_pm['title_embedding'].apply(lambda x: [float(val) for val in x[1:-1].split()])
# #metadata_embeddings_tagged_df_pm['descr_embedding'] = metadata_embeddings_tagged_df_pm['descr_embedding'].apply(lambda x: [float(val) for val in x[1:-1].split()])
#
# metadata_embeddings_tagged_df_pm['sum_advice'] = metadata_embeddings_tagged_df_pm.apply(lambda row: row.filter(like='Advice_').fillna(0).sum(), axis=1)
# metadata_embeddings_tagged_df_pm['advice_binary'] = metadata_embeddings_tagged_df_pm['sum_advice'].apply(lambda x: 1 if x > 0 else 0)
#
# metadata_embeddings_tagged_df_pm.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/metadata_embeddings_tagged_df_pm.csv', index=False)
# print(metadata_embeddings_tagged_df_pm['advice_binary'].value_counts())
# print(metadata_embeddings_tagged_df_pm['sum_advice'].value_counts())

""" 2 c. Splitting into train, validate, test datasets """


""" 2 d. Train models on the training dataset """

