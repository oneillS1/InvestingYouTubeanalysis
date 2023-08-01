
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
df_tagged = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm.csv')
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

print(df_tagged.head())
print(df_tagged.shape)
print(df_tagged.dtype)
print(df_tagged.colnames)
df_tagged['embeddings'] = df_tagged['combined_sentence'].apply(sentence_model.encode)
print(df_tagged.head())
print(df_tagged.shape)
print(df_tagged.dtype)
print(df_tagged.colnames)
df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv', index=False)

""" 1 c. Splitting into train, validate, test datasets """
df_tagged_pm1 = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv')


# offset = int(df_tagged_pm1.shape[0] * 0.9)
# print(offset)

X = df_tagged_pm1['embeddings'].to_list()
y = df_tagged_pm1['Advice'].to_list()

# Split the data into training and testing sets (for k-fold cross-validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

""" 1 d. Train models on the training dataset """
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)