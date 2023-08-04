
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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import ast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE



""" Level 1 """
""" 1 a. Getting transcript chunks that are tagged """
# df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/dataset_for_tagging_tagged.csv')
# print(df['Source'].value_counts())
# # print(df[['Source', 'Advice']].value_counts())
#
# # # As I have only tagged a subset of the dataset originally planned for tagging, I have to keep the tagged for building the model
# df_tagged = df.dropna(subset=['Advice'], axis=0)
# print(df_tagged[['Source', 'Advice']].value_counts())
# print(df_tagged['Source'].value_counts())
# # print(df_tagged.columns)
# df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm.csv', index=False)

""" 1 b. Adding the embeddings to use as inputs """
# df_tagged = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm.csv')
# transcripts_tagged_pm = df_tagged['combined_sentence'].tolist()
#
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

## Code below only needed if you want to save the embeddings
# start_time = time.time()
# embeddings_1_tagged_pm = sentence_model.encode(transcripts_tagged_pm)
# end_time = time.time()
# print("Embeddings 1 time:", end_time - start_time, " seconds")
# np.save('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Embeddings/embeddings_1_tagged_pm.npy', embeddings_1_tagged_pm)
# df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv', index=False)

# df_tagged['embeddings'] = df_tagged['combined_sentence'].apply(sentence_model.encode)
# df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv', index=False)

""" 1 c. Splitting into train, validate, test datasets """
df_tagged_pm1 = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv')

df_tagged_pm1['embeddings'] = df_tagged_pm1['embeddings'].apply(lambda x: [float(val) for val in x[1:-1].split()])

X = np.array(df_tagged_pm1['embeddings'].to_list())
y = np.array(df_tagged_pm1['Advice'].to_list())

# Split the data into training and testing sets (for k-fold cross-validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # """ 1 d. Train models on the training dataset """
# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models_summary, _ = clf.fit(X_train, X_test, y_train, y_test)
#
# print(models_summary)
# models_summary.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/LC_models_summary_pm_embedding_1.csv', index=False)
#
# # Create a text file to save the classification reports
# with open('classification_reports.txt', 'w') as f:
#     # Loop through models and get classification reports
#     for model_name in models_summary.index:
#         model = clf.models[model_name]
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#
#         classification_rep = classification_report(y_test, y_pred)
#
#         # Write the classification report to the text file
#         f.write(f"Classification Report for {model_name}:\n")
#         f.write(classification_rep)
#         f.write('\n\n')
#
# print("Classification reports saved to classification_reports.txt")

""" 1 d Training models that look promising based on 1 c and other models """
# Define cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the list of models
models = [
    ('BernoulliNB', BernoulliNB()),
    ('NearestCentroid', NearestCentroid()),
    ('GaussianNB', GaussianNB()),
    ('LogisticRegression', LogisticRegression()),
    ('KNeighborsClassifier', KNeighborsClassifier()),
    ('LinearSVC', LinearSVC()),
    ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()),
    ('SGDClassifier', SGDClassifier())
]

# Lists to store metric scores for each model
avg_metric_scores = []

# Open a text file to write the output
with open('model_evaluation.txt', 'w') as f:
    # Loop over models
    for model_name, model in models:
        metric_scores = []

        # Loop over cross-validation folds
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Apply SMOTE to the training set
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test)

            accuracy = round(accuracy_score(y_test, y_pred), 2)
            precision = round(precision_score(y_test, y_pred), 2)
            recall = round(recall_score(y_test, y_pred), 2)
            f1 = round(f1_score(y_test, y_pred), 2)
            class_report = classification_report(y_test, y_pred)

            metric_scores.append((accuracy, precision, recall, f1, class_report))

        # Calculate average metrics for the model
        avg_accuracy = round(np.mean([score[0] for score in metric_scores]), 2)
        avg_precision = round(np.mean([score[1] for score in metric_scores]), 2)
        avg_recall = round(np.mean([score[2] for score in metric_scores]), 2)
        avg_f1 = round(np.mean([score[3] for score in metric_scores]), 2)
        avg_class_report = '\n\n'.join([score[4] for score in metric_scores])

        avg_metric_scores.append((model_name, avg_accuracy, avg_precision, avg_recall, avg_f1, avg_class_report))

        # Write model results to the file
        f.write(f"Model: {model_name}\n")
        f.write(f"Final Average Accuracy: {avg_accuracy}\n")
        f.write(f"Final Average Precision: {avg_precision}\n")
        f.write(f"Final Average Recall: {avg_recall}\n")
        f.write(f"Final Average F1-score: {avg_f1}\n")
        f.write(f"Final Classification Report:\n{avg_class_report}\n")
        f.write("----------------------\n")

# Print the summary of results
for model_name, avg_accuracy, avg_precision, avg_recall, avg_f1, avg_class_report in avg_metric_scores:
    print(f"Model: {model_name}")
    print("Final Average Accuracy:", avg_accuracy)
    print("Final Average Precision:", avg_precision)
    print("Final Average Recall:", avg_recall)
    print("Final Average F1-score:", avg_f1)
    print("Final Classification Report:\n", avg_class_report)
    print("----------------------")



""" Level 2 """
# df_tagged = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv')
# df_tagged['embeddings2'] = df_tagged['embeddings'].apply(lambda x: [float(val) for val in x[1:-1].split()])
# df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/checking_embedding.csv', index=False)

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

