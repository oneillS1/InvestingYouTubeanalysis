
""" Predictive modelling:

This script builds the predictive model reported

- Level 1: Transcript chunk: Inputs - transcript chunk embedding ; Predict - Advice flag

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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
import xgboost as xgb  # Import XGBoost
from tensorflow import keras
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
import joblib
from sklearn.model_selection import GridSearchCV
from keras.models import load_model


""" Level 1 - Predicting presence of financial advice from the transcript chunks """
""" 1 a. Getting transcript chunks that are tagged """
df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/dataset_for_tagging_tagged.csv') # dataset that has been manually tagged as having financial advice or guidance
# print(df['Source'].value_counts())
# print(df[['Source', 'Advice']].value_counts())

# # As I have only tagged a subset of the dataset originally planned for tagging, I have to keep the tagged subset for building the model
df_tagged = df.dropna(subset=['Advice'], axis=0)
print(df_tagged[['Source', 'Advice']].value_counts())
print(df_tagged['Source'].value_counts())
df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm.csv', index=False) # saving for subsequent use

""" 1 b. Adding the embeddings to use as inputs """
# 1 b i)
df_tagged = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm.csv')
transcripts_tagged_pm = df_tagged['combined_sentence'].tolist()

# Define model used for the embeddings (chosen as it performed best on topic modelling on similar dataset)
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

## 1 b ii) Code below embedding the transcripts - (only run if you need to save the embeddings - #'d out as I run 1 b iii instead)
# start_time = time.time()
# embeddings_1_tagged_pm = sentence_model.encode(transcripts_tagged_pm)
# end_time = time.time()
# print("Embeddings 1 time:", end_time - start_time, " seconds")
# np.save('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Embeddings/embeddings_1_tagged_pm.npy', embeddings_1_tagged_pm)
# df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv', index=False)

# 1 b iii) Code below to be run instead of 1 b ii if saving the embeddings separately not necessary
df_tagged['embeddings'] = df_tagged['combined_sentence'].apply(sentence_model.encode) # apply the MiniLM model for embedding and add as new variable
df_tagged.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv', index=False)

""" 1 c. Splitting into train, validate, test datasets """
df_tagged_pm1 = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm_embedding_1.csv')
print(df_tagged_pm1[['Source', 'Advice']].value_counts())
print(df_tagged_pm1['Source'].value_counts())

df_tagged_pm1['embeddings'] = df_tagged_pm1['embeddings'].apply(lambda x: [float(val) for val in x[1:-1].split()])

# NP arrays rather than lists needed for the Lazy Classifier subsequently
X = np.array(df_tagged_pm1['embeddings'].to_list())
y = np.array(df_tagged_pm1['Advice'].to_list())

# Check shape of embedding array (needed as an input for neural network later on - all must be the same too) # shape = 384
first_shape = len(X[0])
# Check if all embeddings have the same shape
all_same_shape = all(len(arr) == first_shape for arr in X)
if all_same_shape:
    print("All embeddings have the same shape:", first_shape)
else:
    print("Embeddings have different shapes")

# Split the data into training and testing sets. Stratify = y ensures same ratio of positive cases in the test dataset as in training dataset. 20% to test dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

""" 1 d. Train models on the training dataset """
# Trying LazyClassifer first to get an idea of which types of models appear to do the prediction task well
## THIS CODE IS COMMENTED OUT AS IT IS SUBSEQUENTLY NOT USED AND IS SUPERSEDED BY THE MODELS IN 1 E. LEFT IN FOR COMPLETENESS
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

""" 1 d Training models from 1 c using cross validation and SMOTE """
# Training a long list of models, some from LazyClassifier using k-fold cross validation and also with the inclusion of SMOTE.
# This is to ascertain which type of model would be best for fine tuning (as explained in the report)
# SMOTE is a technique for oversampling of positive cases in an unbalanced dataset

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # defining the cross validaiton

# # Define the list of models I will train (2 not #'d out are the two best models, to see the rest remove the #)
models = [
    ('BernoulliNB', BernoulliNB()),
    ('NearestCentroid', NearestCentroid()),
    ('GaussianNB', GaussianNB()),
    ('LogisticRegression', LogisticRegression()),
    ('KNeighborsClassifier', KNeighborsClassifier()),
    ('LinearSVC', LinearSVC()),
    ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()),
    ('SGDClassifier', SGDClassifier()),
    ('Perceptron', Perceptron()),
    ('PassiveAggressiveClassifier', PassiveAggressiveClassifier()),
    ('RidgeClassifier', RidgeClassifier()),
    ('DecisionTreeClassifier', DecisionTreeClassifier()),
    ('AdaBoostClassifier', AdaBoostClassifier()),
    ('RandomForestClassifier', RandomForestClassifier()),
    ('DummyClassifier', DummyClassifier()),
    ('XGBClassifier', xgb.XGBClassifier()),
    ('SVC', SVC()),
    ('FeedforwardNN', Sequential([
        Dense(64, activation='relu', input_shape=(384,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ]))
]

# Compile neural network model before training
for model_name, model in models:
    if isinstance(model, Sequential):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Recall']) # using recall as it is most relevant metric for application

# I train each model and save the metrics and write to a txt file so I can easily view and compare
# Lists to store metric scores for each model
avg_metric_scores = []#

# #  I RERUN THIS CODE MULTIPLE TIMES WITH VARIOUS PARTS COMMENTED OUT IN ORDER TO GET THE DIFFERENCE .TXT FILES
# # .txt files of model_evaluation.txt and model_evaluation_no_smote.txt run all the models (with and without SMOTE used respectively).
# # Model_evaluation_main2.txt just shows the LR and NN models (as these were the best two performers).
# # To re-run simply (un)comment out the relevant parts
#
# Creating 1 text file - the name is altered for each of the files
with open('model_evaluation_main.txt', 'w') as f:
    # Loop over all the modls listed
    for model_name, model in models:
        metric_scores = []

        # For each cross-validation fold
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Apply SMOTE to the training set (remove and include line #'d in if/else statement below to use no smote)
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train) # resampling using SMOTE

            if isinstance(model, Sequential):
                model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=32, verbose=0)
                y_pred = (model.predict(X_test) > 0.5).astype(int)
            else:
                model.fit(X_train_resampled, y_train_resampled) # smote used
                # model.fit(X_train, y_train) # if no smote to be used
                y_pred = model.predict(X_test)

            # Calculating all metics mentioned , not just recall, to compare on
            accuracy = round(accuracy_score(y_test, y_pred), 2)
            precision = round(precision_score(y_test, y_pred), 2)
            recall = round(recall_score(y_test, y_pred), 2)
            f1 = round(f1_score(y_test, y_pred), 2)
            class_report = classification_report(y_test, y_pred)

            metric_scores.append((accuracy, precision, recall, f1, class_report))

        # Calculate average metrics for each model
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

        # After running the above the best 2 models are chosen and saved (this code was added after first running)
        if model_name == 'FeedforwardNN':
            model.save('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/feedforward_nn_model_recall.h5')
        elif model_name == 'LogisticRegression':
            joblib.dump(model, 'C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/logistic_regression_model.pkl')

""" 1 e Testing the best model on test dataset & parameter tuning the best model """
## recreating nn models which multiple different parameters
from itertools import product

# Setting the parameters for the grid search for the best parameters for the neural network (originally tried to test more parameters but reduced as computing power did not allow)
hidden_layers = [1, 2]
neurons_layer1 = [32, 64]
neurons_layer2 = [16, 32]
activation_functions = ['relu']
optimisers = ['adam']
batch_sizes = [16, 32]
epochs_values = [5, 10]

# Create a list to store results
results = []
model_counter = 1 # initiated so as to allow iteration thru the combination of paraameters

# Loop over hyperparameters (product() creates a combination of the parameters to loop over)
for hl, n1, n2, activation, optimizer, batch_size, epochs in product(hidden_layers, neurons_layer1, neurons_layer2, activation_functions, optimisers, batch_sizes, epochs_values):
    model_name = f'Model-{model_counter}' # to track which model and make subsequent reference easier
    hl_value = hl
    neurons1_value = n1
    neurons2_value = n2
    activation_value = activation
    optimiser_value = optimizer
    batch_size_value = batch_size
    epochs_value = epochs

    model = Sequential() # base nn model

    # Adding a hidden layer (at least 1 for all models)
    model.add(Dense(n1, activation=activation, input_shape=(384,)))

    if hl == 2:
        # Adding second layer for some models (50% of them) - testing to see if improvemnet and therefore if it is necessary
        model.add(Dense(n2, activation=activation))
        neurons2_value = n2
    else:
        neurons2_value = 0 # added as otherwise the output says nn with 1 layer have non negative neurons in 2nd layer

    # Final output layer should always have 1 neuron and as a ersult sigmoid function to output 0-1
    model.add(Dense(1, activation='sigmoid'))

    # Compile the models and optimise them for recall
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['Recall'])

    metric_scores = []

    # Cross validation and training - same as previous traiining code above
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train) # use SMOTE for all trainings this time

        start_time = time.time()
        model.fit(X_train_resampled, y_train_resampled, epochs=epochs, batch_size=batch_size, verbose=0)
        end_time = time.time()
        training_time = round(end_time - start_time, 2) # add in another metric of time taken to train

        y_pred = (model.predict(X_test) > 0.5).astype(int) # threshold of .5 chosen

        accuracy = round(accuracy_score(y_test, y_pred), 2)
        precision = round(precision_score(y_test, y_pred), 2)
        recall = round(recall_score(y_test, y_pred), 2)
        f1 = round(f1_score(y_test, y_pred), 2)

        metric_scores.append((accuracy, precision, recall, f1, training_time)) # append to the list for use in df later

    avg_accuracy = round(np.mean([score[0] for score in metric_scores]), 2)
    avg_precision = round(np.mean([score[1] for score in metric_scores]), 2)
    avg_recall = round(np.mean([score[2] for score in metric_scores]), 2)
    avg_f1 = round(np.mean([score[3] for score in metric_scores]), 2)
    avg_training_time = round(np.mean([score[4] for score in metric_scores]), 2)

    results.append((model_name, hl_value, neurons1_value, neurons2_value, activation_value, optimiser_value, batch_size_value,
                    epochs_value, avg_accuracy, avg_precision, avg_recall, avg_f1, avg_training_time))

    model_counter += 1

    # Saving the best one, chosen by inspection of output after first run (saved to use later without retraiining all the models)
    if hl == 2 and n1 ==64  and n2 ==32  and activation == 'relu' and optimizer == 'adam' and batch_size == 32 and epochs == 10:
        model.save('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/best_feedforward_nn_model.h5')

# Create a df from the full results
nn_tuning_df = pd.DataFrame(results, columns=['Model', 'HL', 'Neurons1', 'Neurons2', 'Activation', 'Optimiser', 'Batch Size', 'Epochs',
                                              'Average Accuracy', 'Average Precision', 'Average Recall', 'Average F1-score',
                                             'Average Training Time'])

# Save the DataFrame to a CSV file
nn_tuning_df.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/nn_fine_tuning_results.csv', index=False)

""" 1 f Loading and testing best model on test dataset"""
# Load the best model
best_model = load_model('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/best_feedforward_nn_model.h5')

# Evaluate the best model on the test dataset
test_predictions = best_model.predict(X_test)
test_predictions = (test_predictions > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, test_predictions)
recall = recall_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions)
f1 = f1_score(y_test, test_predictions)
#
with open("Neural Network model - Saved, loaded and tested on test dataset.txt", "w") as f:
    # Print metrics
    f.write(f"Test Accuracy: {accuracy:.2f} \n")
    f.write(f"Test Recall: {recall:.2f} \n")
    f.write(f"Test Precision: {precision:.2f} \n")
    f.write(f"Test F1-score: {f1:.2f} \n")

    # Generate and print classification report
    class_report = classification_report(y_test, test_predictions)
    f.write("Classification Report: \n")
    f.write(class_report)



""" 1 g Applied context: Using the model on the untagged data to flag which transcript chunks and thus which videos & YouTubers are giving financial advice """
nn_model = load_model('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/best_feedforward_nn_model.h5')

## Combining metadata and transcript chunk data for the untagged transcripts. Model will help identify which of these require further investigation

df_tagged = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/df_tagged_pm.csv')
all_transcript_chunks_with_ID = pd.read_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/df_for_randomisation.csv")
all_transcript_chunks_with_ID = all_transcript_chunks_with_ID.drop_duplicates(subset=['ID', 'combined_sentence'])

all_transcript_chunks_with_ID_advice = pd.merge(all_transcript_chunks_with_ID, df_tagged, on=['ID', 'combined_sentence', 'Source'], how='outer')
all_transcript_chunks_with_ID_advice.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/full_transcript_chunks_AdviceTag.csv', index=False)

untagged_df = all_transcript_chunks_with_ID_advice[all_transcript_chunks_with_ID_advice['Advice'].isnull()] # the untagged dataset
print(untagged_df.shape)
print(untagged_df.columns)

## Have to recreate the embeddings for the untagged ones
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Embed the transcript chunks using same embedding method as we did for the training & testing datasets
# untagged_df['embeddings'] = untagged_df['combined_sentence'].apply(sentence_model.encode)
# untagged_df.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/full_transcript_chunks_AdviceTag_embedding1.csv', index=False)
#
# # import to avoid embedding each time the script runs (can skip the lines above)
untagged_df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/full_transcript_chunks_AdviceTag_embedding1.csv')
untagged_df['embeddings'] = untagged_df['embeddings'].apply(lambda x: [float(val) for val in x[1:-1].split()])

X_untagged = np.array(untagged_df['embeddings'].to_list())

# Make predictions using the trained model
predictions = nn_model.predict(X_untagged)
predicted_labels = (predictions > 0.5).astype(int)

# Add the predicted labels as a new column to the DataFrame
untagged_df['predicted_advice'] = predicted_labels
untagged_df.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/full_transcript_chunks_AdviceTag_embedding1_predicted.csv', index=False)

### Identifying the videos that are predicted to have advice and should be further investigated
video_data_metadata = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/video_data_chunks_count.csv') # the dataset with metadata for each video
video_metadata_predictions = video_data_metadata.merge(untagged_df, on='ID', how='left') # merge with transcript chunks and predictions
video_metadata_predictions = video_metadata_predictions.dropna(subset=['predicted_advice'], axis=0)


# Keep the relevant columns for the transcript chunks that are predicted to have advice
predicted_advice_flag = video_metadata_predictions['predicted_advice'] == 1 # just keep the rows that are predicted to have advice
relevant_columns = ['channelId', 'ID', 'publishedAt', 'tags', 'title', 'description',
       'likeCount', 'viewCount', 'commentCount', 'Transcript', 'combined_sentence',
       'predicted_advice']

video_transcript_chunks_further_investigation = video_metadata_predictions.loc[predicted_advice_flag, relevant_columns]
video_transcript_chunks_further_investigation.to_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/predicted_advice_df_rel_var.csv', index=False)

# With this dataset, I will write a document that investigators could use to identify the YouTube channels and videos predicted to contain advice
# ID the channel and videos for further inspection and write to file for investigators
video_transcript_chunks_further_investigation = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/Embeddings/Predictive model/Datasets/predicted_advice_df_rel_var.csv')

# Group by channelId and video_ID to count unique video IDs and rows (instances of fin advice) per video ID
channel_video_ids_predAdvice = video_transcript_chunks_further_investigation.groupby(['channelId', 'ID']).size().reset_index(name='row_count')
channel_video_counts = video_transcript_chunks_further_investigation.groupby('channelId')['ID'].nunique()

unique_channel_count = video_transcript_chunks_further_investigation['channelId'].nunique()
unique_video_id_count = video_transcript_chunks_further_investigation['ID'].nunique()

# Write to txt file
with open('channel_video_counts.txt', 'w', encoding='utf-8') as f:
    f.write("\n This document outlines the channels and videos that the predictive model has identified as possibly containing financial advice. \n \n")
    f.write("It is meant to aid investigation into financial advice on YouTube investing videos \n by identifying videos (and their creators) that may have financial advice present - thus narrowing the search significantly. \n")
    f.write("\n Video metadata, full transcript and subset of transcript where advice is present is also available \n \n")
    f.write("For any video ID mentioned below, one can watch it by adding the video id to the prefix here: https://www.youtube.com/watch?v= \n \n \n")
    f.write(f'Total number of channels with financial advice: {unique_channel_count}\n') # overall stats to see lie of land
    f.write(f'Total number of videos with financial advice: {unique_video_id_count}\n\n') # overall stats to see lie of land
    f.write("Channels and videos that are predicted to contain financial advice: \n \n")
    # For each channel, which videos contain fin advice and in howm many chunks are there fin advice
    for channel_id, unique_video_count in channel_video_counts.items():
        f.write(f'Channel ID: {channel_id}\n')
        f.write(f'No of videos with financial advice predicted: {unique_video_count}\n')

        # Get the relevant video IDs for the channel
        video_ids = video_transcript_chunks_further_investigation[video_transcript_chunks_further_investigation['channelId'] == channel_id]['ID'].unique()
        f.write(f'Video IDs with financial advice predicted: {", ".join(video_ids)}\n')

        f.write('Video ID, title and number of transcript chunks containing financial advice:\n')

        # Count rows (which are instances of predicted advice) per video ID in the channel
        for video_id in video_ids:
            row_count = channel_video_ids_predAdvice[(channel_video_ids_predAdvice['channelId'] == channel_id)
                                                                      & (channel_video_ids_predAdvice['ID'] == video_id)]['row_count'].values[0]
            video_name = video_transcript_chunks_further_investigation[video_transcript_chunks_further_investigation['ID'] == video_id]['title'].iloc[0]
            f.write(f'   {video_id} - {video_name}: {row_count}\n')

        f.write('\n')