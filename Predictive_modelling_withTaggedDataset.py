
""" Predictive modelling:

Prediction tasks at 2 levels.
- Level 1: Transcript chunk: Inputs - transcript chunk embedding ; Predict - Advice flag
- Level 2: Video level: Inputs - metadata, transcript chunk embeddings as new variables; Predict - Advice flag
 """

""" Importing necessary packages """
import pandas as pd

""" Level 1 """
""" 1 a. Getting transcript chunks that are tagged """
df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/dataset_for_tagging_tagged.csv')
print(df[['Source', 'Advice']].value_counts())

""" 1 b. Adding the embeddings to use as inputs """


""" 1 c. Splitting into train, validate, test datasets """

""" 1 d. Train models on the training dataset """
## Creating the full dataset
#transcript_chunks_combined_df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/transcript_chunks_combined_df.csv')
#all_video_df_chunks_count = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/all_video_df_chunks_count.csv', usecols=['Source', 'ID'])
#df_for_randomisation = pd.merge(transcript_chunks_combined_df, all_video_df_chunks_count, on='ID')