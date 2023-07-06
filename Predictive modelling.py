
""" The predictive modelling to see can one predict whether a YouTube investing video is advice / guidance / information """

import pandas as pd


""" 1. FCA rules on what constitutes financial advice 
"""

""" 2. Randomly selecting train, validate and test sets """
# Dataset = 1890 videos, 17958 chunks of transcript - 9ish chunks on average
transcript_chunks_combined_df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/transcript_chunks_combined_df.csv')
all_video_df_chunks_count = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/all_video_df_chunks_count.csv', usecols=['Source', 'ID'])
print(transcript_chunks_combined_df.head())
print(all_video_df_chunks_count.head())

# Stratified sampling - randomly select 50% of the 3 types of Sourced videos for tagging
df_for_randomisation = pd.merge(transcript_chunks_combined_df, all_video_df_chunks_count, on='ID')

num_ids_to_select = df_for_randomisation.groupby('Source')['ID'].nunique() * 0.5

# Create an empty list to store the selected DataFrames
selected_dfs = []

# Iterate over each unique Source
for source in df_for_randomisation['Source'].unique():

    group_df = df_for_randomisation[df_for_randomisation['Source'] == source]

    # Sample 'num_ids_to_select' unique IDs from the group
    sampled_ids = group_df['ID'].drop_duplicates().sample(n=int(num_ids_to_select[source]), random_state=42)

    # Filter the original DataFrame for the sampled IDs and append to 'selected_dfs'
    selected_dfs.append(group_df[group_df['ID'].isin(sampled_ids)])

# Concatenate the selected DataFrames into a single DataFrame
selected_ids_df = pd.concat(selected_dfs)

print(selected_ids_df.groupby('Source')['ID'].nunique())
selected_ids_df.to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/dataset_for_tagging.csv", index=False)