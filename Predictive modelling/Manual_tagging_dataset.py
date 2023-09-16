
""" The predictive modelling requires the manual tagging of a subset of the transcript chunks. This
 script chooses the subset which is then tagged """

import pandas as pd

""" 1. FCA rules on what constitutes financial advice 
"""
# Value judgment on the merit or demerit of buying or selling a specific financial instrument (crypto included and equities)

""" 2. Randomly selecting subset """
# Dataset = 16899 chunks of transcript - 9ish chunks on average
transcript_chunks_combined_df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/transcript_chunks_combined_df.csv')
all_video_df_chunks_count = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/all_video_df_chunks_count.csv', usecols=['Source', 'ID'])
df_for_randomisation = pd.merge(transcript_chunks_combined_df, all_video_df_chunks_count, on='ID') # adding source to ensure some of each source is tagged e.g., free search, short and medium videos
# print(df_for_randomisation.head())
df_for_randomisation.to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/df_for_randomisation.csv", index=False)

# Stratified sampling - randomly select 50% of the 3 types of Sourced videos for tagging (in the end a subset of this 50% was tagged due to time taken to tag -  all sources were tagged however)
num_ids_to_select = (df_for_randomisation.groupby('Source')['ID'].nunique() * 0.5).astype(int) # ensuring 50% of video ids from each source

selected_dfs = []
for source in df_for_randomisation['Source'].unique(): # selecting transcripts from all sources
    group_df = df_for_randomisation[df_for_randomisation['Source'] == source]
    # Sample 'num_ids_to_select' unique IDs from the group
    sampled_ids = group_df['ID'].drop_duplicates().sample(n=int(num_ids_to_select[source]), random_state=42)
    # just pick the selected ids and append to list to subsequently make new dataframe just for tagging
    selected_dfs.append(group_df[group_df['ID'].isin(sampled_ids)])

selected_ids_df = pd.concat(selected_dfs)
print(selected_ids_df.groupby('Source')['ID'].nunique())
print(selected_ids_df['Source'].value_counts())
selected_ids_df.to_csv("C:/Users/Steve.HAHAHA/Desktop/Dissertation/dataset_for_tagging.csv", index=False)

df_tagging = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/dataset_for_tagging.csv') # to be tagged dataset, printed and tagged manually
print(df_tagging['Source'].value_counts())