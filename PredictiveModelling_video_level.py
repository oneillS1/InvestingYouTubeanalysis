
                    """ Predictive modelling at the video level """

""" Code is written for the first parts but decision taken not to proceed for now due to sucess of model in the 
predicitve modelling at transcript chunk level being sufficient to id which videos without use of metadata """

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

