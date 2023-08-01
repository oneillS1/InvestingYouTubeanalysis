
import pandas as pd

df = pd.read_csv('C:/Users/Steve.HAHAHA/Desktop/Dissertation/dataset_for_tagging_tagged.csv')
print(df[['Source', 'Advice']].value_counts())