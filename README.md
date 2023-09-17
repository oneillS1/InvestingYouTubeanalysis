# InvestingYouTubeanalysis

This project is focused on retail investment videos on YouTube and is completed for the MSc Data Science course at Birkbeck. 

Its focus is on the content of retail investment videos on YouTube, using Topic Modelling to understand the topics/themes that arise in the videos. 
In addition, there is a predictive modelling stage where a model is created to predict whether the video contains 'financial advice' as per the regulator's, the FCA, definition.
This is intended as a proof of concept but also has an immediate applied context for regulators.

See full project report for more details.

The project is split into 3 parts which follows the structure in this GitHub repository. 
- The first part 1) involves creating a novel dataset of metadata and transcripts from relevant retail investment videos on YouTube
- The second part 2) uses topic modelling (BERTopic) to understand the themes that are covered in these videos
- The third part 3) creates a predictive model that can predict whether a transcript (and subset) contains 'financial advice' as per the FCA definition.

** 1. Webscraping: Finding relevant videos On YouTube and scraping metadata and transcript data from each video **

_Webscraping 1 Medium videos.py_ : A script where videos between 4 and 15 mins long are found via keyword searching channels on YouTube and metadata and transcript data is scraped <br>
_Webscraping 2 Short videos.py_ : A script where videos <4 mins long are found via keyword searching channels on YouTube and metadata and transcript data is scraped <br>
_Webscraping 3 Free search.py_ : A script where videos are found via keyword searching YouTube and metadata and transcript data is scraped <br>
_YouTube_scraping_functions.py_ : A file containing the custom functions written and used to search YouTube for videos, & scrape the necessary data <br>
_Installing & loading packages.py_ : A file where relevant packages are installed/loaded and referenced in other scripts in this section <br>
_Hard_coding.py_ : A file containing some outputs of sections of code that are then hard-coded as inputs to subsequent sections. Only done to reduce time and avoid exceeding YouTube API limits and script can be 
run without these <br>
_Exploring dataset.py_ : A file containing exploratory analysis of the full scraped dataset <br>
_Description of logic followed.py_ : A file describing the logic followed in the search for relevant channels <br>
<br>

** 2. Topic modelling, using BERTopic **

_Topic modelling.py_ : A file where 3 BERTtopic models (based on different embeddings of the transcripts) are created, visualised and compared <br>
_topic_modelling_metrics_model_1.txt_ : A .txt file containing the metrics for Topic Model 1 <br>
_topic_modelling_metrics_model_1_20.txt_ : A .txt file containing the metrics for Topic Model 1, when topic model updated to have 20 topics <br>
_topic_modelling_metrics_model_1_50.txt_ : A .txt file containing the metrics for Topic Model 1, when topic model updated to have 50 topics <br>
_topic_modelling_metrics_model_2.txt_ : A .txt file containing the metrics for Topic Model 2 <br>
_topic_modelling_metrics_model_3.txt_ : A .txt file containing the metrics for Topic Model 3 <br>
_Model Figures_ : The model figures contains the png and interactive html visualisations of the output of each topic model. The subfolder TM1 refer to topic model 1, TM1_20 to topic model 1 reduced to 20 topics and so on... <br>

** 3. Predictive modelling **

_Manual_tagging_dataset.py_ : A script which takes the full scraped dataset and subsets it for manual tagging of the videos (in prep for the predictive modelling)  <br>
_Predictive_modelling_withTaggedDataset.py_ : A script which builds, evaluates, compares and fine tunes the predictive models. It also uses the model in an applied context <br>
_PredictiveModelling_video_level.py_ : Unused script. Not necessary to the project report <br>
_Neural Network model - Saved, loaded and tested on test dataset.txt_ : A txt file containing the evaluation of the neural network model <br>
_model_evaluation.txt_ : A txt file containing the evaluation scores of the full list of trained models using SMOTE technique <br>
_model_evaluation_main2.txt_ : A txt file containing the evaluation scores of the logistic regression and neural network models (trained primarily for accuracy) <br>
_model_evaluation_main2_recall.txt_ : A txt file containing the evaluation scores of the logistic regression and neural network models (trained primarily for recall) <br>
_model_evaluation_no_smote.txt_ : A txt file containing the evaluation scores of the full list of trained models without using SMOTE technique <br>
_classification_reports.txt_ : An unused txt file containing the evaluation scores of models from the LazyClassifier <br>
_channel_video_counts.txt_ : The .txt output of the application of the best model to an applied context. The document which is immediately applicable to regulatory context <br>
 <br>

The above contain the scripts written for this project. There is one additional folder in the repository for completeness - Datasets. This contains any dataset that is used as an input or output of the scripts in the 3 sections. The naming is the same as in each script. However, the scripts refer to local paths so to re-run one would just have to change the path. GitHub can hold file sizes of under 25 MB. 4 of the datasets created / used in this project are over that threshold. These have been compressed for uploading to GitHub. In one case the file was split in two, just for uploading to GitHub. One would need to append these 2 to rerun the script. This is marked by adding 2 as a suffix to the filename.


