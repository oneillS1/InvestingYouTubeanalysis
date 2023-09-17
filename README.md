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

_Webscraping 1 Medium videos.py_ : A script where videos between 4 and 15 mins long are found via keyword searching channels on YouTube and metadata and transcript data is scraped

_Webscraping 2 Short videos.py_ : A script where videos <4 mins long are found via keyword searching channels on YouTube and metadata and transcript data is scraped

_Webscraping 3 Free search.py_ : A script where videos are found via keyword searching YouTube and metadata and transcript data is scraped

_YouTube_scraping_functions.py_ : A file containing the custom functions written and used to search YouTube for videos, & scrape the necessary data

_Installing & loading packages.py_ : A file where relevant packages are installed/loaded and referenced in other scripts in this section

_Hard_coding.py_ : A file containing some outputs of sections of code that are then hard-coded as inputs to subsequent sections. Only done to reduce time and avoid exceeding YouTube API limits and script can be 
run without these.

_Exploring dataset.py_ : A file containing exploratory analysis of the full scraped dataset

_Description of logic followed.py_ : A file describing the logic followed in the search for relevant channels
