# FOMC-minutes-analysis-and-interest-rate-prediction
Using NLP text analytics and machine learning to predict the interest rate change between two FOMC meetings
We use request and beautiful soup to download all the FOMC minutes from 1968 to 2019 and create different document-word matrix by different algorithm such as bow and tf-idf. 
Then we use machine learning to find a best model to predict the interest rate change direction (up or down) between two FOMC meetings interval, the result is quite promising and we then turn to some industry level data such as REIT index from 1977 to 2018, the result is even much better.

The first thing we did is scrape all the minutes from the FOMC websites.
