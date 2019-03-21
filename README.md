# FOMC-minutes-analysis-and-interest-rate-prediction
Using NLP text analytics and machine learning to predict the interest rate change between two FOMC meetings
We use request and beautiful soup to download all the FOMC minutes from 1968 to 2019 and create different document-word matrix by different algorithm such as bow and tf-idf. 
Then we use machine learning to find a best model to predict the interest rate change direction (up or down) between two FOMC meetings interval, the result is quite promising and we then turn to some industry level data such as REIT index from 1977 to 2018, the result is even much better.

At the beginning, we have to use webscarping to download the minutes from FOME website:
import re
import urllib.request
import os

base_url = "https://www.federalreserve.gov/monetarypolicy/fomchistorical"

transcript_links = {}
for year in range(1968, 1993):
  html_doc = requests.get(base_url + str(year) +'.htm') # get the link
  soup = BeautifulSoup(html_doc.content, 'html.parser') # extra the content
  links = soup.find_all("a", string=re.compile('Minutes*')) # find all links in the content
  print(links)
  link_base_url = "https://www.federalreserve.gov"
  transcript_links[str(year)] = [link_base_url + link["href"] for link in links] # store all links in each year
  print("Year Complete: ", year)
  
