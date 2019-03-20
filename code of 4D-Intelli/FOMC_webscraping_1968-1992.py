from bs4 import BeautifulSoup
import requests
import re
import urllib.request
import os

base_url = "https://www.federalreserve.gov/monetarypolicy/fomchistorical"

transcript_links = {}
for year in range(1968, 1993):
  html_doc = requests.get(base_url + str(year) +'.htm')
  soup = BeautifulSoup(html_doc.content, 'html.parser')
  links = soup.find_all("a", string=re.compile('Minutes*'))
  link_base_url = "https://www.federalreserve.gov"
  transcript_links[str(year)] = [link_base_url + link["href"] for link in links]
  print("Year Complete: ", year)
  
for year in transcript_links.keys():
    if not os.path.exists("./feddata/" + year):
        os.makedirs("./feddata/" + year)
    for link in transcript_links[year]:
        response = urllib.request.urlopen(str(link))
        name = re.search("[^/]*$", str(link))
        print(link)
        with open("./feddata/" + year + "/" + name.group(), 'wb') as f:
            f.write(response.read())
        print("file uploaded")