from bs4 import BeautifulSoup
import requests
import re
import urllib
import os

base_url = "https://www.federalreserve.gov/monetarypolicy/fomchistorical"

transcript_links = {}
for year in range(2008, 2014):
  html_doc = requests.get(base_url + str(year) +'.htm')
  soup = BeautifulSoup(html_doc.content, 'html.parser')
  links = soup.find_all("a", string=re.compile('HTML*'))
  print(links)
  link_base_url = "https://www.federalreserve.gov"
  transcript_links[str(year)] = [link_base_url + link["href"] for link in links]
  print("Year Complete: ", year)
for year in transcript_links.keys():
    if not os.path.exists("./feddata/" + year):
        os.makedirs("./feddata/" + year)
    for link in transcript_links[year]:
        try:
            html = requests.get(link)
            soup = BeautifulSoup(html.content, 'html.parser')

		# kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()    # rip it out

		# get text
            text = soup.get_text()

		# break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
		# break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
		# drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            name = re.search("[^/]*$", str(link))
            print(link)
            with open("./feddata/" + year + "/" + name.group()+'.txt', 'w',encoding='utf-8') as f:
                f.write(text)
            print("file uploaded")
        except:
            continue