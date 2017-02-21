import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
from sklearn.cluster import KMeans


attributes=['Crossing','Finishing','Heading Accuracy',
 'Short Passing','Volleys','Dribbling','Curve',
 'Free Kick Accuracy','Long Passing','Ball Control','Acceleration',
 'Sprint Speed','Agility','Reactions','Balance',
 'Shot Power','Jumping','Stamina','Strength',
 'Long Shots','Aggression','Interceptions','Positioning',
 'Vision','Penalties','Composure','Marking',
 'Standing Tackle','Sliding Tackle','GK Diving',
 'GK Handling','GK Kicking','GK Positioning','GK Reflexes']
 
links=[]   #get all argentinian players
for offset in ['0','100','200']:
    page=requests.get('http://sofifa.com/players?na=52&offset='+offset) 
    soup=BeautifulSoup(page.content,'html.parser')
    for link in soup.find_all('a'):
        links.append(link.get('href'))
links=['http://sofifa.com'+l for l in links if 'player/'in l]  

#pattern regular expression 
pattern=r"""\s*([\w\s]*)"""   #file starts with empty spaces... players name...-other stuff     
for attr in attributes:
    pattern+=r""".*?(\d*\s*"""+attr+r""")"""  #for each attribute we have other stuff..number..attribute..other stuff
pat=re.compile(pattern, re.DOTALL)    #parsing multiline text

rows=[]
for j,link in enumerate(links):
    print (j,link)
    row=[link]
    playerpage=requests.get(link)
    playersoup=BeautifulSoup(playerpage.content,'html.parser')
    text=playersoup.get_text()
    a=pat.match(text)
    row.append(a.group(1))
    for i in range(2,len(attributes)+2):
        row.append(int(a.group(i).split()[0]))
    rows.append(row)
    print (row[1])
df=pd.DataFrame(rows,columns=['link','name']+attributes)
df.to_csv('ArgentinaPlayers.csv',index=False)

##Q3
##for english players would change 
#for offset in ['0','100','200']:
#to
#for offset in ['0','100','200','300','400']:
#and
#page=requests.get('http://sofifa.com/players?na=52&offset='+offset)
#to
#page=requests.get('http://sofifa.com/players?na=14&offset='+offset)

##Q4
clusters = KMeans(5).fit(df.iloc[ : , 3: ].as_matrix())
df['cluster'] = clusters.labels_
df.to_csv('ArgentinaPlayerswithClusters.csv',index=False)

##Q5
##Meaning of clusters:
##cluster 0 are mainly center backs
####w high strength, aggression and tackle attributes
##cluster 1 are mainly strikers
####w high finishing, penalties and shot power
##cluster 2 are all goalkeepers
####and all goalkeepers are cluster 2
##cluster 3 are a mix of midfield and back players
####with generally high attributes (except GK attributes)
###especially marking and tackles, and not so much finishing + free kicks
##cluster 4 are a mix of midfield and forward players
###with generally high atttributes (except GK attributes)
###especially finishing + free kicks, and not so much marking and tackles

##Q6 - using SM's code
centroids = clusters.cluster_centers_

newPlayerAttr = np.array([45, 40, 35, 45, 60, 40, 15]) # Attribute of new player

selectedAttrs = [attributes.index('Crossing'), attributes.index('Sprint Speed'), 
                 attributes.index('Long Shots'), attributes.index('Aggression'),
                 attributes.index('Marking'), attributes.index('Finishing'), attributes.index('GK Handling')]
centroidsSelected = centroids[ : , selectedAttrs]
distCentroids = np.linalg.norm(newPlayerAttr - centroidsSelected, axis = 1)
np.argmin(distCentroids)