import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import unicodedata
from sklearn.cluster import KMeans

# Q2 and Q3

# Attributes to scrap
attributes=['Crossing','Finishing','Heading Accuracy',
 'Short Passing','Volleys','Dribbling','Curve',
 'Free Kick Accuracy','Long Passing','Ball Control','Acceleration',
 'Sprint Speed','Agility','Reactions','Balance',
 'Shot Power','Jumping','Stamina','Strength',
 'Long Shots','Aggression','Interceptions','Positioning',
 'Vision','Penalties','Composure','Marking',
 'Standing Tackle','Sliding Tackle','GK Diving',
 'GK Handling','GK Kicking','GK Positioning','GK Reflexes']
 
links=[]   # links containing the player pages to get
# Q3: Download first 500
#for offset in ['0','100','200', '300', '400']:
for offset in ['0','100','200']:
    #page=requests.get('http://sofifa.com/players?na=14&offset='+offset) # 14 is for England team
    page=requests.get('http://sofifa.com/players?na=52&offset='+offset) # 52 is for Argentina team
    soup=BeautifulSoup(page.content,'html.parser')
    for link in soup.find_all('a'):
        links.append(link.get('href'))
links=['http://sofifa.com'+l for l in links if 'player/'in l]  # Get all the URL links to players' pages

#pattern regular expression 
pattern=r"""\s*([\w\s]*)"""   #file starts with empty spaces... players name...-other stuff
pattern += r""".*\d*\)\s*?([\w\s]*)Age""" # New addition: extraction of positions
for attr in attributes:
    pattern+=r""".*?(\d*\s*"""+attr+r""")"""  #for each attribute we have other stuff..number..attribute..other stuff
pat=re.compile(pattern, re.DOTALL)    #parsing multiline text

rows=[]
for j, link in enumerate(links):
    print(j, link) # Print the index and URL link
    row=[link] # URL link to player's page
    playerpage=requests.get(link) # Get the player's page content and parse into HTML parser
    playersoup=BeautifulSoup(playerpage.content,'html.parser')
    text=playersoup.get_text()
    text=unicodedata.normalize('NFKD', text).encode('ascii','ignore') # Unicode normalise and encode using ASCII
    a=pat.match(text.decode('ascii')) # Match all the fields specified in the regular expressions
    row.append(a.group(1)) # Player's Name
    row.append(a.group(2).strip()) # New Addition: Position Info
    for i in range(3,len(attributes)+3): # The attributes of the players
        row.append(int(a.group(i).split()[0]))
    rows.append(row) # Append all the information of this player
    print(row[1]) # Print Player Name
df=pd.DataFrame(rows,columns=['link','name','position']+attributes) # Initialise Panda Dataframe
#df.to_csv('EnglishPlayers.csv',index=False) # Write to EnglishPlayers.csv
df.to_csv('ArgentinaPlayers.csv',index=False) # Write to ArgentinaPlayers.csv

# Q4

# Number of clusters
numClusters = 5
         
#engPlayers = pd.read_csv('EnglishPlayers.csv')
argPlayers = pd.read_csv('ArgentinaPlayers.csv')
#engAttrs = engPlayers.iloc[ : , 3: ].as_matrix() # Only use the attributes for clustering
argAttrs = argPlayers.iloc[ : , 3: ].as_matrix() # Only use the attributes for clustering
#kmeans = KMeans(init='k-means++', n_clusters = numClusters, n_init = 10, random_state = 99).fit(engAttrs)
kmeans = KMeans(init='k-means++', n_clusters = numClusters, n_init = 10, random_state = 99).fit(argAttrs)

# Q5

# From the five Excel files, we can see:
# See the following link for football positons, https://en.wikipedia.org/wiki/Association_football_positions
# For 500 English Players
# Cluster 0 = Wide Range of Defenders (Left-Backs, Centre-Backs, Right-Backs)
# Cluster 1 = Mainly Strikers and Attacking Midfielders
# Cluster 2 = Mainly Goalkeepers
# Cluster 3 = Mainly Defensive Midfielders
# Cluster 4 = Mainly Centre-Backs (Defenders)
# For 300 Argentina Players
# Cluster 0 = Mainly Strikers
# Cluster 1 = Mainly Centre-Backs (Defenders)
# Cluster 2 = All Goalkeepers
# Cluster 3 = Mainly Defenders (Left-Backs, Right-Backs) and Defensive Midfielders
# Cluster 4 = Mainly Attacking Midfielders and some strikers
for i in range(numClusters):
    #filename = 'EnglishCluster' + str(i) + '.csv'
    filename = 'ArgentinaCluster' + str(i) + '.csv'
    #engPlayers.iloc[kmeans.labels_ == i, 1:3].to_csv(filename)
    argPlayers.iloc[kmeans.labels_ == i, 1:3].to_csv(filename)

# Q6
newPlayerAttr = np.array([45, 40, 35, 45, 60, 40, 15]) # Attribute of new player

centroids = kmeans.cluster_centers_
#kmeans.fit(reduced_data)

# Indices of the available attributes
selectedAttrs = [attributes.index('Crossing'), attributes.index('Sprint Speed'), 
                 attributes.index('Long Shots'), attributes.index('Aggression'),
                 attributes.index('Marking'), attributes.index('Finishing'), attributes.index('GK Handling')]

# For each centroid, obtain only the values of selected attributes
centroidsSelected = centroids[ : , selectedAttrs]

# Calculate the distance to each of the 5 centroids, using only these selected attributes
distCentroids = np.linalg.norm(newPlayerAttr - centroidsSelected, axis = 1)

# (500 English Players) closest cluster = Cluster with index 4, centre-backs (defenders) cluster
# (300 Argentina Players) closest cluster = Cluster with index 1, centre-backs (defenders) cluster
np.argmin(distCentroids)
         