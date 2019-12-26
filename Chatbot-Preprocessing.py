import nltk, numpy, tflearn, tensorflow, random, json
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

#Preprocessing
with open("intents.json") as file:
	data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern) #Seperate words in JSON
		words.extend(wrds)
		docs_x.append(pattern)
		docs_y.append(intent["tag"])

	if intent["tag"] not in labels:
		labels.append(intent["tag"] )#Add tags to labels dict
