import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')
nltk.download('wordnet')




#Preprocess
#loading training data
trData = pd.read_csv('train.csv', sep=',', usecols = ['Id', 'Title','Content','Label'], error_bad_lines=False, nrows=1000000)
df=  pd.DataFrame(trData)
#clean rows of Data with NaN/None values
DF=df.dropna()
#making all data strings so that I can do the cleaning (funct not applied on floats)
DF=DF.astype({"Title":'str',"Content":'str',"Label":'str'})
stop_words = set(stopwords.words('english'))
#Converting to Lowercase
DF['Title'] = DF['Title'].apply(str.lower)
DF['Content'] = DF['Content'].apply(str.lower)
DF['Label'] = DF['Label'].apply(str.lower)
#Remove all the special characters
DF['Title'] = DF['Title'].apply(lambda x: re.sub(r'\W',' ', x))
DF['Content'] = DF['Content'].apply(lambda x: re.sub(r'\W',' ', x))
DF['Label'] = DF['Label'].apply(lambda x: re.sub(r'\W',' ', x))
# removing stop words
DF['Title']=DF['Title'].apply(lambda x: ' '.join(w for w in x.split() if w not in stop_words))
DF['Content']=DF['Content'].apply(lambda x: ' '.join(w for w in x.split() if w not in stop_words))
DF['Label']=DF['Label'].apply(lambda x: ' '.join(w for w in x.split() if w not in stop_words))
# Substituting multiple spaces with single space
DF['Title'] = DF['Title'].apply(lambda x: re.sub(r'\s+', ' ',x,flags=re.I))
DF['Content'] = DF['Content'].apply(lambda x: re.sub(r'\s+', ' ',x,flags=re.I))
DF['Label'] = DF['Label'].apply(lambda x: re.sub(r'\s+', ' ',x,flags=re.I))
#tokenizing & lemmatizing
stemmer = WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w,pos="v") for w in w_tokenizer.tokenize(text)]
DF['Title'] = DF['Title'].apply(lemmatize_text)
DF['Title'] = DF['Title'].apply(lambda x : " ".join(x))

DF['Content'] = DF['Content'].apply(lemmatize_text)
DF['Content'] = DF['Content'].apply(lambda x : " ".join(x))

DF['Label'] = DF['Label'].apply(lemmatize_text)
DF['Label'] = DF['Label'].apply(lambda x : " ".join(x))
#droping all labels that arent one of the 4 cat
DF = DF[DF["Label"].isin(['health','technology','entertainment','business'])]
# Create an instance of LabelEncoder
le = LabelEncoder()
# Fit the encoder on the 'Label' column of the dataframe
le.fit(DF['Label'])
# Replace the 'Label' column of the dataframe with the encoded values
DF['Label'] = le.transform(DF['Label'])

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(DF['Title']+DF['Content'], DF['Label'], test_size=0.14)

# Vectorize the text data 
vectorizerRF = TfidfVectorizer(stop_words='english')
X = vectorizerRF.fit_transform(X_train)

X_train = vectorizerRF.fit_transform(X_train)
X_test = vectorizerRF.transform(X_test)

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Print the accuracy of the classifier
print("Accuracy: ", accuracy_score(y_test, y_pred))