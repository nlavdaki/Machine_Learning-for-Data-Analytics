import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
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
# Define the features and the label
DF = DF.reset_index(drop=True)
X = DF[['Title','Content']]
y = DF['Label']
# Split the data into training and testing sets test_size and random_state have been tested in many different values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14, random_state=42)
#1500 unique words, words that have a very low frequency of occurrence are unusually not a good parameter for classifying documents -> Bag of words
vectorizer = CountVectorizer(max_features=1500, min_df=1, max_df=0.7)
# Fit the vectorizer on the x_train  so it that we would have more reliable resutls unbiased from the training set
vectorizer.fit(X_train)
# Transform the all the sets in a split the dataset in transforamtion so it d be possible to assign different weights to each of them
train = vectorizer.transform(X_train['Title']+X_train['Content'])
test = vectorizer.transform(X_test['Title']+X_test['Content'])
# TF-IDF tried as an alternative for better results instead bag of words
#vectorizer = TfidfVectorizer(max_features=1500, min_df=1, max_df=0.7)
#vectorizer.fit(X_train)
#same as above
#train = vectorizer.transform(X_train['Title']+X_train['Content'])
#test = vectorizer.transform(X_test['Title']+X_test['Content'])

#LinearSVC has low scores ~40-50% but it is very fast
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(train, y_train)
accuracy = clf.score(test, y_test)
print("Accuracy: ", accuracy)