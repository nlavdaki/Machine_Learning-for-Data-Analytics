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

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

X = DF[['Title','Content']]
y = DF['Label']
#spliting data set , many sizes have been used
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.14)
tokenizerCNN = Tokenizer()
tokenizerCNN.fit_on_texts(X_train['Title'] + X_train['Content'])
sequences = tokenizerCNN.texts_to_sequences(X_train['Title'] + X_train['Content'])
#padding
data = pad_sequences(sequences)
modelCNN = Sequential()
modelCNN.add(Embedding(input_dim=len(tokenizerCNN.word_index) + 1, output_dim=100, input_length=data.shape[1]))
modelCNN.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
modelCNN.add(GlobalMaxPooling1D())
modelCNN.add(Dense(4, activation='softmax'))
num_classes = len(set(DF['Label']))

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#many optimizer have been tested
modelCNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

data = pd.DataFrame(data)
modelCNN.fit(data, y_train, batch_size=32, epochs=5)

# Tokenize test data
test_sequences = tokenizerCNN.texts_to_sequences(X_test['Title'] + X_test['Content'])
test_data = pad_sequences(test_sequences, maxlen=data.shape[1])
test_loss, test_acc = modelCNN.evaluate(test_data, y_test)
print("Test accuracy: ", test_acc)

#we have a good accuracy on testing data so we move forward with the prediction
#loading the unlabeled data
test_withought_labels = pd.read_csv('test_without_labels.csv',sep=',', usecols = ['Id','Title','Content'],  nrows=1000000)
#doing the same procces as above
test_DF = pd.DataFrame(test_withought_labels)
test_DF=test_DF.astype({"Title":'str',"Content":'str'})
#Converting to Lowercase
test_DF['Title'] = test_DF['Title'].apply(str.lower)
test_DF['Content'] = test_DF['Content'].apply(str.lower)
#Remove all the special characters
test_DF['Title'] = test_DF['Title'].apply(lambda x: re.sub(r'\W',' ', x))
test_DF['Content'] = test_DF['Content'].apply(lambda x: re.sub(r'\W',' ', x))
# removing stop words
test_DF['Title']=test_DF['Title'].apply(lambda x: ' '.join(w for w in x.split() if w not in stop_words))
test_DF['Content']=test_DF['Content'].apply(lambda x: ' '.join(w for w in x.split() if w not in stop_words))
# Substituting multiple spaces with single space
test_DF['Title'] = test_DF['Title'].apply(lambda x: re.sub(r'\s+', ' ',x,flags=re.I))
test_DF['Content'] = test_DF['Content'].apply(lambda x: re.sub(r'\s+', ' ',x,flags=re.I))
#tokenizing & lemmatizing
stemmer = WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w,pos="v") for w in w_tokenizer.tokenize(text)]
test_DF['Title'] = test_DF['Title'].apply(lemmatize_text)
test_DF['Title'] = test_DF['Title'].apply(lambda x : " ".join(x))
test_DF['Content'] = test_DF['Content'].apply(lemmatize_text)
test_DF['Content'] = test_DF['Content'].apply(lambda x : " ".join(x))

test_sequences = tokenizerCNN.texts_to_sequences(test_DF['Title'] + test_DF['Content'])
test_data = pad_sequences(test_sequences,maxlen=data.shape[1])
#prediction
predictions = modelCNN.predict(test_data)
class_predictions = np.argmax(predictions, axis=1)
#creating labels
original_labels = le.inverse_transform(class_predictions)
original_labels = np.array(original_labels)
original_labels = np.array([x.capitalize() for x in original_labels])
#storing results
test_DF['Predicted'] = original_labels
test_DF[['Id','Predicted']]
test_DF[['Id','Predicted']].to_csv('ID_labels.csv', index=False, sep=',')