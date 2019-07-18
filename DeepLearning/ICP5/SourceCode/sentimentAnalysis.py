import numpy as np   # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import TensorBoard

data = pd.read_csv('Sentiment.csv')
# Keeping only the necessary columns
data = data[['text','sentiment']]

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

print(data[data['sentiment'] == 'Positive'].size)
print(data[data['sentiment'] == 'Negative'].size)
print(data[data['sentiment'] == 'Neutral'].size)

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
print("tokenizer.texts_to_sequences",X)
X = pad_sequences(X)
print("\n pad_sequences \n",X)

embed_dim = 128
lstm_out = 196

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 32
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
tb = TensorBoard(log_dir="logs/{}",histogram_freq=0, write_graph=True, write_images=True)
model.fit(X_train, Y_train, epochs = 5, batch_size=batch_size, verbose = 2,callbacks=[tb])
model.save('my_model.h5')
model = load_model('my_model.h5')

text = np.array(['A lot of good things are happening. We are respected again throughout the world, and thats a great thing'])
max_df = pd.DataFrame(text, index=range(0, 1, 1), columns=list('t'))
max_df['t'] = max_df['t'].apply(lambda x: x.lower())
max_df['t'] = max_df['t'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
print(max_df)

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
print(tokenizer)
tokenizer.fit_on_texts(max_df['t'].values)
X = tokenizer.texts_to_sequences(max_df['t'].values)
print(X)
X = pad_sequences(X, maxlen=28)
print(model.predict(X))
print("input vector")
print(X)

O = model.predict(X)
print(O)
print(np.where(max(O[0])),":",(max(O[0])))
print(np.argmax(O))