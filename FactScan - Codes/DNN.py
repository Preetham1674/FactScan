import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Load the dataset
df = pd.read_csv(r"C:\Users\preet\Desktop\Project\DATASET\complete_data_set.csv")

# Convert labels to binary numeric values
label_mapping = {"REAL": 1, "True": 1, "true": 1, "real": 1, "FAKE": 0, "Fake": 0, "fake": 0, "-1": 1}
df['label'] = df['label'].replace(label_mapping)

# Drop rows with missing title or label values
df.dropna(subset=['title', 'label'], inplace=True)
df.dropna(subset=["label"], inplace=True)

# Remove non-numeric entries in 'label' column
df = df[~df['label'].apply(lambda x: isinstance(x, str))]

# Remove 10,000 entries with label 0
num_entries_to_delete = 10000
num_entries_label_0 = len(df[df['label'] == 0])

if num_entries_label_0 <= num_entries_to_delete:
    df = df[df['label'] != 0]
else:
    df = df.drop(df[df['label'] == 0].sample(n=num_entries_to_delete).index)

# Preprocess text data
def preprocess_text(text):
    stemmed_content = re.sub('[^a-zA-Z]',' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [PorterStemmer().stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

df['title'] = df['title'].apply(preprocess_text)

# Ensure label column is of the same type
df['label'] = df['label'].astype(int)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['title'], df['label'], test_size=0.2, random_state=42)

# Convert text data to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Build and compile the deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=(X_train_vec.shape[1],)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1, activation='relu')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_vec, y_train, epochs=15, batch_size=32, validation_data=(X_test_vec, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_vec, y_test)
print("Test Accuracy:", accuracy)
