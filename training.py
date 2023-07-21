import nltk
import json
import joblib
import random
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

stemmer = PorterStemmer()

def preprocess_data():
    with open('intents.json') as file:
        Corpus = json.load(file)

    # tokenization
    W = set()  # Use a set instead of a list for faster membership checks
    L = set()  # Use a set instead of a list for faster membership checks
    doc_x = []
    doc_y = []

    for intent in Corpus['intents']:
        for pattern in intent['patterns']:
            w_temp = nltk.word_tokenize(pattern)
            w_temp = [w.lower() for w in w_temp if w.isalpha()]  # Filter out non-alphabetic tokens
            w_temp = [stemmer.stem(w) for w in w_temp]

            if w_temp:  # Only add non-empty token lists
                W.update(w_temp)
                doc_x.append(' '.join(w_temp))
                doc_y.append(intent["tag"])

                L.add(intent['tag'])
    return doc_x, doc_y, W, L


def train_and_evaluate():
    doc_x, doc_y, W, L = preprocess_data()

    # Split the preprocessed data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(doc_x, doc_y, test_size=0.2, random_state=42)

    # Convert the preprocessed text into TF-IDF features
    vectorizer = TfidfVectorizer()
    X_train_tf_idf = vectorizer.fit_transform(X_train)
    X_test_tf_idf = vectorizer.transform(X_test)

    # Create and train the SVM classifier
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train_tf_idf, y_train)

    # Make predictions on the testing data
    y_pred = classifier.predict(X_test_tf_idf)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Save the trained model
    joblib.dump(classifier, 'chatbot_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')


def load_model():
    classifier = joblib.load('chatbot_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return classifier, vectorizer