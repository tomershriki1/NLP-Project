import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# Function to read lyrics and label them
def read_lyrics(folder_path, genre):
    lyrics = []
    genres = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                lyrics.append(file.read())
                genres.append(genre)
    return lyrics, genres

# Preprocess the lyrics
def preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text) 
    # Convert to lower case
    tokens = [word.lower() for word in tokens]    
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [word.translate(table) for word in tokens]   
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]   
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]        
    return ' '.join(tokens)

# Read pop and country songs
pop_lyrics, pop_genres = read_lyrics('pop', 'pop')
country_lyrics, country_genres = read_lyrics('country', 'country')

# Combine the lists
lyrics = pop_lyrics + country_lyrics
genres = pop_genres + country_genres

# Preprocess the lyrics
preprocessed_lyrics = [preprocess(lyric) for lyric in lyrics]
y = genres

classifier = SVC(kernel='linear')


# Feature: Raw Text
print("Testing Classifier with Raw Text...")
vectorizer_raw_text = TfidfVectorizer()
X_raw_text = vectorizer_raw_text.fit_transform(lyrics)
X_train_raw_text, X_test_raw_text, y_train_raw_text, y_test_raw_text = train_test_split(X_raw_text, y, test_size=0.2, random_state=42)
classifier_raw_text = SVC(kernel='linear')
classifier.fit(X_train_raw_text, y_train_raw_text)
y_pred_raw_text = classifier.predict(X_test_raw_text)
print("Classifier Report with Raw Text:")
print(classification_report(y_test_raw_text, y_pred_raw_text))
print()


# Feature: Lemmas
print("Testing Classifier with Lemmas...")
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)])
lemmatized_lyrics = [lemmatize_text(lyrics) for lyrics in preprocessed_lyrics]
vectorizer_lemmas = CountVectorizer()
X_lemmas = vectorizer_lemmas.fit_transform(lemmatized_lyrics)
X_train_lemmas, X_test_lemmas, y_train_lemmas, y_test_lemmas = train_test_split(X_lemmas, y, test_size=0.2, random_state=42)
classifier_lemmas = SVC(kernel='linear')
classifier.fit(X_train_lemmas, y_train_lemmas)
y_pred_lemmas = classifier.predict(X_test_lemmas)
print("Classifier Report with Lemmas:")
print(classification_report(y_test_lemmas, y_pred_lemmas))
print()

# Feature: Word Counts
print("Testing Classifier with Word Counts...")
vectorizer_word_counts = CountVectorizer()
X_word_counts = vectorizer_word_counts.fit_transform(preprocessed_lyrics)
X_train_wc, X_test_wc, y_train_wc, y_test_wc = train_test_split(X_word_counts, y, test_size=0.2, random_state=42)
classifier_wc = SVC(kernel='linear')
classifier.fit(X_train_wc, y_train_wc)
y_pred_wc = classifier.predict(X_test_wc)
print("Classifier Report with Word Counts:")
print(classification_report(y_test_wc, y_pred_wc))
print()

# Feature: Sentiment Analysis
print("Testing Classifier with Sentiment Analysis...")
analyzer = SentimentIntensityAnalyzer()
sentiments = []
for lyric in lyrics:
    sentiment = analyzer.polarity_scores(lyric)
    sentiments.append([sentiment['neg'], sentiment['neu'], sentiment['pos']])
X_sentiments = sentiments
X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(X_sentiments, y, test_size=0.2, random_state=42)
classifier_sent = SVC(kernel='linear')
classifier.fit(X_train_sent, y_train_sent)
y_pred_sent = classifier.predict(X_test_sent)
print("Classifier Report with Sentiment Analysis:")
print(classification_report(y_test_sent, y_pred_sent))
print()

# Feature: POS Tags
print("Testing Classifier with POS Tags...")
tokenized_lyrics = [word_tokenize(lyric) for lyric in lyrics]
pos_tags = [pos_tag(tokens) for tokens in tokenized_lyrics]
X_pos_tags = [[tag[1] for tag in tags] for tags in pos_tags]
X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(X_pos_tags, y, test_size=0.2, random_state=42)
X_train_pos_flat = [' '.join(tags) for tags in X_train_pos]
X_test_pos_flat = [' '.join(tags) for tags in X_test_pos]
vectorizer_pos = TfidfVectorizer()
X_train_pos_vec = vectorizer_pos.fit_transform(X_train_pos_flat)
X_test_pos_vec = vectorizer_pos.transform(X_test_pos_flat)
classifier_pos = SVC(kernel='linear')
classifier.fit(X_train_pos_vec, y_train_pos)
y_pred_pos = classifier.predict(X_test_pos_vec)
print("Classifier Report with POS Tags:")
print(classification_report(y_test_pos, y_pred_pos))
print()

# Feature: Linguistic Features
print("Testing Classifier with Linguistic Features...")
lexical_diversity = [len(set(lyric.split())) / len(lyric.split()) for lyric in preprocessed_lyrics]
average_word_length = [sum(len(word) for word in lyric.split()) / len(lyric.split()) for lyric in preprocessed_lyrics]
sentence_length = [len(lyric.split('.')) for lyric in preprocessed_lyrics]

# Combine linguistic features
X_linguistic = [[diversity, length, avg_length] for diversity, length, avg_length in zip(lexical_diversity, sentence_length, average_word_length)]

# Split the data for linguistic features
X_train_linguistic, X_test_linguistic, y_train_linguistic, y_test_linguistic = train_test_split(X_linguistic, y, test_size=0.2, random_state=42)

# Train and evaluate the classifier for linguistic features
classifier_linguistic = SVC(kernel='linear')
classifier.fit(X_train_linguistic, y_train_linguistic)
y_pred_linguistic = classifier.predict(X_test_linguistic)
print("Classifier Report with Linguistic Features:")
print(classification_report(y_test_linguistic, y_pred_linguistic))
print()

