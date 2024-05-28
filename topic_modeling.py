import os
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


#check
# preprocess
stop_words = set(stopwords.words('english') + [
    "oh", "im", "yeah", "like", "got", "dont", "youre", "ive", "didnt", "well",
    "youll", "cant", "thats", "wanna", "gonna", "cause", "aint", "na", "song", 
    "note", "porch", "late", "fantasy", "said", "kids", "radio", "think", "hope"
])
translator = str.maketrans('', '', string.punctuation)

def preprocess_lyrics(lyric):
    lyric = lyric.translate(translator)  
    tokens = word_tokenize(lyric.lower())
    filtered_tokens = []
    for word in tokens:
        if word not in stop_words:
            filtered_tokens.append(word) 
    return " ".join(filtered_tokens)



# topic modeling
vectorizer = CountVectorizer(stop_words='english')
# NMF
# def topic_modeling_for_song(song_lyrics, num_topics=3):
#     tfidf_matrix = vectorizer.fit_transform([song_lyrics])
#     nmf_model = NMF(n_components=num_topics, random_state=42)
#     nmf_model.fit(tfidf_matrix)
#     feature_names = vectorizer.get_feature_names_out()
#     topic_words = []
#     for topic_idx, topic in enumerate(nmf_model.components_):
#         top_words_idx = topic.argsort()[:-5-1:-1]  # Get indices of top 5 words
#         top_words = [feature_names[i] for i in top_words_idx]
#         topic_words.append(top_words)
#     return topic_words
# LDA
def topic_modeling_for_song(song_lyrics, num_topics=5):
    dt_matrix = vectorizer.fit_transform([song_lyrics])
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(dt_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topic_words = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-3-1:-1]  # Get the top word
        top_words = [feature_names[i] for i in top_words_idx]
        topic_words.append(top_words)
    # Remove duplicate topics
    unique_topic_words = []
    for topic in topic_words:
        if topic not in unique_topic_words:
            unique_topic_words.append(topic)
    return unique_topic_words
# read lyrics from text files
lyrics = []
for filename in os.listdir("country"):
    if filename.endswith(".txt"):
        filepath = os.path.join("country", filename)
        with open(filepath, "r", encoding="utf-8") as file:
            lyrics.append(file.read())

# Perform topic modeling for each song
for i, song_lyrics in enumerate(lyrics):
    preprocessed_lyrics = preprocess_lyrics(song_lyrics)
    song_topics = topic_modeling_for_song(preprocessed_lyrics)
    print(f"Song {i+1} Topics:")
    for j, topic_words in enumerate(song_topics):
         print(f"Topic {j+1}: {', '.join(topic_words)}")
