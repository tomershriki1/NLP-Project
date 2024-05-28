import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np
from load_songs import load_songs_from_folder
from textblob import TextBlob

# Load songs from folders
country_folder = 'data/Taylor Swift - Country'
pop_folder = 'data/Taylor Swift - Pop'
country_songs = load_songs_from_folder(country_folder)
pop_songs = load_songs_from_folder(pop_folder)

nlp = spacy.load("en_core_web_sm")

# Define analysis functions
def unique_word_count(lyrics):
    doc = nlp(lyrics)
    words = set(token.text.lower() for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS)
    return len(words)

def word_count(lyrics):
    doc = nlp(lyrics)
    return len([token.text for token in doc if not token.is_punct])

def adverb_count(lyrics):
    doc = nlp(lyrics)
    return len([token for token in doc if token.pos_ == 'ADV'])

def extract_subjects(lyrics):
    doc = nlp(lyrics)
    subjects = [chunk.text for chunk in doc.noun_chunks if chunk.root.dep_ == 'nsubj']
    return Counter(subjects)

def sentence_length(lyrics):
    doc = nlp(lyrics)
    sentences = list(doc.sents)
    return [len(sent.text.split()) for sent in sentences]

def verb_count(lyrics):
    doc = nlp(lyrics)
    return len([token for token in doc if token.pos_ == 'VERB'])

def named_entities(lyrics):
    doc = nlp(lyrics)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return Counter(entities)

def analyze_sentiment(lyrics):
    blob = TextBlob(lyrics)
    return blob.sentiment.polarity

def analyze_song(lyrics):
    analysis = {}
    analysis['unique_word_count'] = unique_word_count(lyrics)
    analysis['word_count'] = word_count(lyrics)
    analysis['adverb_count'] = adverb_count(lyrics)
    analysis['subjects'] = extract_subjects(lyrics)
    analysis['sentence_lengths'] = sentence_length(lyrics)
    analysis['verb_count'] = verb_count(lyrics)
    analysis['named_entities'] = named_entities(lyrics)
    analysis['sentiment_score'] = analyze_sentiment(lyrics)
    return analysis

# Analyze all songs
country_analyses = {title: analyze_song(lyrics) for title, lyrics in country_songs.items()}
pop_analyses = {title: analyze_song(lyrics) for title, lyrics in pop_songs.items()}

for title, analysis in country_analyses.items():
    print(f'Title: {title}\nAnalysis: {analysis}\n')

# Print analysis for all pop songs
for title, analysis in pop_analyses.items():
    print(f'Title: {title}\nAnalysis: {analysis}\n')

# Calculate average unique word counts
average_country_unique_words = np.mean([analysis['unique_word_count'] for analysis in country_analyses.values()])
average_pop_unique_words = np.mean([analysis['unique_word_count'] for analysis in pop_analyses.values()])

print(f'Average unique word count in country songs: {average_country_unique_words}')
print(f'Average unique word count in pop songs: {average_pop_unique_words}')

# Compare word counts
country_word_counts = [analysis['unique_word_count'] for analysis in country_analyses.values()]
pop_word_counts = [analysis['unique_word_count'] for analysis in pop_analyses.values()]

# Plot the results
plt.figure(figsize=(10, 6))  # Adjust figure size
sns.set_style("whitegrid")  # Adding grid
sns.boxplot(data=[country_word_counts, pop_word_counts], palette=["skyblue", "lightgreen"], notch=True, showfliers=False, linewidth=1.5)
plt.xticks([0, 1], ['Country', 'Pop'], fontsize=12)  # Increase font size
plt.yticks(fontsize=12)  # Increase font size
plt.xlabel('Genre', fontsize=14)  # Label x-axis
plt.ylabel('Unique Word Counts', fontsize=14)  # Label y-axis
plt.title('Unique Word Counts in Country vs. Pop Songs', fontsize=16)  # Add title
plt.grid(True, linestyle='--', alpha=0.7)  # Customize grid
plt.show()
