import spacy
import re
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

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

def adjective_count(lyrics):
    doc = nlp(lyrics)
    return len([token for token in doc if token.pos_ == 'ADJ'])

def noun_count(lyrics):
    doc = nlp(lyrics)
    return len([token for token in doc if token.pos_ == 'NOUN'])

def lemmatized_word_count(lyrics):
    doc = nlp(lyrics)
    lemmatized_words = [token.lemma_ for token in doc if not token.is_punct]
    return len(lemmatized_words)

def lemmatized_unique_word_count(lyrics):
    doc = nlp(lyrics)
    lemmatized_words = set(token.lemma_ for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS)
    return len(lemmatized_words)

def chorus_count(lyrics):
    return len(re.findall(r'\bchorus\b', lyrics.lower()))

def verse_count(lyrics):
    return len(re.findall(r'\bverse\b', lyrics.lower()))

def average_lines_per_verse(lyrics):
    verses = [section.strip() for section in lyrics.lower().split('verse') if section.strip()]
    lines_in_verses = [len(verse.split('\n')) for verse in verses]
    return sum(lines_in_verses) / len(lines_in_verses) if lines_in_verses else 0

def lines_in_chorus(lyrics):
    choruses = [section.strip() for section in lyrics.lower().split('chorus') if section.strip()]
    return len(choruses[0].split('\n')) if choruses else 0

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
    analysis['adjective_count'] = adjective_count(lyrics)
    analysis['noun_count'] = noun_count(lyrics)
    analysis['lemmatized_word_count'] = lemmatized_word_count(lyrics)
    analysis['lemmatized_unique_word_count'] = lemmatized_unique_word_count(lyrics)
    analysis['chorus_count'] = chorus_count(lyrics)
    analysis['verse_count'] = verse_count(lyrics)
    analysis['average_lines_per_verse'] = average_lines_per_verse(lyrics)
    analysis['lines_in_chorus'] = lines_in_chorus(lyrics)
    analysis['sentiment_score'] = analyze_sentiment(lyrics)
    return analysis
