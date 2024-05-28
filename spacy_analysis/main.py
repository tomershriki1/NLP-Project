import numpy as np
import pandas as pd
from load_songs import load_songs_from_folder
import analysis  # Import analysis functions
import plotting  # Import plotting functions

# Load songs from folders
country_folder = 'data/Taylor Swift - Country'
pop_folder = 'data/Taylor Swift - Pop'
country_songs = load_songs_from_folder(country_folder)
pop_songs = load_songs_from_folder(pop_folder)

# Analyze all songs
country_analyses = {title: analysis.analyze_song(lyrics) for title, lyrics in country_songs.items()}
pop_analyses = {title: analysis.analyze_song(lyrics) for title, lyrics in pop_songs.items()}

# Prepare data for visualization
data = {
    'Title': list(country_analyses.keys()) + list(pop_analyses.keys()),
    'Genre': ['Country'] * len(country_analyses) + ['Pop'] * len(pop_analyses),
    'Unique Word Count': [analysis['unique_word_count'] for analysis in country_analyses.values()] + [analysis['unique_word_count'] for analysis in pop_analyses.values()],
    'Word Count': [analysis['word_count'] for analysis in country_analyses.values()] + [analysis['word_count'] for analysis in pop_analyses.values()],
    'Adverb Count': [analysis['adverb_count'] for analysis in country_analyses.values()] + [analysis['adverb_count'] for analysis in pop_analyses.values()],
    'Verb Count': [analysis['verb_count'] for analysis in country_analyses.values()] + [analysis['verb_count'] for analysis in pop_analyses.values()],
    'Adjective Count': [analysis['adjective_count'] for analysis in country_analyses.values()] + [analysis['adjective_count'] for analysis in pop_analyses.values()],
    'Noun Count': [analysis['noun_count'] for analysis in country_analyses.values()] + [analysis['noun_count'] for analysis in pop_analyses.values()],
    'Lemmatized Word Count': [analysis['lemmatized_word_count'] for analysis in country_analyses.values()] + [analysis['lemmatized_word_count'] for analysis in pop_analyses.values()],
    'Lemmatized Unique Word Count': [analysis['lemmatized_unique_word_count'] for analysis in country_analyses.values()] + [analysis['lemmatized_unique_word_count'] for analysis in pop_analyses.values()],
    'Chorus Count': [analysis['chorus_count'] for analysis in country_analyses.values()] + [analysis['chorus_count'] for analysis in pop_analyses.values()],
    'Verse Count': [analysis['verse_count'] for analysis in country_analyses.values()] + [analysis['verse_count'] for analysis in pop_analyses.values()],
    'Average Lines per Verse': [analysis['average_lines_per_verse'] for analysis in country_analyses.values()] + [analysis['average_lines_per_verse'] for analysis in pop_analyses.values()],
    'Lines in Chorus': [analysis['lines_in_chorus'] for analysis in country_analyses.values()] + [analysis['lines_in_chorus'] for analysis in pop_analyses.values()],
    'Sentiment Score': [analysis['sentiment_score'] for analysis in country_analyses.values()] + [analysis['sentiment_score'] for analysis in pop_analyses.values()]
}

df = pd.DataFrame(data)

# Save and open the charts
plotting.save_charts(df)

# Calculate and display the average number of times "I" and "you" were subjects
average_country_i_subjects = analysis.calculate_subject_averages(country_analyses, 'I')
average_pop_i_subjects = analysis.calculate_subject_averages(pop_analyses, 'I')
average_country_you_subjects = analysis.calculate_subject_averages(country_analyses, 'you')
average_pop_you_subjects = analysis.calculate_subject_averages(pop_analyses, 'you')

data_subjects = {
    'Genre': ['Country', 'Pop', 'Country', 'Pop'],
    'Subject': ['I', 'I', 'you', 'you'],
    'Average Count': [average_country_i_subjects, average_pop_i_subjects, average_country_you_subjects, average_pop_you_subjects]
}

df_subjects = pd.DataFrame(data_subjects)

# Plot subject comparison
plotting.plot_subject_comparison(df_subjects)

# Print analysis for debugging
for title, analysis in country_analyses.items():
    print(f'Title: {title}\nAnalysis: {analysis}\n')

for title, analysis in pop_analyses.items():
    print(f'Title: {title}\nAnalysis: {analysis}\n')

# Calculate average unique word counts
average_country_unique_words = np.mean([analysis['unique_word_count'] for analysis in country_analyses.values()])
average_pop_unique_words = np.mean([analysis['unique_word_count'] for analysis in pop_analyses.values()])

print(f'Average unique word count in country songs: {average_country_unique_words}')
print(f'Average unique word count in pop songs: {average_pop_unique_words}')
