# Taylor Swift Song Analysis and Classification

This repository contains the code and analysis for distinguishing between Taylor Swift's early country songs and her more recent pop songs. The project was conducted as part of a Natural Language Processing (NLP) course at Ben-Gurion University.

## Project Overview

Taylor Swift began her career as a country singer with the release of her debut album in 2006. Over the years, her music style evolved, transitioning towards pop. This project aims to analyze and classify the lyrical differences between her country and pop songs using various NLP tools and techniques.

## Methodology

### Data Collection

- **Songs Selection**: Songs were selected from Taylor Swift's discography, focusing on two distinct periods: her early country phase and her later pop phase.
- **Lyrics Source**: Lyrics were collected from publicly available sources.

### Text Processing and Analysis

1. **Tokenization**: Using the `spaCy` library for tokenizing the text into words.
2. **Stop Words Removal**: Common words that do not carry significant meaning were removed using `spaCy`'s predefined list.
3. **Lemmatization**: Converting words to their base forms to reduce variations of the same word.
4. **Unique Words Calculation**: Counting the number of unique words in each song.
5. **POS Tagging**: Analyzing parts of speech (nouns, verbs, adjectives) to identify stylistic differences.

### Tools Used

- **spaCy**: For tokenization, stop words removal, lemmatization, and POS tagging.
- **TextBlob**: For additional text processing and sentiment analysis.

### Key Findings

- **Word Count**: The average word count per song is slightly higher in pop songs compared to country songs.
- **Unique Words**: Pop songs tend to have more unique words than country songs.
- **Parts of Speech**: 
  - **Adjectives**: More prevalent in country songs.
  - **Verbs**: Slightly more common in country songs.
  - **Nouns**: Similar frequency in both genres.

## Repository Structure

- `data/`: Contains the lyrics of the songs analyzed.
- `notebooks/`: Jupyter notebooks used for analysis and visualization.
- `scripts/`: Python scripts for data processing and analysis.
- `results/`: Results of the analysis including plots and summary statistics.

## Installation

To run the code, you will need to install the required Python packages. You can do this using the following command:

```bash
pip install -r requirements.txt
