Taylor Swift Song Analysis and Classification
This repository contains the code and analysis for distinguishing between Taylor Swift's early country songs and her more recent pop songs. The project was conducted as part of a Natural Language Processing (NLP) course at Ben-Gurion University.

Project Overview
Taylor Swift began her career as a country singer with the release of her debut album in 2006. Over the years, her music style evolved, transitioning towards pop. This project aims to analyze and classify the lyrical differences between her country and pop songs using various NLP tools and techniques.

Methodology
Data Collection
Songs Selection: Songs were selected from Taylor Swift's discography, focusing on two distinct periods: her early country phase and her later pop phase.
Lyrics Source: Lyrics were collected from publicly available sources.
Text Processing and Analysis
Tokenization: Using the spaCy library for tokenizing the text into words.
Stop Words Removal: Common words that do not carry significant meaning were removed using spaCy's predefined list.
Lemmatization: Converting words to their base forms to reduce variations of the same word.
Unique Words Calculation: Counting the number of unique words in each song.
POS Tagging: Analyzing parts of speech (nouns, verbs, adjectives) to identify stylistic differences.
Tools Used
spaCy: For tokenization, stop words removal, lemmatization, and POS tagging.
TextBlob: For additional text processing and sentiment analysis.
Key Findings
Word Count: The average word count per song is slightly higher in pop songs compared to country songs.
Unique Words: Pop songs tend to have more unique words than country songs.
Parts of Speech:
Adjectives: More prevalent in country songs.
Verbs: Slightly more common in country songs.
Nouns: Similar frequency in both genres.
Repository Structure
data/: Contains the lyrics of the songs analyzed.
notebooks/: Jupyter notebooks used for analysis and visualization.
scripts/: Python scripts for data processing and analysis.
results/: Results of the analysis including plots and summary statistics.
Installation
To run the code, you will need to install the required Python packages. You can do this using the following command:

bash
Copy code
pip install -r requirements.txt
Usage
Running the Analysis
Clone this repository:
bash
Copy code
git clone https://github.com/tomershriki1/NLP-Project.git
cd NLP-Project
Ensure you have all the necessary libraries installed:
bash
Copy code
pip install -r requirements.txt
Run the Jupyter notebooks in the notebooks/ directory to see the analysis steps and results.
Scripts
scripts/text_processing.py: Contains functions for text preprocessing including tokenization, stop words removal, and lemmatization.
scripts/analysis.py: Functions for performing the main analysis including unique word counts and POS tagging.
Results
Results of the analysis can be found in the results/ directory, including visualizations and summary statistics that highlight the differences between Taylor Swift's country and pop songs.

Conclusion
This project demonstrates how natural language processing techniques can be applied to analyze and distinguish between different musical styles through lyrical content. The findings provide insights into the stylistic evolution of Taylor Swift's music from country to pop.
