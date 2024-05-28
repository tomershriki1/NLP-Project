import altair as alt
import pandas as pd
import webbrowser
import os

def save_charts(df):
    charts = []

    # Create bar charts for each metric
    metrics = [
        ('Unique Word Count', 'Unique Word Count per Song by Genre'),
        ('Word Count', 'Word Count per Song by Genre'),
        ('Adverb Count', 'Adverb Count per Song by Genre'),
        ('Verb Count', 'Verb Count per Song by Genre'),
        ('Adjective Count', 'Adjective Count per Song by Genre'),
        ('Noun Count', 'Noun Count per Song by Genre'),
        ('Lemmatized Word Count', 'Lemmatized Word Count per Song by Genre'),
        ('Lemmatized Unique Word Count', 'Lemmatized Unique Word Count per Song by Genre'),
        ('Chorus Count', 'Chorus Count per Song by Genre'),
        ('Verse Count', 'Verse Count per Song by Genre'),
        ('Average Lines per Verse', 'Average Lines per Verse per Song by Genre'),
        ('Lines in Chorus', 'Lines in Chorus per Song by Genre'),
        ('Sentiment Score', 'Sentiment Score per Song by Genre')
    ]

    for metric, title in metrics:
        charts.append(alt.Chart(df).mark_bar().encode(
            x=alt.X('Title:N', sort='-y', title='Song Title'),
            y=alt.Y(f'{metric}:Q', title=metric),
            color='Genre:N'
        ).properties(
            title=title,
            width=800,
            height=400
        ).configure_axis(
            labelAngle=45
        ).interactive())

    # Save and open all charts as HTML files
    for i, chart in enumerate(charts):
        filename = f'chart_{i}.html'
        chart.save(filename)
        webbrowser.open('file://' + os.path.realpath(filename))

    print("Charts have been saved as HTML files and opened in the default web browser.")

def plot_subject_comparison(df_subjects):
    chart = alt.Chart(df_subjects).mark_bar().encode(
        x=alt.X('Subject:N', title='Subject'),
        y=alt.Y('Average Count:Q', title='Average Count'),
        color='Genre:N'
    ).properties(
        title='Average Count of Subjects "I" and "You" by Genre',
        width=800,
        height=400
    ).configure_axis(
        labelAngle=0
    ).interactive()

    filename = 'subject_comparison.html'
    chart.save(filename)
    webbrowser.open('file://' + os.path.realpath(filename))
    print("Subject comparison chart has been saved as an HTML file and opened in the default web browser.")
