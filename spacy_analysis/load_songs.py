import os

def load_songs_from_folder(folder):
    songs = {}
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                song_title = os.path.splitext(filename)[0]
                lyrics = file.read()
                songs[song_title] = lyrics
    return songs

