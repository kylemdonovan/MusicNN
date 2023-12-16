# Names: Kyle Donovan, Philip Hopkins, Marco Scandroglio
# Course: CS 467 Fall 2023
# Project: Top-n Music Genre Classification Neural Network
# GitHub Repo: https://github.com/pdhopkins/CS467_music_NN
# Description: Flask app.py for web-based GUI

from flask import Flask, render_template, request
from genre_prediction import predict_genre
from music_file_conversion import youtube_get_audio
import os

app = Flask(__name__)


# function to determine the genre of the uploaded song or YouTube URL
def determine_genre(input_path_or_url: str) -> list:
    """
    Determine the genre of the uploaded song or YouTube URL.

    Args:
        input_path_or_url (str): Path to the uploaded song or YouTube URL.

    Returns:
        list: List of tuples containing genre predictions.

    Example:
        result_list = determine_genre("path/to/song.mp3")
    """

    result_list = predict_genre(input_path_or_url)
    return result_list


@app.route('/')
def index():
    """
    Render the index.html template.

    Returns:
        str: Rendered HTML content.

    Example:
        return index()
    """

    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle file upload or YouTube URL input and determine the genre.

    Returns:
        str: Rendered HTML content with the result list.

    Example:
        return upload()
    """

    result_list = []

    if 'song' in request.files:
        # file upload
        song = request.files['song']

        if song.filename == '':
            return render_template('error.html', error_message="No file selected for upload")

        # save uploaded file to temp location
        upload_path = f"temp_{song.filename}"
        song.save(upload_path)

        # use the uploaded file path in the determine_genre function
        result_list = determine_genre(upload_path)

        # remove temp file
        os.remove(upload_path)

    elif 'url' in request.form:
        # YouTube URL input
        url = request.form['url']

        # make sure this isn't breaking the file upload and results in errors later
        if not url:
            return render_template('error.html', error_message="No URL provided")

        # use the process_youtube_url function
        youtube_get_audio(url)
        result_list = determine_genre("temp_file_youtube.mp3")
        os.remove("temp_file_youtube.mp3")

    return render_template('results.html', result_list=result_list)


if __name__ == '__main__':
    app.run(debug=True)
