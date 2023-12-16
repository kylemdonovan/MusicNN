# Names: Kyle Donovan, Philip Hopkins, Marco Scandroglio
# Course: CS 467 Fall 2023
# Project: Top-n Music Genre Classification Neural Network
# GitHub Repo: https://github.com/pdhopkins/CS467_music_NN
# Description: Content suggestion program for the Genre NN

import os
import sys
import ast
import csv
import numpy as np
import pandas as pd
import genre_prediction as gp


def build_recommender_db(audio_dir: str, file_name: str) -> None:
    """
    Build a database/csv file containing audio titles and genre predictions.

    Args:
        audio_dir (str): The directory containing audio files.
        file_name (str): The name of the CSV file to be created.

    Returns:
        None

    Note:
        - The CSV file will have columns: 'title', 'genre_predictions'.
        - Genre predictions are obtained using the predict_genre function.

    Example:
        build_recommender_db("path/to/audio/files", "audio_database.csv")
    """

    if not os.path.exists(audio_dir):
        print(f'ERROR: {audio_dir} is not a valid directory path')
        return

    file_types = ['.mp3', '.wav']

    # create .csv file with column names
    csv_file_path = f'./{file_name}'
    column_names = ['title', 'genre_predictions']

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_names)

        # iterate over audio files and store track name and prediction vector to .csv file
        for root, sub_dirs, files in os.walk(audio_dir):

            for track_name in files:
                track_path = os.path.join(audio_dir, track_name)
                _, file_extension = os.path.splitext(track_path)
                if file_extension in file_types:
                    track_prediction = gp.predict_genre(track_path, return_list=True)
                    csv_writer.writerow([track_name, f'[{",".join(map(str, track_prediction))}]'])


def calculate_absolute_difference(input_array: list, content_array: list) -> float:
    """
    Calculate the absolute difference between two arrays element-wise.

    Args:
        input_array (List[float]): The first array of numbers.
        content_array (List[float]): The second array of numbers.

    Returns:
        float: The sum of the absolute differences between corresponding elements.

    Example:
        calculate_absolute_difference([1.0, 2.0, 3.0], [4.0, 2.0, 1.0])
    """

    # element-wise difference of arrays
    abs_diff = np.abs(content_array - input_array)
    return np.sum(abs_diff)


def recommender(genre_prediction: list, content_db_dir: str, rec_num: int) -> None:
    """
    Provide content recommendations based on genre predictions.

    Args:
        genre_prediction (List[float]): Genre predictions for one audio file.
        content_db_dir (str): The path to the preprocessed content CSV file.
        rec_num (int): The number of content recommendations to return.

    Returns:
        None

    Note:
        - The content recommendations are determined by finding minimal
        differences in genre prediction values.

    Example:
        recommender([0.1, 0.2, 0.7], "path/to/content_database.csv", 5)
    """

    # load preprocessed content .csv as pandas dataframe object
    df = pd.read_csv(content_db_dir, converters={'genre_predictions': ast.literal_eval})
    # convert lists in dataframe to numpy arrays before calculating differences
    df['genre_predictions'] = df['genre_predictions'].apply(np.array)
    # convert to numpy array
    np_genre_prediction = np.array(genre_prediction)
    # create new column in dataframe with difference values
    df['absolute_difference'] = df['genre_predictions'].apply(
        lambda row: calculate_absolute_difference(np_genre_prediction, row)
    )
    # sort dataframe by difference values
    df = df.sort_values(by='absolute_difference', ascending=True)

    result_df = df.head(rec_num)[['title', 'absolute_difference']]
    result_df_str = result_df.to_string(index=False)
    print(result_df_str)


if __name__ == "__main__":

    # navigate to git directory and run: python3 content_suggestion.py <path-to-audio-file>
    RECOMMENDER_DB = './content_suggestion/recommender_csv'
    PATH_TO_AUDIO = './sample_songs'
    NUMBER_OF_RECOMMENDATIONS = 6

    # the following code extends the functionality of this program
    if len(sys.argv) != 2:
        print()
        print("Usage: python3 content_suggestion.py <path-to-audio-file>")
        sys.exit(1)

    TRACK_FOR_RECOMMENDER = sys.argv[1]

    if not os.path.exists(RECOMMENDER_DB):
        build_recommender_db(PATH_TO_AUDIO, RECOMMENDER_DB)

    prediction_list = gp.predict_genre(TRACK_FOR_RECOMMENDER, return_list=True)
    recommender(prediction_list, RECOMMENDER_DB, NUMBER_OF_RECOMMENDATIONS)
