# Names: Kyle Donovan, Philip Hopkins, Marco Scandroglio
# Course: CS 467 Fall 2023
# Project: Top-n Music Genre Classification Neural Network
# GitHub Repo: https://github.com/pdhopkins/CS467_music_NN
# Description: Genre prediction program for the Genre NN

# imports
import os
import json
import numpy as np
import tensorflow as tf
import librosa


def save_json_genre_labels() -> None:
    """
    Save genre labels in a JSON file_to_check based on the structure of the 'genres_original' directory.

    Raises:
        FileNotFoundError: If 'genres_original' directory does not exist or is not accessible.

    Returns:
        None

    Example:
        save_json_genre_labels()
    """

    data_directory = "genres_original"
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(f'{data_directory} does not exist or is not accessible.')
    dict_genre_labels = {}
    last_unused_label = 0

    # iterate over check_files in subfolders
    for check_root, sub_dirs, check_files in os.walk(data_directory):
        subfolder_name = os.path.basename(check_root)

        for file_to_check in check_files:
            if file_to_check.endswith(".npy"):
                # build lists of matrices and labels
                if subfolder_name not in dict_genre_labels:
                    dict_genre_labels[subfolder_name] = last_unused_label
                    last_unused_label += 1

    with open("genre_labels.json", "w") as output_file:
        json.dump(dict_genre_labels, output_file)


def process_audio_file(audio_file_to_process: str) -> np.ndarray:
    """
    Load an audio file with Librosa, limiting the duration to the first 30 seconds.
    If the audio file has a duration greater than 65 second then the beginning of the
    30 second sample starts at the middle of the audio files duration.

    Args:
        audio_file_to_process (str): The path to the audio file.

    Returns:
        np.ndarray: Processed audio spectrogram as a NumPy array.

    Example:
        processed_audio = process_audio_file("path/to/audio/file.mp3")
    """

    # load audio file with Librosa, limiting the duration to first 30 seconds
    song_length = librosa.get_duration(path=audio_file_to_process)
    start_time = 0
    if song_length > 65.0:
        start_time = song_length // 2

    y, sr = librosa.load(audio_file_to_process, offset=start_time, duration=30.0)

    frame_size = 2048
    hop_size = 512

    # extract Short-Time Fourier Transform
    audio_spec = librosa.feature.melspectrogram(y=y, sr=sr,  n_fft=frame_size, hop_length=hop_size)

    audio_spec_db = librosa.power_to_db(audio_spec, ref=np.max)

    audio_spec_db = audio_spec_db[:, :, np.newaxis]
    audio_spec_db = np.expand_dims(audio_spec_db, axis=0)
    return audio_spec_db


def predict_genre(audio_file_dir: str, return_list=False) -> list:
    """
    Predict genre(s) for an audio file and return the results.

    Args:
        audio_file_dir (str): The path to the audio file.
        return_list (bool): If True, return a list of percentages only.

    Returns:
        Union[List[float], List[tuple]]: List of genre predictions with percentages.

    Example:
        predict_genre("path/to/audio/file.mp3", "my_trained_model", return_list=True)
    """

    # Load the saved model and use it to predict the genre
    audio_file_array = process_audio_file(audio_file_dir)
    trained_model = tf.saved_model.load("model_saved")
    results = trained_model.serve(audio_file_array)
    # Change results to a readable format
    results = results.numpy()
    results = results.flatten()
    results = results.tolist()

    with open("genre_labels.json") as input_file:
        loaded_json_genres = json.load(input_file)

    results_dictionary = {}
    for each_key in loaded_json_genres.keys():
        results_dictionary[each_key] = results[loaded_json_genres[each_key]]

    results_list = results_dictionary.items()
    sorted_results = sorted(results_list, key=lambda genre: genre[1], reverse=True)

    if return_list:
        return [x[1] * 100 for x in results_list]  # list of percentages only

    return sorted_results


if __name__ == "__main__":
    # Predict each song in sample_songs
    for root, dirs, files in os.walk("sample_songs"):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.au')):
                audio_file = os.path.join(root, file)
                predict_results = predict_genre(audio_file)
                print(audio_file)
                for each_tuple in predict_results:
                    if each_tuple[1] * 100 > 1:
                        print(
                            f"{each_tuple[0]} : {(each_tuple[1] * 100):.4f} %")
