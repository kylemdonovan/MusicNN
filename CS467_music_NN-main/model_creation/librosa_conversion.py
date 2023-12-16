# Names: Kyle Donovan, Philip Hopkins, Marco Scandroglio
# Course: CS 467 Fall 2023
# Project: Top-n Music Genre Classification Neural Network
# GitHub Repo: https://github.com/pdhopkins/CS467_music_NN
# Description: Librosa conversion of audio files into mel-spectrograms

import os
import librosa
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(y: np.ndarray, sr: float, hop_length: int, y_axis: str = "linear") -> None:
    """
    Plot a visualization of a mel-spectrogram.

    Args:
        y (np.ndarray): Audio time series or spectrogram.
        sr (int): Sampling rate of `y`.
        hop_length (int): Number of samples between successive frames.
        y_axis (str): Scale of the y-axis. Default is "linear".

    Returns:
        None

    Example:
        plot_spectrogram(y, sr, hop_length)
    """

    plt.figure(figsize=(25, 10))
    librosa.display.specshow(y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")


def process_audio_file(audio_file_path: str, plot=False) -> bool:
    """
    Convert an audio file to a mel-spectrogram and save as a .npy file.

    Args:
        audio_file_path (str): Path to the audio file.
        plot (bool): If True, plot the mel-spectrogram. Default is False.

    Returns:
        bool: True if the processing is successful, False otherwise.

    Example:
        process_audio_file("path/to/audio/file.mp3", plot=True)
    """

    # load audio file with Librosa, limiting the duration to first 30 seconds
    try:
        song_length = librosa.get_duration(path=audio_file_path)  # returns a float

        # do not process files with a duration of less than 30 seconds
        if song_length < 30.0:
            print(f"The following audio clip is less than 30 seconds: {audio_file_path}")
            return False

        start_time = 0
        if song_length > 130.0:
            start_time = song_length // 2

        audio_data, sample_rate = librosa.load(audio_file_path, offset=start_time, duration=30.0)

        frame_size = 2048
        hop_size = 512

        # extract Short-Time Fourier Transform
        audio_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_fft=frame_size,
            hop_length=hop_size
        )

        audio_spec_db = librosa.power_to_db(audio_spec, ref=np.max)

        spectrogram_file = os.path.splitext(audio_file_path)[0] + '_mel_spectrogram.txt'
        np.save(spectrogram_file, audio_spec_db)

        if song_length > 130.0:
            y, sr = librosa.load(audio_file_path, offset=start_time - 30.0, duration=30.0)
            frame_size = 2048
            hop_size = 512
            audio_spec = librosa.feature.melspectrogram(y=y, sr=sr,  n_fft=frame_size, hop_length=hop_size)
            audio_spec_db = librosa.power_to_db(audio_spec, ref=np.max)
            spectrogram_file = os.path.splitext(audio_file_path)[0] + '_30bef_mel_spectrogram.txt'
            np.save(spectrogram_file, audio_spec_db)

        if song_length > 130.0:
            y, sr = librosa.load(audio_file_path, offset=start_time - 60.0, duration=30.0)
            frame_size = 2048
            hop_size = 512
            audio_spec = librosa.feature.melspectrogram(y=y, sr=sr,  n_fft=frame_size, hop_length=hop_size)
            audio_spec_db = librosa.power_to_db(audio_spec, ref=np.max)
            spectrogram_file = os.path.splitext(audio_file_path)[0] + '_60bef_mel_spectrogram.txt'
            np.save(spectrogram_file, audio_spec_db)

        if song_length > 130.0:
            y, sr = librosa.load(audio_file_path, offset=start_time + 30.0, duration=30.0)
            frame_size = 2048
            hop_size = 512
            audio_spec = librosa.feature.melspectrogram(y=y, sr=sr,  n_fft=frame_size, hop_length=hop_size)
            audio_spec_db = librosa.power_to_db(audio_spec, ref=np.max)
            spectrogram_file = os.path.splitext(audio_file_path)[0] + '_30aft_mel_spectrogram.txt'
            np.save(spectrogram_file, audio_spec_db)

        # plot spectrograms
        if plot:
            plot_spectrogram(audio_spec_db, sample_rate, hop_size)
            plt.title(f'Mel-Spectrogram for {os.path.basename(audio_file_path)} (30 seconds)')

        return True

    except Exception as e:
        print(e)
        print(f"Error loading: {audio_file_path}")
        return False


def process_audio_database(audio_directory: str) -> None:
    """
    Convert audio files in a directory to mel-spectrograms using process_audio_file.

    Args:
        audio_directory (str): Directory containing audio files.

    Returns:
        None

    Example:
        process_audio_database("path/to/audio/directory")
    """

    # process all audio files in the folder
    # edit to individual if individual file processing desired
    file_count = 0
    for root, dirs, files in os.walk(audio_directory):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.au')):
                audio_file = os.path.join(root, file)
                if process_audio_file(audio_file):
                    file_count += 1
                    print(f"Processed file {file_count}", end="\r")

    print(f"Processed {file_count} files!")

# show the spectrograms
# plt.show()


if __name__ == "__main__":
    DIRECTORY = 'genres_original'
    process_audio_database(DIRECTORY)
