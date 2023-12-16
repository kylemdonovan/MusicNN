# Names: Kyle Donovan, Philip Hopkins, Marco Scandroglio
# Course: CS 467 Fall 2023
# Project: Top-n Music Genre Classification Neural Network
# GitHub Repo: https://github.com/pdhopkins/CS467_music_NN
# Description: TKinter GUI for Genre Prediction

import tkinter as tk
from tkinter import filedialog
from genre_prediction import predict_genre
from music_file_conversion import youtube_get_audio
import os


def open_music_file() -> None:
    """
    Open a music file dialog, predict its genre, and display the results.

    Potential Args:
        NOTE: these are global values that could become arguments.
        list_of_labels (list): List of label widgets to display results.
        window: Reference to the tkinter window.
        number_of_genres (int): Number of genres to display.

    Returns:
        None

    Example:
        open_music_file()
    """

    # Clear old labels
    for each_current_label in list_of_labels:
        each_current_label['text'] = ""
    # Get the prediction from the user selected file
    window.filename = filedialog.askopenfilename(filetypes=((".wav", "*.wav"), (".mp3", "*.mp3")))
    list_of_labels[0]['text'] = f"Results for file: {os.path.basename(window.filename)}"
    result_list = predict_genre(window.filename)
    display_genre_number = number_of_genres.get()
    for each_index in range(display_genre_number):
        list_of_labels[each_index + 1]['text'] = \
            f"{result_list[each_index][0]} : {(result_list[each_index][1] * 100):.4f}%"


def open_url() -> None:
    """
    Open a URL input, download audio, predict its genre, and display the results.

    Potential Args:
        NOTE: these are global values that could become arguments
        list_of_labels (list): List of label widgets to display results.
        url_input: Reference to the tkinter URL input widget.
        number_of_genres (int): Number of genres to display.

    Returns:
        None

    Example:
        open_url()
    """

    # Clear old labels
    for each_current_label in list_of_labels:
        each_current_label['text'] = ""
    url_to_use = url_input.get()
    list_of_labels[0]['text'] = f"Results for URL: {url_to_use}"
    # Get the prediction from the URL
    youtube_get_audio(url_to_use)
    result_list = predict_genre("temp_file_youtube.mp3")
    os.remove("temp_file_youtube.mp3")
    display_genre_number = number_of_genres.get()
    for each_index in range(display_genre_number):
        list_of_labels[each_index + 1]['text'] = \
            f"{result_list[each_index][0]} : {(result_list[each_index][1] * 100):.4f}%"


if __name__ == "__main__":

    # Establish the TKinter window, including dimensions
    window = tk.Tk()
    window.title("Top-N Genre Classification by Donovan, Hopkins, Scandroglio")
    new_width = window.winfo_screenwidth() // 2
    new_height = window.winfo_screenheight() // 2
    window.geometry(f'{new_width}x{new_height}')
    list_of_labels = []
    # Add buttons and layout to TKinter window
    get_file_button = tk.Button(window, text="Open Music File", command=open_music_file, font=("Times New Roman", 16))
    get_file_button.pack()
    url_explain = tk.Label(window, text="Or use a YouTube URL", font=("Times New Roman", 16))
    url_explain.pack()
    url_input = tk.Entry(window, width=50, font=("Times New Roman", 16))
    url_input.pack()
    get_url_button = tk.Button(window, text="Open YouTube URL", command=open_url, font=("Times New Roman", 16))
    get_url_button.pack()
    number_of_genres = tk.IntVar()
    list_genre_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    number_of_genres.set(10)
    user_genre_number = tk.OptionMenu(window, number_of_genres, *list_genre_numbers)
    genre_number_label_text = "Choose how many genres from the list you want displayed:"
    user_genre_number_label = tk.Label(window, text=genre_number_label_text, font=("Times New Roman", 16))
    user_genre_number_label.pack()
    user_genre_number.pack()
    for each_value in range(11):
        list_of_labels.append(tk.Label(window, font=("Times New Roman", 16)))
    for each_label in list_of_labels:
        each_label.pack()
    exit_button = tk.Button(window, text="Exit Program", command=window.quit, font=("Times New Roman", 16))
    exit_button.place(relx=1, anchor="ne")
    # Have the window loop
    window.mainloop()
