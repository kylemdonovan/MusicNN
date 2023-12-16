# Names: Kyle Donovan, Philip Hopkins, Marco Scandroglio
# Course: CS 467 Fall 2023
# Project: Top-n Music Genre Classification Neural Network
# GitHub Repo: https://github.com/pdhopkins/CS467_music_NN
# Description: Converts music files to mp3 for use in the NN,
#              and includes function to get audio from YouTube

import pydub
import yt_dlp


def audio_conversion(file_path: str, new_format: str) -> None:
    """
    Convert audio file to a specified format using pydub.

    Args:
        file_path (str): Path to the input audio file.
        new_format (str): Desired format for the output file.

    Returns:
        None

    Example:
        audio_conversion("path/to/audio/file.mp3", "wav")
    """

    audio_piece = pydub.AudioSegment.from_file(file_path)
    # Make sure new filepath is the correct format
    audio_piece.export(file_path + "." + new_format, format=new_format)

# #############
# modified from https://stackoverflow.com/questions/27473526/
# download-only-audio-from-youtube-video-using-youtube-dl-in-python-script/
# #############


def youtube_get_audio(video_url: str, genre_string: str = "temp_file_youtube") -> None:
    """
    Download audio from a YouTube video using youtube-dl.

    Args:
        video_url (str): YouTube video URL.
        genre_string (str): String for the output file name.

    Returns:
        None

    Example:
        youtube_get_audio("https://www.youtube.com/watch?v=your_video_id", "temp_file_youtube")
    """
    # set options to have the output be only audio, and filename as chosen
    ydl_opts = {
        'outtmpl': f'{genre_string}',
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


if __name__ == "__main__":
    songs_to_convert = [
        "https://www.youtube.com/watch?v=9VSerKr1vBM",
        "https://www.youtube.com/watch?v=cTC1TEVo3Aw",
        "https://www.youtube.com/watch?v=NtBwVWWa3Ss",
        "https://www.youtube.com/watch?v=EnhIIGfOw4A",
        "https://www.youtube.com/watch?v=7JR10AThY8M",
        "https://www.youtube.com/watch?v=xoLE2yEvuPM"
    ]
    genre = "valrock"
    current_index = 0
    for each_song in songs_to_convert:
        current_song = genre + str(current_index)
        youtube_get_audio(each_song, current_song)
        current_index += 1
