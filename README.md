# Top-N  Music Genre Neural Network 
README for CS 467 Top-N Music Genre Neural Network, by Kyle Donovan, Philip Hopkins, Marco Scandroglio

## Note on usage
This software is distributed as-is, and any use of this software in a way that would violate any laws is strictly prohibited.

## Project Description
This project seeks to automate the process of determining the genre of a provided music sample. Artificial intelligence is a major factor in this project, and neural networks in particular are uniquely positioned to mimic the human functionality of determining genre of music by comparing the provided sample to previous samples of music with defined genres. The project seeks to use Python, and Python packages, to accomplish these tasks. In particular, the conversion of audio files to numeric data will be accomplished using the Librosa package to perform short-time Fourier transforms, and TensorFlow will be used for the neural network. This also presents the opportunity for our group, which has experience with Python but little to no experience with TensorFlow or Librosa, to work with new technologies and expand our knowledge of artificial intelligence while creating a usable product.

## Requirements
This software requires the following Python packages, as well as any associated dependencies:
(*NOTE*: The Python packages required are also found in the requirements.txt file)
+ flask~=3.0.0
+ tensorflow~=2.14.0
+ numpy~=1.26.1
+ librosa~=0.10.1
+ keras~=2.14.0
+ pandas~=2.1.3
+ pydub~=0.25.1
+ yt-dlp~=2023.10.13
+ FFMPEG (installed, and part of the PATH variable)

This software also requires Python to be installed, at a version equal to or greater than 3.xx, and Pythonxx\ as well as the Pythonxx\Scripts folder should both be part of the PATH variable.

## How to Install
1. Download zip of Github repository, then extract. 
2. Using the command line, install dependencies using the following command from the extracted project directory: `pip install -r requirements.txt`
3. If FFMPEG is not installed, install it by following the instructions [here](https://ffmpeg.org/download.html). For Windows, this is most easily done via the command line by `winget install ffmpeg`.
4. Check to make sure that FFMPEG is part of the PATH environment variable. If it is not, add the bin folder for FFMPEG to the PATH variable.

## How to Use the Browser-based Web App
1. Navigate to extracted folder using command line/terminal, then run `app.py`. You need python installed, and then run command `python app.py`.
2. A series of messages should appear in terminal. Using a browser, navigate to the http address listed. Default is http://127.0.0.1:5000
3. Once you have completed the steps to this point, you should have a webpage opened at the address from the command line.
4. At the webpage, you should see three buttons, one to choose file, one to upload, and one to predict genre. 
5. From here, you should be able to upload song files (wav/mp3/au formats) or select a YouTube song link to determine a genre. For further instructions, please see below:
   - Local File Upload: If you want to upload a song file from your computer, select Choose File and find the desired song from your local machine. Then, when the No file chosen text changes to the song file, click the upload button and wait for the genre results to appear.
   - Youtube Link: If you want to predict the genre from a YouTube link, copy and paste the YouTube link into the “Or use a YouTube URL” section and click the “Predict Genre” button and wait for the genre results to appear. 
6. When finished, close the browser window and use Control-C in the terminal to stop the program.

## How to Use the Desktop GUI
1. Navigate to extracted folder using command line/terminal, then run `gui_prediction.py`. You need python installed, and then run command `python gui_prediction.py`.
2. A program with the desktop GUI should pop up.
3. From here, you can choose to either open a music file located on your computer, or type (or paste) in a YouTube URL. You can also choose how many genres from the list you want displayed. Then, press either "Open YouTube URL" or "Open Music File" and wait for the results. 
4. When finished, press the "Exit Program" button.

## How to Use the Content Suggestion System
This Python script provides a simple genre recommendation system based on audio file genre predictions. The recommendation system uses a pre-built database of audio titles and their corresponding genre predictions to suggest content similar to a given input audio file.

### Prerequisites

Make sure the `content_suggestion` directory contains a `recommender_csv` file with rows of data. If not, place audio files in the `content_suggestion/sample_songs` directory so a new `recommender_csv` file can be generated. If you want to generate your own `recommender_csv` file, remove the `recommender_csv` file (if it exists) and add your own audio files to the `content_suggestion/sample_songs` directory. See `instructions.txt` in the `content_suggestion/sample_songs` directory for more information.

### Usage

Execute the script by providing the path to the audio file you want recommendations for. For example:

`python3 content_suggestion.py <path-to-audio-file>`

Replace `<path-to-audio-file>` with the actual path to the audio file for which you want genre recommendations. This will output a sorted table with recommended audio titles and their absolute difference values based on genre predictions.

### Example

Navigate to the extracted folder and run the following command for a demonstration:

`python3 content_suggestion.py ./sample_songs/nirvana_smells_like_teen.mp3`

### Notes

   * The CSV file will have columns: 'title', 'genre_predictions'.
   * Genre predictions are obtained using the predict_genre function from the genre_prediction module.

## Creating Dataset from Scratch
If you would like to create your own dataset from scratch, you can use the song details found in the model_creation/new_dataset_song_details.csv file.

## Credits
Created by Kyle Donovan(https://github.com/kylemdonovan), 
Philip Hopkins(https://github.com/pdhopkins/CS467_music_NN), and 
Marco Scandroglio (https://github.com/marcoscandroglio)

