# You are free to either implement both test() and evaluate() function, or implement test_batch() and evaluate_batch() function. Apart from the 2 functions which you must mandatorily implement, you are free to implement some helper functions as per your convenience.

# Import all necessary python libraries here
# Do not write import statements anywhere else
import os
import pandas as pd
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model
import librosa

TEST_DATA_DIRECTORY_ABSOLUTE_PATH = "/home/pc/test_data"
OUTPUT_CSV_ABSOLUTE_PATH = "/home/pc/output.csv"


# The above two variables will be changed during testing. The current values are an example of what their contents would look like.

def evaluate(file_path):
    
    max_frames=365 
    new_sr=44100
    model = load_model("model_99.keras")
    audio, sr = librosa.load(file_path, sr=None)
    # Resample audio if necessary
    if sr != new_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=new_sr)
    sr = new_sr

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    # Post-process MFCCs to ensure consistent dimensions
    if mfccs.shape[1] < max_frames:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_frames - mfccs.shape[1])), mode='constant')
    elif mfccs.shape[1] > max_frames:
        mfccs = mfccs[:, :max_frames]
        
    if mfccs is not None:
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        predicted_class = np.argmax(model.predict(mfccs))
        if (predicted_class == 0):
            predicted_class = 13
    else: 
        predicted_class = 13 # Default

    # Write your code to predict class for a single audio file instance here
    return predicted_class


def evaluate_batch(file_path_batch, batch_size=32):
    # Write your code to predict class for a batch of audio file instances here
    return predicted_class_batch


def test():
    filenames = []
    predictions = []
    # for file_path in os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
    for file_name in os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
        # prediction = evaluate(file_path)
        absolute_file_name = os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_name)
        
        prediction = evaluate(absolute_file_name)

        filenames.append(absolute_file_name)
        predictions.append(prediction)
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)


def test_batch(batch_size=32):
    filenames = []
    predictions = []

    # paths = os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH)
    paths = os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH)
    paths = [os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, i) for i in paths]
    
    # Iterate over the batches
    # For each batch, execute evaluate_batch function & append the filenames for that batch in the filenames list and the corresponding predictions in the predictions list.

    # The list "paths" contains the absolute file path for all the audio files in the test directory. Now you may iterate over this list in batches as per your choice, and populate the filenames and predictions lists as we have demonstrated in the test() function. Please note that if you use the test_batch function, the end filenames and predictions list obtained must match the output that would be obtained using test() function, as we will evaluating in that way only.
    
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)


# Uncomment exactly one of the two lines below, i.e. either execute test() or test_batch()
test()
# test_batch()