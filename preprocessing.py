import os
import numpy as np
import librosa
import soundfile as sf
from audiomentations import PitchShift, TimeMask, AddGaussianNoise

# from google.colab import drive
# drive.mount('/content/drive')

label_mapping = {
    "car_horn": 1,
    "dog_barking": 2,
    "drilling": 3,
    "Fart": 4,
    "Guitar": 5,
    "Gunshot_and_gunfire": 6,
    "Hi-hat": 7,
    "Knock": 8,
    "Laughter": 9,
    "Shatter": 10,
    "siren": 11,
    "Snare_drum": 12,
    "Splash_and_splatter": 0
}

# Function to compute MFCCs
def compute_mfcc(audio, sr,n_mfcc=20):
    return librosa.feature.mfcc(y=audio, sr=sr,n_mfcc=n_mfcc)  # Renamed for clarity

def compute_mel(audio, sr, n_mels=128):
    return librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)

# Function to process audio files and store MFCCs and labels
def process_audio_files(data_dir, max_frames, new_sr, aug):
    mfcc_data = []
    labels = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                try:
                    audio, sr = librosa.load(file_path)  # Use soundfile for robustness
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=new_sr)  # Resample if needed
                    mfccs = compute_mfcc(audio, new_sr)
                    mfccs = _pad_or_trim(mfccs, max_frames)  # Handle variable-length MFCCs
                    if aug:
                        audio = audio.T
                        pitch_shift = PitchShift(min_semitones=-5, max_semitones=5, p=1.0)
                        audio_sp = pitch_shift(audio, sample_rate=sr)
                        audio_sp = audio_sp.T
                        mfccs_aug = compute_mfcc(audio_sp, sr)
                        mfccs_aug = _pad_or_trim(mfccs_aug, max_frames)
                        mfcc_data.append(mfccs_aug)
                        labels.append(label_mapping[label])

                        # time_mask = TimeMask(min_band_part=0.1, max_band_part=0.15, fade=False, p=1.0)
                        # audio_tm = time_mask(audio, sample_rate=sr)
                        # audio_tm = audio_tm.T
                        # mfccs_tm = compute_mfcc(audio_tm, sr)

                        # gaussian_noise = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=1.0)
                        # audio_gn = gaussian_noise(audio, sample_rate=sr)
                        # audio_gn = audio_gn.T
                        # mfccs_gn = compute_mfcc(audio_gn, sr)

                        # labels.append(label)
                        # labels.append(label)

                    mfcc_data.append(mfccs)
                    labels.append(label_mapping[label])
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    return np.array(mfcc_data), np.array(labels)

def _pad_or_trim(mfccs, max_frames):
    if mfccs.shape[1] < max_frames:
        return np.pad(mfccs, ((0, 0), (0, max_frames - mfccs.shape[1])), mode='constant')
    else:
        return mfccs[:, :max_frames]


train_dir = "audio_dataset/train"
test_dir = "audio_dataset/val"
max_frames = 365
new_sr = 44100

try:
    mfcc_train, labels_train = process_audio_files(train_dir, max_frames, new_sr, False)
    np.save('mel_train.npy', mfcc_train)
    np.save('labels_train.npy', labels_train)
    print(mfcc_train.shape)
    del mfcc_train
    del labels_train
    

    mfcc_test, labels_test = process_audio_files(test_dir, max_frames, new_sr, False)
    np.save('mel_test.npy', mfcc_test)
    np.save('labels_test.npy', labels_test)
    
except Exception as e:
    print(f"Error processing audio files: {str(e)}")


