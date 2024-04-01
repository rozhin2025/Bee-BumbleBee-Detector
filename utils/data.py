import numpy as np
import pandas as pd
import librosa
from .audio_preprocessing import segmentation, normalize
from scipy.signal import butter, filtfilt

def train_test_split_by_bee(rec_paths, test_size=0.3, exclude_noisy_bee=True, random_state=None):
    '''
    Split the recordings into train and test setS, balancing the total duration of the bees in each set.
    rec_paths: list of Path objects, the paths to the recordings.
    test_size: float, the proportion of recordings to be in the test set.
    exclude_noisy_bee: bool, whether to exclude the frames with noisy bee.
    random_state: int, the random seed.
    return: tuple of lists of Path objects, the train and test sets.
    '''
    
    rng = np.random.default_rng(random_state)
    rec_paths = sorted(rec_paths)
    rng.shuffle(rec_paths)
    bee_durations = [pd.read_csv(rec_path.with_suffix('.csv')) for rec_path in rec_paths]
    if exclude_noisy_bee:
        bee_durations = [bee_duration[bee_duration['label'].str.lower().str.strip() != 'noisybee'] for bee_duration in bee_durations]
    bee_durations = [(labels.end - labels.start).sum() for labels in bee_durations]
    bee_durations = np.array(bee_durations)
    bee_durations_cumsum = bee_durations.cumsum()
    
    test_threshold = bee_durations.sum() * test_size
    train_threshold = bee_durations.sum() * (1 - test_size)

    if np.min(np.abs(bee_durations_cumsum - test_threshold)) < np.min(np.abs(bee_durations_cumsum - train_threshold)):
        test_idx = np.argmin(np.abs(bee_durations_cumsum - test_threshold))
        test_set = rec_paths[:test_idx+1]
        train_set = rec_paths[test_idx+1:]
    else:
        test_idx = np.argmin(np.abs(bee_durations_cumsum - train_threshold))
        test_set = rec_paths[test_idx+1:]
        train_set = rec_paths[:test_idx+1]
    return train_set, test_set


def load_annotations_by_file(rec_path, frame_length_seconds, hop_length_seconds):
    '''
    Load the annotations for a single recording file.
    rec_path: Path object, the path to the recording file.
    frame_length_seconds: float, the length of the frames in seconds.
    hop_length_seconds: float, the hop length in seconds.
    return: pandas DataFrame, the annotations per sample.
    '''
    
    annots_file = rec_path.with_suffix('.csv')
    file_duration = librosa.get_duration(path=rec_path)
    annotations = pd.DataFrame(
        data=np.arange(0, file_duration + frame_length_seconds, hop_length_seconds),
        columns=['start']
    )
    annotations['end'] = annotations['start'] + frame_length_seconds
    annotations = annotations[annotations['end'] <= file_duration]
    annotations_raw = pd.read_csv(annots_file)
    annotations_raw['label'] = annotations_raw['label'].str.lower().str.strip()
    for label in annotations_raw['label'].unique():
        annotations_raw_label = annotations_raw[annotations_raw['label'] == label][['start', 'end']].values
        annotations[label] = annotations.apply(lambda x: np.diff(np.clip(annotations_raw_label, x.start, x.end), axis=1).sum(), axis=1) / frame_length_seconds
    annotations['recording'] = rec_path
    annotations['sample_idx'] = np.arange(len(annotations))
    return annotations


def select_samples_by_annotation(annotations_, exclude_noisybee=True, bee_fraction_threshold=1, eps=1e-6):
    '''
    Select the samples based on the annotations.
    annotations_: pandas DataFrame, the annotations per sample.
    exclude_noisybee: bool, whether to exclude the frames with noisy bee.
    bee_fraction_threshold: float, the threshold for the bee fraction.
    eps: float, a small number.
    return: pandas DataFrame, the selected annotations.
    '''
    
    annotations = annotations_.copy()
    annotations['is_selected'] = True
    
    annotations['total_bee'] = annotations['bee'] + annotations['noisybee']
    annotations['is_bee'] = annotations['total_bee'] > bee_fraction_threshold - eps
    annotations['is_selected'] &= annotations['is_bee'] | (annotations['total_bee'] < eps)
    
    if exclude_noisybee: 
        annotations['is_selected'] &= annotations['noisybee'] < eps
    return annotations


def balance_samples_by_annotation(annotations_, random_seed=13):
    '''
    Balance the samples based on the annotations.
    annotations_: pandas DataFrame, the annotations per sample.
    random_seed: int, the random seed.
    return: pandas DataFrame, the balanced annotations.
    '''
    
    rng = np.random.default_rng(random_seed)
    annotations = annotations_.copy()
    
    true_idx = np.where(annotations['is_bee'] & annotations['is_selected'])[0]
    false_idx = np.where(~annotations['is_bee'] & annotations['is_selected'])[0]
    min_len = min(len(true_idx), len(false_idx))
    true_idx = rng.choice(true_idx, min_len, replace=False)
    false_idx = rng.choice(false_idx, min_len, replace=False)
    idx = np.concatenate([true_idx, false_idx])
    idx = np.sort(idx)
    annotations['is_selected'] = False
    annotations.iloc[idx, annotations.columns.get_loc('is_selected')] = True
    
    return annotations


def create_dataset_annotations(files, sr, frame_length, hop_length, balancing=True, exclude_noisybee=True, bee_fraction_threshold=1, random_seed=13):
    '''
    Create a dataset from multiple recording files.
    files: list of Path objects, the paths to the recording files.
    sr: int, the sampling rate.
    frame_length: int, the length of the frames.
    hop_length: int, the hop length.
    exclude_noisybee: bool, whether to exclude the frames with noisy bee.
    bee_fraction_threshold: float, the threshold for the bee fraction.
    random_seed: int, the random seed.
    return: pandas DataFrame, the annotations.
    '''
    
    frame_length_seconds = frame_length / sr
    hop_length_seconds = hop_length / sr
    
    annotations = pd.concat(
        list(map(lambda x: load_annotations_by_file(x, frame_length_seconds, hop_length_seconds), files)),
        ignore_index=True,
    )
    
    annotations = select_samples_by_annotation(annotations, exclude_noisybee=exclude_noisybee, bee_fraction_threshold=bee_fraction_threshold)
    if balancing:
        annotations = balance_samples_by_annotation(annotations, random_seed=random_seed)
    return annotations


def create_dataset_from_annotations(annotations=None, files=None, sr=None, frame_length=None, hop_length=None, balancing=True, exclude_noisybee=True, bee_fraction_threshold=1, random_seed=13, preprocessing_fn=None):
    '''
    Create a dataset from multiple recording files.
    annotations: pandas DataFrame, the annotations.
    files: list of Path objects, the paths to the recording files.
    sr: int, the sampling rate.
    frame_length: int, the length of the frames.
    hop_length: int, the hop length.
    random_seed: int, the random seed.
    return: tuple of numpy arrays, the input and output tensors and the annotations.
    '''
    
    annotations_grouped_by_rec = annotations.groupby('recording')
    recordings = list(annotations.recording.unique())
    
    def load_selected_samples(rec_path):
        rec_annotations = annotations_grouped_by_rec.get_group(rec_path)
        if type(rec_path) is str:
            rec_path = rec_path.replace('\\', '/')
        x, _ = librosa.load(rec_path, sr=sr)
        x = segmentation(x, sr, frame_length, hop_length)
        selected_indices = np.sort(rec_annotations[rec_annotations.is_selected].sample_idx.values)
        x = x[selected_indices, ...]
        if preprocessing_fn is not None:
            x = preprocessing_fn(x)
        x = normalize(x)
        return x
    
    if annotations is None:
        if (files is None) or (sr is None) or (frame_length is None) or (hop_length is None):
            raise ValueError('Either annotations or files, sr, frame_length and hop_length must be provided.')
        annotations = create_dataset_annotations(files, sr, frame_length, hop_length, balancing=balancing, exclude_noisybee=exclude_noisybee, bee_fraction_threshold=bee_fraction_threshold, random_seed=random_seed)
    elif (frame_length is None) or (hop_length is None):
        raise ValueError('frame_length and hop_length must be provided.')
    
    x = np.concatenate(list(map(load_selected_samples, recordings)), axis=0)
    annotations.sort_values(['recording', 'sample_idx'], key=lambda x: x.apply(lambda x: recordings.index(x)) if x.name == 'recording' else x, inplace=True)
    y = annotations[annotations.is_selected].is_bee.values
    return x, y, annotations


def compute_specs(rec_path, frame_length, hop_length, sr):
    '''
    Compute the spectrograms for a recording.
    rec_path: Path object, the path to the recording file.
    frame_length: int, the length of the frames.
    hop_length: int, the hop length.    
    sr: int, the sampling rate.
    return: tuple of numpy arrays, the recording, the spectrogram, the mel spectrogram, the phase and the times in samples.
    '''
    rec, _ = librosa.load(rec_path, sr=sr)
    S, phase = librosa.magphase(librosa.stft(rec, n_fft=frame_length, hop_length=hop_length, center=False))
    S_mel = librosa.feature.melspectrogram(S=S, sr=sr)
    times_samples = librosa.frames_to_samples(np.arange(S.shape[1]), hop_length=hop_length, n_fft=frame_length)
    times_samples -= times_samples[0]
    return rec, S, S_mel, phase, times_samples


def extract_features(rec, S, S_mel, frame_length, hop_length, sr, valid_frames_indices=None, return_feat_indices=False):
    '''
    Extract the features from the recording.
    rec: numpy array, the recording.
    S: numpy array, the spectrogram.
    S_mel: numpy array, the mel spectrogram.
    frame_length: int, the length of the frames.
    hop_length: int, the hop length.
    sr: int, the sampling rate.
    valid_frames_indices: numpy array, the indices of the valid frames.
    return_feat_indices: bool, whether to return the feature indices.
    return: tuple of numpy arrays, the features and the frames.
    '''
    
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S_mel), sr=sr)
    flatness = librosa.feature.spectral_flatness(S=S, power=2)
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(rec, frame_length=frame_length, hop_length=hop_length, center=False)
    if valid_frames_indices is not None:
        zero_crossing_rate = zero_crossing_rate[:, valid_frames_indices]
    
    cqt = librosa.cqt(np.pad(rec, (frame_length//2, frame_length//2)), sr=sr, hop_length=hop_length)
    frames_per_pad = (cqt.shape[1] - S.shape[1]) // 2
    cqt = np.abs(cqt)[:, frames_per_pad:-frames_per_pad]
    if valid_frames_indices is not None:
        cqt = cqt[:, valid_frames_indices]
    
    features = np.concatenate([cqt, mfccs, flatness, spectral_centroid, spectral_rolloff, spectral_contrast, zero_crossing_rate, chroma, S], axis=0).T
    feat_indices = {
        'cqt': np.arange(cqt.shape[0]),
        'mfccs': np.arange(cqt.shape[0], cqt.shape[0]+mfccs.shape[0]),
        'flatness': np.arange(cqt.shape[0]+mfccs.shape[0], cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]),
        'spectral_centroid': np.arange(cqt.shape[0]+mfccs.shape[0]+flatness.shape[0], cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]+spectral_centroid.shape[0]),
        'spectral_rolloff': np.arange(cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]+spectral_centroid.shape[0], cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]+spectral_centroid.shape[0]+spectral_rolloff.shape[0]),
        'spectral_contrast': np.arange(cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]+spectral_centroid.shape[0]+spectral_rolloff.shape[0], cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]+spectral_centroid.shape[0]+spectral_rolloff.shape[0]+spectral_contrast.shape[0]),
        'zero_crossing_rate': np.arange(cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]+spectral_centroid.shape[0]+spectral_rolloff.shape[0]+spectral_contrast.shape[0], cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]+spectral_centroid.shape[0]+spectral_rolloff.shape[0]+spectral_contrast.shape[0]+zero_crossing_rate.shape[0]),
        'chroma': np.arange(cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]+spectral_centroid.shape[0]+spectral_rolloff.shape[0]+spectral_contrast.shape[0]+zero_crossing_rate.shape[0], cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]+spectral_centroid.shape[0]+spectral_rolloff.shape[0]+spectral_contrast.shape[0]+zero_crossing_rate.shape[0]+chroma.shape[0]),
        'STFT': np.arange(cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]+spectral_centroid.shape[0]+spectral_rolloff.shape[0]+spectral_contrast.shape[0]+zero_crossing_rate.shape[0]+chroma.shape[0], cqt.shape[0]+mfccs.shape[0]+flatness.shape[0]+spectral_centroid.shape[0]+spectral_rolloff.shape[0]+spectral_contrast.shape[0]+zero_crossing_rate.shape[0]+chroma.shape[0]+S.shape[0])
    }
    if return_feat_indices:
        return features, feat_indices
    return features


def create_features_dataset_from_annotations(annotations=None, sr=None, frame_length=None, hop_length=None):
    '''
    Create a dataset from multiple recording files.
    annotations: pandas DataFrame, the annotations.
    sr: int, the sampling rate.
    frame_length: int, the length of the frames.
    hop_length: int, the hop length.
    return: tuple of numpy arrays, the input and output tensors and the annotations.
    '''
    
    frame_length += frame_length % 2 # make sure frame_length is even
    annotations_grouped_by_rec = annotations.groupby('recording')
    recordings = list(annotations.recording.unique())
    
    dataset_data, dataset_labels = [], []
    def load_selected_samples(rec_path):
        rec_annotations = annotations_grouped_by_rec.get_group(rec_path)
        if type(rec_path) is str:
            rec_path = rec_path.replace('\\', '/')
        rec, S, S_mel, _, _ = compute_specs(rec_path, frame_length, hop_length, sr)
        features, feat_indices = extract_features(rec, S, S_mel, frame_length, hop_length, sr, return_feat_indices=True)
        selected_indices = np.sort(rec_annotations[rec_annotations.is_selected].sample_idx.values)
        normalized_features = normalize_by_feature(features[selected_indices, :], feat_indices)
        normalized_features = np.round(normalized_features, 5)
        normalized_features = normalized_features.astype(np.float32)
        dataset_data.append(normalized_features)
        dataset_labels.append(rec_annotations[rec_annotations.is_selected].is_bee.values)
        return feat_indices
    feat_indices = list(map(load_selected_samples, recordings))[0]
    dataset_data = np.concatenate(dataset_data, axis=0)
    dataset_labels = np.concatenate(dataset_labels, axis=0)
    
    return dataset_data, dataset_labels, feat_indices


def normalize_by_feature(data, feat_indices):
    '''
    Normalize the data by feature.
    data: numpy array, the data.
    feat_indices: dict, the indices of the features.
    return: numpy array, the normalized data.
    '''
    
    for feat_name, feat_indices in feat_indices.items():
        data[:, feat_indices] /= np.abs(data[:, feat_indices]).max(axis=0, keepdims=True)
    return data


def generate_noise(noise_type, num_samples, fs=44100):
    """
    Generate different types of noise.

    Parameters:
    noise_type (str): Type of noise to generate ('white', 'pink', 'brown', 'blue', 'violet', 'grey').
    num_samples (int): Number of samples of the noise.
    fs (int): Sampling frequency, default is 44100 Hz.

    Returns:
    np.ndarray: Array containing the generated noise.
    """

    # Generate white noise
    noise = np.random.normal(0, 1, num_samples)

    if noise_type == 'white':
        return noise

    elif noise_type in ['pink', 'brown', 'blue', 'violet']:
        # Creating a filter for the specific noise color
        if noise_type == 'pink':
            b, a = butter(1, [1], btype='low', fs=fs)
        elif noise_type == 'brown':
            b, a = butter(1, [1], btype='low', fs=fs, output='ba')
        elif noise_type == 'blue':
            b, a = butter(1, [1], btype='high', fs=fs)
        elif noise_type == 'violet':
            b, a = butter(2, [1], btype='high', fs=fs)

        # Applying the filter to the white noise
        colored_noise = filtfilt(b, a, noise)
        return colored_noise

    elif noise_type == 'grey':
        # Grey noise generation (approximate)
        freqs = np.fft.rfftfreq(num_samples, d=1/fs)
        factors = np.sqrt(1 / (freqs + 1e-10)) # Avoid division by zero
        grey_noise = np.fft.irfft(np.fft.rfft(noise) * factors)
        return grey_noise

    else:
        raise ValueError("Invalid noise type. Choose from 'white', 'pink', 'brown', 'blue', 'violet', 'grey'.")


def add_noise_to_signal(audio_signal, noise_type, desired_snr_db, fs=44100):
    """
    Add noise to a given audio_signal to achieve a specified SNR.

    Parameters:
    audio_signal (np.ndarray): Input audio_signal to which noise is to be added.
    noise_type (str): Type of noise to add ('white', 'pink', 'brown', 'blue', 'violet', 'grey').
    desired_snr_db (float): Desired SNR in decibels.
    fs (int): Sampling frequency, default is 44100 Hz.

    Returns:
    np.ndarray: audio_signal with added noise.
    """
    # Generate noise
    noise = generate_noise(noise_type, len(audio_signal), fs)

    # Calculate the power of the audio_signal and the noise
    audio_signal_power = np.mean(audio_signal ** 2)
    noise_power = np.mean(noise ** 2)

    # Calculate the necessary scaling factor for the noise to achieve the desired SNR
    desired_snr_linear = 10 ** (desired_snr_db / 10)
    scaling_factor = np.sqrt(audio_signal_power / (desired_snr_linear * noise_power))

    # Scale and add the noise to the audio_signal
    noisy_audio_signal = audio_signal + scaling_factor * noise
    return noisy_audio_signal


if __name__ == "__main__":
    pass
