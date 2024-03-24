import numpy as np
import pandas as pd
import librosa
from .audio_preprocessing import segmentation, normalize

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


def create_dataset_from_annotations(annotations=None, files=None, sr=None, frame_length=None, hop_length=None, balancing=True, exclude_noisybee=True, bee_fraction_threshold=1, random_seed=13):
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
        x, _ = librosa.load(rec_path, sr=sr)
        x = segmentation(x, sr, frame_length, hop_length)
        selected_indices = np.sort(rec_annotations[rec_annotations.is_selected].sample_idx.values)
        return x[selected_indices, ...]
    
    if annotations is None:
        if (files is None) or (sr is None) or (frame_length is None) or (hop_length is None):
            raise ValueError('Either annotations or files, sr, frame_length and hop_length must be provided.')
        annotations = create_dataset_annotations(files, sr, frame_length, hop_length, balancing=balancing, exclude_noisybee=exclude_noisybee, bee_fraction_threshold=bee_fraction_threshold, random_seed=random_seed)
    elif (frame_length is None) or (hop_length is None):
        raise ValueError('frame_length and hop_length must be provided.')
    
    x = np.concatenate(list(map(load_selected_samples, recordings)), axis=0)
    annotations.sort_values(['recording', 'sample_idx'], key=lambda x: x.apply(lambda x: recordings.index(x)) if x.name == 'recording' else x, inplace=True)
    x = normalize(x)
    y = annotations[annotations.is_selected].is_bee.values
    return x, y, annotations

if __name__ == "__main__":
    pass
