import os
import torch
from torch.utils.data import Dataset
from src.utils import audio_to_spectrogram, midi_to_pianoroll, SAMPLE_RATE 
import librosa
import numpy as np

MIN_TIME_STEPS = 100
MAX_TIME_STEPS = 4000

NOISE_FACTOR = 0.005

# __len__, __getitem__ 로 크기가 정의된 이터레이터로 만듬
class DrumDataset(Dataset):
    def __init__(self, data_dir, num_classes, is_train=True):

        self.data_dir = data_dir
        self.num_classes = num_classes
        
        # 해당 디렉토리에서 .wav 파일 목록만 가져옴
        self.audio_files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.wav', '.mp3'))])

        self.is_train = is_train # ★★★ 훈련 데이터셋인지 여부 저장 ★★★


    def __len__(self):
        return len(self.audio_files)


    # 인덱스(idx)에 해당하는 샘플(스펙트로그램, 피아노롤)을 반환합니다.
    def __getitem__(self, idx):
        audio_name = self.audio_files[idx]
        base_name = os.path.splitext(audio_name)[0]
        
        audio_path = os.path.join(self.data_dir, audio_name) # 어떤 os에서도 올바른 경로로 만들어줌
        midi_path = os.path.join(self.data_dir, base_name + '.midi')
        
        try:
            # 1. 오디오 파일을 불러옵니다. (+ 학습데이터면 노이즈 추가)
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            if self.is_train:
                noise = np.random.randn(len(y))
                noised_y = y + NOISE_FACTOR * noise
                noised_y = np.clip(noised_y, -1.0, 1.0)
                y = noised_y

            # 2. 불러온 오디오 데이터(y)로 스펙트로그램을 생성합니다.
            spectrogram = audio_to_spectrogram(y, sr)
            
            # 너무 길거나 짧은 데이터는 건너뛰기
            if spectrogram.shape[2] < MIN_TIME_STEPS or spectrogram.shape[2] > MAX_TIME_STEPS:
                return None, None

            # 3. 오디오 데이터(y)로 길이를 계산하고, 미디파일로 피아노롤(ground-truth) 생성.
            audio_duration = len(y) / sr
            piano_roll = midi_to_pianoroll(midi_path, self.num_classes, audio_duration)

            return spectrogram, piano_roll

        except Exception as e:
            # 파일 로딩 또는 처리 중 에러 발생 시 None을 반환하여 DataLoader가 무시하도록 함
            print(f"Error processing {audio_path}: {e}")
            return None, None