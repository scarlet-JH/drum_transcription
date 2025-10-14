import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import DrumDataset
from src.utils import midi_to_pianoroll, SAMPLE_RATE, HOP_LENGTH
from tqdm import tqdm
import librosa
import os

# --- 설정값 ---
DATA_DIR = 'D:\data\data_split'
NUM_CLASSES = 8
NUM_SAMPLES_TO_CHECK = 1000


def collate_fn_filter_none(batch):
    # batch는 (spectrogram, piano_roll) 튜플의 리스트입니다.
    # 튜플의 첫 번째 항목(spectrogram)이 None이 아닌 샘플만 남깁니다.
    batch = list(filter(lambda x: x[0] is not None, batch))
    
    # 만약 한 배치 전체가 필터링되어 비어버린 경우
    if not batch:
        # 루프에서 처리할 수 있도록 (None, None)을 반환합니다.
        return (None, None)
        
    # 필터링된 정상적인 배치에 대해 기본 collate 함수를 호출하여 배치를 생성합니다.
    return torch.utils.data.dataloader.default_collate(batch)

# --- 데이터셋 준비 ---
temp_dataset = DrumDataset(data_dir=DATA_DIR, num_classes=NUM_CLASSES)
temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_filter_none)

# 활성/비활성 타임스텝 수를 저장할 변수
class_ones = np.zeros(NUM_CLASSES)
num_active_steps = 0
num_silent_steps = 0

print(f"Checking first {NUM_SAMPLES_TO_CHECK} samples for active/silent time step ratio...")

for i, data in enumerate(tqdm(temp_loader)):
    if i >= NUM_SAMPLES_TO_CHECK:
        break
    if data == (None, None):
        continue

    spectrogram, piano_roll = data
    piano_roll = piano_roll.squeeze(0)

    try:
        # 3. ★★★ 핵심 로직 ★★★
        # 각 시간 스텝(행)에 1이 하나라도 있는지 확인 -> (True, False, True...)
        is_active = torch.any(piano_roll == 1, dim=1)
        class_ones += torch.sum(piano_roll == 1, dim=0).numpy()
        
        # 활성 스텝(True)과 비활성 스텝(False)의 개수를 셈
        num_active_steps += torch.sum(is_active)
        num_silent_steps += len(is_active) - torch.sum(is_active)

    except Exception as e:
        print(f"Skip due to error: {e}")
        continue

# 4. 최종 가중치 계산
epsilon = 1e-8
time_step_weight = num_silent_steps / (num_active_steps + epsilon)

print("\n--- Time-Step Based Weight Calculation ---")
print(f"Total Active Time Steps: {num_active_steps}")
print(f"Total Silent Time Steps: {num_silent_steps}")
print(f"Calculated POSITIVE_WEIGHT: {time_step_weight:.2f}")
print(class_ones)

# 이 계산된 time_step_weight를 train.py의 POSITIVE_WEIGHT로 사용
# POSITIVE_WEIGHT = time_step_weight