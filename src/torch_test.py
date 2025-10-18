import torch
import torchvision
import torchaudio

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")





import numpy as np

# --- 파라미터 설정 ---
n_mels = 256
fmin = 0.0
fmax = 22050

# 1. 분석할 주파수 범위를 멜 스케일로 변환
m_min = 2595 * np.log10(1 + fmin / 700)
m_max = 2595 * np.log10(1 + fmax / 700)

# 2. 멜 스케일 상에서 n_mels개 만큼 선형적으로 분할
mel_points = np.linspace(m_min, m_max, n_mels)

# 3. 분할된 멜 값들을 다시 주파수(Hz)로 변환
hz_points = 700 * (10**(mel_points / 2595) - 1)

for v in hz_points:
    print(v)



import mido

MIDI_FILE_PATH = 'data/train/1_funk-groove1_138_beat_4-4_1.midi'

mid = mido.MidiFile(MIDI_FILE_PATH)

new_mid2 = mido.MidiFile()
new_mid2.tracks.append(mid.tracks[0]) # 메타정보 트랙 복사
new_track2 = mido.MidiTrack()
new_mid2.tracks.append(new_track2)

tempo = 0
for msg in mid:
    if msg.type == 'set_tempo':
        tempo = msg.tempo
        # print(f"tempo: {tempo} microsec per beat")
        break

for i, msg in enumerate(mid.tracks[1]):
        
    if i>100: break

    # print(f"{msg.type}, {msg.time}")

    new_track2.append(msg)

print()
for i, msg in enumerate(new_mid2.tracks[1]):
    if i>10: break
    # print(f"{msg.type}, {msg.time}")



for epoch in range(1,30+1):
    model = torch.load(f'models_50/drum_crnn_epoch_{epoch}.pth')
    print(f"Epoch {epoch}: {model['loss']:.4f}")