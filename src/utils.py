import librosa
import numpy as np
import mido
import torch

# --- 오디오 및 MIDI 처리를 위한 설정값 ---
SAMPLE_RATE = 44100
N_FFT = 1024 # FFT를 수행할 샘플의 길이
HOP_LENGTH = 64
N_MELS = 256 # freq-bin
DOWNSAMPLE_FACTOR = 4 # pooling에 의한 시간 차원 downsample 비율


# 오디오 데이터 배열(y)을 Mel-Spectrogram으로 변환합니다.
# Mel-Spectrogram: Mel-scale로 spectrogram을 만든 것.
# Mel-scale: 실제 주파수를 사람이 인지하는 주파수로 변형하는 척도
def audio_to_spectrogram(y, sr):
    try:
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max) # 작은값과 큰값의 편차를 줄임
        
        # 1. torch에서 데이터를 텐서로써 다룰 수 있게
        # 2. CNN 입력 형식에 맞게 (freq, time) -> (channel, freq, time)
        spectrogram_tensor = torch.from_numpy(log_mel_spectrogram).float().unsqueeze(0)
        return spectrogram_tensor
    
    except Exception as e:
        print(f"Error converting audio array to spectrogram: {e}")
        return None




def midi_to_pianoroll(midi_path, num_classes, duration_in_sec):
    """MIDI 파일을 Piano Roll 형태로 변환합니다. (시간 계산 버그 최종 수정 버전)"""
    
    mid = mido.MidiFile(midi_path)
    
    # 피아노 롤 생성
    num_time_steps = int(duration_in_sec * SAMPLE_RATE / HOP_LENGTH / DOWNSAMPLE_FACTOR)
    piano_roll = np.zeros((num_time_steps, num_classes))
    
    # 드럼 맵 정의
    # 크래쉬(0), 하이햇(1), 킥(2), 스네어(3), 스몰탐(4), 라지탐(5), 플로어탐(6), 라이드(7)
    drum_map = {49: 0 ,  52: 0 ,  55: 0 ,  57: 0 , 
                42: 1 ,  44: 1 ,  46: 1 ,  
                35: 2 ,  36: 2 , 
                37: 3 ,  38: 3 ,  39: 3 ,  40: 3 , 
                48: 4 ,  50: 4 , 
                45: 5 ,  47: 5 , 
                41: 6 ,  43: 6 , 
                51: 7 ,  53: 7 ,  59: 7 ,
                }

    absolute_time_sec = 0.0
    for msg in mid:
        # 델타 타임(초)을 누적
        absolute_time_sec += msg.time

        # print(f"msg.type: {msg.type}")
        # print(f"msg.time: {msg.time}")
        # print(f"absolute_time_sec: {absolute_time_sec}")

        if msg.type == 'note_on' and msg.velocity > 0:
            if msg.note in drum_map:
                class_idx = drum_map[msg.note]
                
                # 누적된 절대 시간(초)을 피아노 롤의 시간 스텝 인덱스로 변환
                time_step = int(absolute_time_sec * SAMPLE_RATE / HOP_LENGTH / DOWNSAMPLE_FACTOR)
                
                if time_step < num_time_steps:
                    piano_roll[time_step, class_idx] = 1
                    
    return torch.from_numpy(piano_roll).float()




def output_to_midi(predictions, output_path, bpm=120, threshold=0.5):
    """모델의 출력(확률)을 MIDI 파일로 변환합니다."""
    # predictions shape: (time_steps, num_classes)
    
    # 빈 미디파일, 빈 트랙 생성 후 미디파일에 트랙 삽입
    mid = mido.MidiFile()
    meta_track = mido.MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # 드럼 노트 번호 매핑
    # 크래쉬(0), 하이햇(1), 킥(2), 스네어(3), 스몰탐(4), 라지탐(5), 플로어탐(6), 라이드(7)
    drum_map = {0:49, 1:42, 2:35, 3:38, 4:48, 5:45, 6:41, 7:51 }

    # 1 tick 당 시간(sec) = tick 당 4분음표 수 * 4분음표 당 시간(min) * min 당 sec
    ticks_per_sec = mid.ticks_per_beat * bpm / 60
    
    last_event_time_ticks = 0
    
    # 활성화된 노트를 추적하기 위한 배열
    active_notes = [False] * predictions.shape[1]

    for time_step, frame in enumerate(predictions):
        current_time_ticks = int((time_step * DOWNSAMPLE_FACTOR * HOP_LENGTH / SAMPLE_RATE) * ticks_per_sec)
        
        for drum_idx, prob in enumerate(frame):
            note = drum_map.get(drum_idx)
            if note is None: continue

            if prob > threshold and not active_notes[drum_idx]:
                # 노트 시작
                delta_ticks = current_time_ticks - last_event_time_ticks
                track.append(mido.Message('note_on', note=note, velocity=100, time=delta_ticks))
                active_notes[drum_idx] = True
                last_event_time_ticks = current_time_ticks

            elif prob <= threshold and active_notes[drum_idx]:
                # 노트 종료
                delta_ticks = current_time_ticks - last_event_time_ticks
                track.append(mido.Message('note_off', note=note, velocity=0, time=delta_ticks))
                active_notes[drum_idx] = False
                last_event_time_ticks = current_time_ticks
    
    mid.save(output_path)
    print(f"MIDI file saved to {output_path}")