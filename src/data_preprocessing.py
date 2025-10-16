import pyloudnorm as pyln
import librosa
import soundfile as sf
import mido
import numpy as np
import os
from tqdm import tqdm
import concurrent.futures # 병렬 처리를 위해 추가



DATA_DIR = r'D:\data\train'
PREPROCESSED_DIR = r'D:\data\train_split'
TARGET_LUFS = -23.0
NUM_WORKERS = 1 


# 이 함수는 변경되지 않았습니다.
def split_midi(midi_path, output_dir, chunk_beat=16):
    mid = mido.MidiFile(midi_path)
    name, ext = os.path.splitext(os.path.basename(midi_path))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tempo = 500000 
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break

    chunk_infos = []

    current_chunk = mido.MidiFile()
    current_chunk.tracks.append(mid.tracks[0])
    current_track = mido.MidiTrack()
    current_chunk.tracks.append(current_track)
    
    absolute_time = 0
    chunk_start_time = 0
    chunk_index = 1
    
    for msg in mid.tracks[1]:
        absolute_time += msg.time
        
        if msg.type == 'note_on' and msg.velocity > 0:
            if absolute_time - chunk_start_time > chunk_beat*mid.ticks_per_beat:
                chunk_filename = name + f"_{chunk_index}" + '.midi'
                chunk_path = os.path.join(output_dir, chunk_filename)
                current_chunk.save(chunk_path)

                chunk_start_sec = mido.tick2second(chunk_start_time, mid.ticks_per_beat, tempo)
                absolute_sec = mido.tick2second(absolute_time, mid.ticks_per_beat, tempo)
                chunk_infos.append((chunk_start_sec, absolute_sec, chunk_filename))
                
                chunk_index += 1
                chunk_start_time = absolute_time
                
                current_chunk = mido.MidiFile()
                current_chunk.tracks.append(mid.tracks[0])
                current_track = mido.MidiTrack()
                current_chunk.tracks.append(current_track)
        
        current_track.append(msg)
    
    if len(current_track) > 0:
        chunk_filename = name + f"_{chunk_index}" + '.midi'
        chunk_path = os.path.join(output_dir, chunk_filename)
        current_chunk.save(chunk_path)
        
        chunk_start_sec = mido.tick2second(chunk_start_time, mid.ticks_per_beat, tempo)
        absolute_sec = mido.tick2second(absolute_time, mid.ticks_per_beat, tempo)
        chunk_infos.append((chunk_start_sec, absolute_sec, chunk_filename))

    return chunk_infos



# --- 병렬 처리를 위한 워커 함수 ---
def process_file(midi_file, data_dir, preprocessed_dir, target_lufs):
    """하나의 MIDI 파일과 그에 해당하는 오디오 파일을 처리하는 함수"""
    try:
        midi_path = os.path.join(data_dir, midi_file)
        audio_path = os.path.join(data_dir, os.path.splitext(midi_file)[0] + '.wav')

        # 오디오 파일이 없는 경우 건너뛰기
        if not os.path.exists(audio_path):
            return f"Skipped: Audio for {midi_file} not found."

        # 1. 미디 파일을 4마디(16비트) 단위로 자르고, 자른 미디 조각들의 정보 리스트를 받음
        chunk_infos = split_midi(midi_path, preprocessed_dir, chunk_beat=16)

        # 2-1. 오디오 라우드니스 정규화
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        y_normalized = pyln.normalize.loudness(y, loudness, target_lufs)

        # 2-2. 미디 조각들에 맞춰 오디오도 자르고 저장
        for chunk_start_sec, chunk_end_sec, chunk_filename in chunk_infos:
            start_sample = int(chunk_start_sec * sr)
            end_sample = int(chunk_end_sec * sr)
            
            # 오디오 길이를 초과하지 않도록 방지
            if end_sample > len(y_normalized):
                end_sample = len(y_normalized)

            y_chunk = y_normalized[start_sample:end_sample]
            
            output_audio_path = os.path.join(preprocessed_dir, chunk_filename.replace('.midi', '.wav'))
            sf.write(output_audio_path, y_chunk, sr)
        
        # # 3. 처리된 원본 파일 삭제 (필요 시 주석 해제)
        # os.remove(midi_path)
        # os.remove(audio_path)
        
        return f"Successfully processed {midi_file}"

    except Exception as e:
        return f"Failed to process {midi_file}: {e}"


def main():

    if not os.path.exists(PREPROCESSED_DIR):
        os.makedirs(PREPROCESSED_DIR)

    midi_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.midi')])
    
    num_workers = NUM_WORKERS
    print(f"--- Starting preprocessing with {num_workers} worker(s) ---")

    # ProcessPoolExecutor를 사용하여 병렬 처리
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 각 파일에 대한 작업을 executor에 제출(submit)
        futures = [executor.submit(process_file, midi_file, DATA_DIR, PREPROCESSED_DIR, TARGET_LUFS) 
                   for midi_file in midi_files]
        
        # 작업 완료를 기다리며 tqdm으로 진행 상황 표시
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(midi_files), desc="Preprocessing files"):
            # 결과나 에러를 확인하고 싶다면 아래 주석을 해제하세요.
            print(future.result())
            pass
            
    print("--- Preprocessing finished ---")


# 멀티프로세싱을 위한 필수 코드
if __name__ == "__main__":
    main()