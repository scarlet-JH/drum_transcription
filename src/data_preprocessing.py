import pyloudnorm as pyln
import librosa
import soundfile as sf
import mido
import numpy as np
import os
from tqdm import tqdm


# 미디를 청크(4마디=16비트) 단위로 잘라서 미디 형태로 저장하고, 
# 자른 조각들의 (start_sec, end_sec, filename) 리스트를 반환
def split_midi(midi_path, output_dir, chunk_beat=16):
    mid = mido.MidiFile(midi_path)
    name, ext = os.path.splitext(os.path.basename(midi_path))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tempo = 500000 
    for msg in mid.tracks[0]: # 0번트랙은 메타정보
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break

    chunk_infos = []

    current_chunk = mido.MidiFile()
    current_chunk.tracks.append(mid.tracks[0]) # 메타정보 트랙 복사
    current_track = mido.MidiTrack()
    current_chunk.tracks.append(current_track)
    
    absolute_time = 0
    chunk_start_time = 0
    chunk_index = 1
    
    for msg in mid.tracks[1]: # 1번트랙은 실제음표
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
                current_chunk.tracks.append(mid.tracks[0]) # 메타정보 트랙 복사
                current_track = mido.MidiTrack()
                current_chunk.tracks.append(current_track)
        
        current_track.append(msg)
    
    # 마지막 청크 저장
    if len(current_track) > 0:
        chunk_filename = name + f"_{chunk_index}" + '.midi'
        chunk_path = os.path.join(output_dir, chunk_filename)
        current_chunk.save(chunk_path)
        
        chunk_start_sec = mido.tick2second(chunk_start_time, mid.ticks_per_beat, tempo)
        absolute_sec = mido.tick2second(absolute_time, mid.ticks_per_beat, tempo)
        chunk_infos.append((chunk_start_sec, absolute_sec, chunk_filename))

    return chunk_infos



def main():

    DATA_DIR = r'D:\data\train'
    PREPROCESSED_DIR = r'D:\data\train_split'
    TARGET_LUFS = -23.0


    if not os.path.exists(PREPROCESSED_DIR):
        os.makedirs(PREPROCESSED_DIR)

    midi_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.midi')])
    for midi_file in tqdm(midi_files, desc="Preprocessing MIDI & Audio"):
        midi_path = os.path.join(DATA_DIR, midi_file)
        audio_path = os.path.join(DATA_DIR, os.path.splitext(midi_file)[0] + '.wav')
        # print(f"Processing '{midi_file}' and corresponding audio...")
        
        # 1. 미디 파일을 4마디(16비트) 단위로 자르고, 자른 미디 조각들의 정보 리스트를 받음
        chunk_infos = split_midi(midi_path, PREPROCESSED_DIR, chunk_beat=16)

        # 2-1. 오디오 라우드니스 정규화
        y, sr = librosa.load(audio_path, sr=44100)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        y_normalized = pyln.normalize.loudness(y, loudness, TARGET_LUFS)

        # 2-2. 미디 조각들에 맞춰 오디오도 자르고 저장
        for chunk_start_sec, chunk_end_sec, chunk_filename in chunk_infos:
            # print(f"  -> Processing audio segment: {chunk_start_sec:.2f}s to {chunk_end_sec:.2f}s")
            
            y_chunk = y_normalized[int(chunk_start_sec*sr):int(chunk_end_sec*sr)]
            
            output_audio_path = os.path.join(PREPROCESSED_DIR, chunk_filename.replace('.midi', '.wav'))
            sf.write(output_audio_path, y_chunk, sr)
        
        # # 3. 원본 미디, 오디오 파일 삭제 (공간 절약)
        # os.remove(midi_path)
        # os.remove(audio_path)


# 디버깅용: 자른 미디와 오디오의 길이 비교
'''
    # 4개로 나뉜 미디의 시간
    splited_midifile1 = mido.MidiFile(os.path.join(PREPROCESSED_DIR, '1_funk-groove1_138_beat_4-4_1_1.midi'))
    splited_midifile2 = mido.MidiFile(os.path.join(PREPROCESSED_DIR, '1_funk-groove1_138_beat_4-4_1_2.midi'))
    splited_midifile3 = mido.MidiFile(os.path.join(PREPROCESSED_DIR, '1_funk-groove1_138_beat_4-4_1_3.midi'))
    splited_midifile4 = mido.MidiFile(os.path.join(PREPROCESSED_DIR, '1_funk-groove1_138_beat_4-4_1_4.midi'))

    print(f"Duration of splited_midifile1: {splited_midifile1.length:.4f} sec")
    print(f"Duration of splited_midifile2: {splited_midifile2.length:.4f} sec")
    print(f"Duration of splited_midifile3: {splited_midifile3.length:.4f} sec")
    print(f"Duration of splited_midifile4: {splited_midifile4.length:.4f} sec")
    # 4개 총합 시간
    print(f"Total duration of splited_midifiles: {(splited_midifile1.length + splited_midifile2.length + splited_midifile3.length + splited_midifile4.length):.4f} sec")

    # 원본 미디 시간
    original_midifile = mido.MidiFile(os.path.join(DATA_DIR, '1_funk-groove1_138_beat_4-4_1.midi'))
    print(f"Duration of original_midifile: {original_midifile.length:.4f} sec")

    # 4개로 나뉜 오디오의 시간
    splited_audiofile1 = os.path.join(PREPROCESSED_DIR, '1_funk-groove1_138_beat_4-4_1_1.wav')
    splited_audiofile2 = os.path.join(PREPROCESSED_DIR, '1_funk-groove1_138_beat_4-4_1_2.wav')
    splited_audiofile3 = os.path.join(PREPROCESSED_DIR, '1_funk-groove1_138_beat_4-4_1_3.wav')
    splited_audiofile4 = os.path.join(PREPROCESSED_DIR, '1_funk-groove1_138_beat_4-4_1_4.wav')

    y1, sr1 = librosa.load(splited_audiofile1, sr=44100)
    y2, sr2 = librosa.load(splited_audiofile2, sr=44100)
    y3, sr3 = librosa.load(splited_audiofile3, sr=44100)
    y4, sr4 = librosa.load(splited_audiofile4, sr=44100)

    print(f"Duration of splited_audiofile1: {len(y1)/sr1:.4f} sec")
    print(f"Duration of splited_audiofile2: {len(y2)/sr2:.4f} sec")
    print(f"Duration of splited_audiofile3: {len(y3)/sr3:.4f} sec")
    print(f"Duration of splited_audiofile4: {len(y4)/sr4:.4f} sec")
    # 4개 총합 시간
    print(f"Total duration of splited_audiofiles: {(len(y1)/sr1 + len(y2)/sr2 + len(y3)/sr3 + len(y4)/sr4):.4f} sec")

    # 원본 미디 시간
    original_audiofile = os.path.join(DATA_DIR, '1_funk-groove1_138_beat_4-4_1.wav')
    y_orig, sr_orig = librosa.load(original_audiofile, sr=44100)
    print(f"Duration of original_audiofile: {len(y_orig)/sr_orig:.4f} sec")
'''

       


if __name__ == "__main__":
    main()
