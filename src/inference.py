import torch
import librosa
import pyloudnorm as pyln
from src.model import DrumCRNN
from src.utils import audio_to_spectrogram, output_to_midi


NUM_CLASSES = 8 
BPM = 87
TARGET_LUFS = -23.0 

filename = 'silly-silly-love_(Drums)_htdemucs_ft'
AUDIO_PATH = f'drum_audio/{filename}.wav'
OUTPUT_PATH = f'output/{filename}_pred.mid'

MODEL_PATH = 'models_50/drum_crnn_epoch_10.pth'  # 학습된 모델 파일 경로



def transcribe(y, sr, model_path, output_path):
    """오디오 파일을 입력받아 MIDI로 변환합니다."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 오디오 전처리 ---
    spectrogram = audio_to_spectrogram(y, sr)
    if spectrogram is None:
        return
    
    # 배치 차원 추가 및 디바이스로 이동
    spectrogram = spectrogram.unsqueeze(0).to(device)

    # --- 모델 불러오기 ---
    # 모델 파라미터는 저장된 모델과 동일해야 함
    model = DrumCRNN(num_classes=NUM_CLASSES, rnn_hidden_size=128, freq_bins=128).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # 추론 모드로 설정

    print("Model loaded successfully.")

    
    # --- 추론 수행 ---
    with torch.no_grad():
        predictions = model(spectrogram)
    
    # 배치 차원 제거
    predictions = predictions.squeeze(0).cpu().numpy()

    # --- MIDI 파일로 변환 ---
    output_to_midi(predictions, output_path, bpm=BPM, threshold=0.5)

def main():

    # 1. 오디오 로드
    y, sr = librosa.load(AUDIO_PATH, sr=44100)
    
    # 2. 라우드니스 정규화 수행
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    normalized_y = pyln.normalize.loudness(y, loudness, TARGET_LUFS)
    
    transcribe(y=normalized_y, 
               sr=sr,
               model_path=MODEL_PATH, 
               output_path=OUTPUT_PATH)
    

if __name__ == '__main__':
    main()