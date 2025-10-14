import os
import logging # 프로그램 실행 상태 표기
from audio_separator.separator import Separator

AUDIO_FILENAME = "silly-silly-love"
AUDIO_PATH = f"songs/{AUDIO_FILENAME}.mp3"

OUTPUT_DIR = "drum_audio" 
OUTPUT_FORMAT = "wav"

EXTRACTOR_MODEL_DIR = "drum_extractor_models"

# Demucs 모델은 드럼 분리에 효과적입니다.
# 'htdemucs_ft.yaml', 'htdemucs.yaml', 'htdemucs_6s.yaml' 등을 사용할 수 있습니다.
MODEL_FILENAME = "htdemucs_ft.yaml"


def separate_drums_from_cli(audio_path, output_dir, output_format, model_dir, model_filename):
    
    if not os.path.exists(audio_path):
        print(f"오류: 입력 파일 '{audio_path}'를 찾을 수 없습니다.")
        return

    # 출력 폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # Separator 객체 생성
    separator = Separator(
        log_level=logging.INFO, # 처리 과정을 콘솔에 출력
        model_file_dir=model_dir, 
        output_dir=output_dir,
        output_format=output_format,
        output_single_stem="drums", # 드럼만 추출
        demucs_params={ # Demucs 모델에 맞는 파라미터 설정
            "shifts": 2, # shift 횟수 (shift: 시간차를 두고 여러번 처리하고 평균내는 기법)
            "overlap": 0.25, # 오버랩 비율
        }
    )

    # 사용할 모델 불러오기
    try:
        separator.load_model(model_filename=model_filename)
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return

    # 오디오 분리 실행
    try:
        output_files = separator.separate(audio_path)
        print("\n분리가 완료되었습니다!")
        print("저장된 파일 목록:")
        for file_path in output_files:
            print(f"- {file_path}")
    except Exception as e:
        print(f"오디오 분리 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    separate_drums_from_cli(
        audio_path=AUDIO_PATH,
        output_dir=OUTPUT_DIR,
        output_format=OUTPUT_FORMAT,
        model_dir=EXTRACTOR_MODEL_DIR,
        model_filename=MODEL_FILENAME
    )