import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import DrumCRNN
from src.dataset import DrumDataset
from src.utils import N_MELS
from tqdm import tqdm



# --- 하이퍼파라미터 설정 ---
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_CLASSES = 8
NUM_WORKERS = 0 # Windows에서는 0으로 설정 (Linux/Mac에서는 CPU 코어 수에 맞게 조절 가능)
POSITIVE_WEIGHTS = [50]*NUM_CLASSES # 클래스 불균형 문제 완화

RESUME_MODEL = 'models/drum_crnn_epoch_1.pth' # None이면 처음부터 학습 시작

# --- 데이터 경로 설정 ---
CHECKPOINT_DIR = 'checkpoints' # 에포크별 모델 저장 경로
TRAIN_DATA_DIR = 'data/train'



# def create_loss_weights(labels, device, left=1, right=3, pos_loss=50.0):

#     surround_loss = pos_loss/(left + right)

#     # label과 동일한 크기의 가중치 텐서를 1로 초기화
#     loss_weights = torch.ones_like(labels, device=device)
    
#     # 라벨 값이 1인 위치(t)의 인덱스를 찾습니다.
#     positive_indices = (labels == 1).nonzero(as_tuple=True) # (배치인덱스리스트, 시간인덱스리스트, 클래스인덱스리스트)
#     if positive_indices[0].numel() == 0:
#         return loss_weights

#     positive_mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
#     for batch, time, class_idx in zip(*positive_indices):
#         start_time = max(0, time - left)
#         end_time = min(labels.shape[1], time + right + 1)
        
#         positive_mask[batch, start_time:end_time, class_idx] = True
        
#     # 마스크에 True로 표시된 구간의 loss가중치를 변경
#     loss_weights[positive_mask] = surround_loss
#     loss_weights[positive_indices] = pos_loss

#     return loss_weights



# 1. 동적 패딩을 위한 collate_fn 함수를 main 함수 밖에 정의
def collate_fn_pad(batch):
    """
    데이터셋에서 반환된 가변 길이의 샘플들을 배치로 묶을 때,
    배치 내 최대 길이에 맞춰 동적으로 패딩을 적용합니다.
    """
    # 데이터 로딩 중 에러가 발생한 샘플(None)을 필터링합니다.
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return torch.empty(0), torch.empty(0) # 배치가 비어있으면, 스펙트로그램과 GT피아노롤을 빈 텐서로 리턴

    # 언패킹 & zip 하여 스펙트로그램과 GT피아노롤 분리
    # [(스펙트로그램1, GT피아노롤1), (스펙트로그램2, GT피아노롤2), ...]  ->  (스펙트로그램1, 스펙트로그램2, ...), (GT피아노롤1, GT피아노롤2, ...)
    spectrograms, labels = zip(*batch)


    # 현재 배치에서 가장 긴 스펙트로그램의 길이를 찾습니다.
    max_spec_len = max(s.shape[2] for s in spectrograms) # 스펙트로그램.shape = (채널, 주파수, 시간)
    
    # 스펙트로그램 패딩 처리
    padded_spectrograms = [] # 패딩이 완료된 스펙트로그램이 저장되는 배열
    for s in spectrograms:
        pad_size = max_spec_len - s.shape[2] # 각 스펙트로그램의 부족한 길이 계산
        padded_s = torch.nn.functional.pad(s, (0, pad_size)) # 각 스펙트로그램의 마지막차원(시간차원)의 오른쪽에 부족한 만큼 패딩
        padded_spectrograms.append(padded_s)


    # 현재 배치에서 가장 긴 피아노롤의 길이를 찾습니다.
    max_label_len = max(l.shape[0] for l in labels) # 피아노롤.shape = (시간, 클래스=8)

    # 피아노롤 패딩 처리
    padded_labels = [] # 패딩이 완료된 피아노롤이 저장되는 배열
    for l in labels:
        pad_size = max_label_len - l.shape[0] # 각 피아노롤의 부족한 길이 계산
        padded_l = torch.nn.functional.pad(l, (0, 0, 0, pad_size)) # 각 피아노롤의 마지막에서 두번째차원(시간차원)의 오른쪽에 부족한 만큼 패딩
        padded_labels.append(padded_l)

    # 패딩이 완료된 텐서들을 쌓아서 최종 배치 텐서를 만듭니다.
    return torch.stack(padded_spectrograms), torch.stack(padded_labels)


def main():

    # 파서: 터미널에서 인자를 받는 객체
    parser = argparse.ArgumentParser(description='Train a Drum CRNN model.') # 파서 생성
    parser.add_argument('--resume', type=str, default=RESUME_MODEL) # 파서에 학습을 이어서하는 옵션 추가
    args = parser.parse_args() # 파서로 받은 인자들을 args에 저장

    # --- 에포크별 모델 저장을 위한 디렉토리 생성 ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 장치 설정 (GPU or CPU) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터셋 준비 
    train_dataset = DrumDataset(data_dir=TRAIN_DATA_DIR, num_classes=NUM_CLASSES)

    # 데이터로더 준비 (DataLoader에 새로 정의한 collate_fn_pad 함수를 지정)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn_pad, num_workers=NUM_WORKERS) # 멀티프로세싱을 위해 num_workers 조절 가능

    # --- 모델(CRNN), 손실 함수(BCEWithLogitsLoss), 옵티마이저(Adam) 정의 ---
    model = DrumCRNN(num_classes=NUM_CLASSES, freq_bins=N_MELS).to(device)
    pos_weight_tensor = torch.tensor(POSITIVE_WEIGHTS, device=device) # 클래스 불균형 문제 완화
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) # , reduction='none'
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 1
    # --- 체크포인트에서 학습 이어서하기 옵션 적용 ---
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume) # 
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            print(f"Error: Checkpoint file not found at '{args.resume}'. Starting from scratch.")


    # --- 학습 루프 ---
    print("Starting training...")
    for epoch in range(start_epoch, NUM_EPOCHS+1):
        torch.cuda.empty_cache()
        model.train() # 모델을 학습 모드로 전환
        running_loss = 0.0 # 한 epoch의 누적 loss
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        
        for i, (spectrograms, labels) in enumerate(progress_bar):

            # 오류가 발생한 배치는 학습 생략
            if spectrograms.numel() == 0:
                continue

            spectrograms, labels = spectrograms.to(device), labels.to(device) # GPU로 입력데이터, GT데이터 보냄

            # 모델에 스펙트로그램 입력
            outputs = model(spectrograms) 

            # 텐서 크기 맞추기: outputs와 labels의 시간 차원 길이를 작은 쪽에 맞춤
            target_len = min(outputs.shape[1], labels.shape[1])
            outputs = outputs[:, :target_len, :]
            labels = labels[:, :target_len, :]
            # loss_weights = create_loss_weights(labels, device)

            # 모델의 출력으로 loss 계산
            loss = criterion(outputs, labels)
            # loss = (loss * loss_weights).mean() # 가중치 적용 후 평균 loss 계산

            optimizer.zero_grad() # 이전에 계산한 가중치 별 gradient 값을 0으로 초기화
            loss.backward() # 가중치 별 gradient 계산
            optimizer.step() # 모든 가중치 업데이트

            running_loss += loss.item()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}") # 가중치 업데이트마다 loss 출력

        # epoch이 끝난 후, 평균 loss 계산 및 출력
        avg_loss = running_loss / len(train_loader) 
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] completed. Average Loss: {avg_loss:.4f}")
        
        # epoch이 끝난 후, 모델 저장
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"drum_crnn_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    main()