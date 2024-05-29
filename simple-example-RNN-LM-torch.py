import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 샘플 데이터
texts = [
    "안녕하세요",
    "저는 학생입니다",
    "오늘 날씨가 좋네요",
    "여기는 서울입니다"
]

# 텍스트 토크나이저
class Tokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.total_words = 0

    def fit_on_texts(self, texts):
        for text in texts:
            for word in text:
                if word not in self.word_index:
                    self.total_words += 1
                    self.word_index[word] = self.total_words
                    self.index_word[self.total_words] = word

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_index[word] for word in text]
            sequences.append(seq)
        return sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# 입력 시퀀스 생성
input_sequences = []
for line in texts:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# 패딩 처리 및 데이터셋 생성
class TextDataset(Dataset):
    def __init__(self, sequences, total_words):
        self.sequences = [torch.tensor(seq) for seq in sequences]
        self.total_words = total_words

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq[:-1], seq[-1]

dataset = TextDataset(input_sequences, tokenizer.total_words)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: pad_sequence([item[0] for item in x], batch_first=True), )

# 모델 정의
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        h, _ = self.lstm(x)
        h = h[:, -1, :]
        out = self.fc(h)
        return out

# 하이퍼파라미터
vocab_size = tokenizer.total_words + 1
embed_size = 10
hidden_size = 100
num_epochs = 100
learning_rate = 0.001

model = RNNModel(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 훈련
for epoch in range(num_epochs):
    for seqs, targets in dataloader:
        outputs = model(seqs)
        targets = torch.tensor([target for seq, target in dataset])
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 전사 결과 수정 함수
def correct_transcription(asr_output):
    token_list = tokenizer.texts_to_sequences([asr_output])[0]
    token_list = pad_sequence([torch.tensor(token_list)], batch_first=True)
    predicted = model(token_list)
    predicted_word_index = torch.argmax(predicted, dim=1)
    predicted_word = tokenizer.index_word[predicted_word_index.item()]
    return predicted_word

# 예시 사용
asr_output = "저는 학생"
corrected_output = correct_transcription(asr_output)
print("수정된 전사 결과:", corrected_output)
