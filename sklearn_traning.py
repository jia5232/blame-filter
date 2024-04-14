import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

# "씨123발" 등을 학습 시키기 위한 학습 데이터 증강 함수
def augment_text(text, num_augment=2):
    special_chars = ['1', '2', '3', '!', '@', '#']
    augmented_texts = [text]
    for _ in range(num_augment):
        random_char = random.choice(special_chars)
        insertion_point = random.randint(0, len(text))
        augmented_text = text[:insertion_point] + random_char + text[insertion_point:]
        augmented_texts.append(augmented_text)
    return augmented_texts

# 텍스트 정규화 함수 -> 공백제거
def normalize_text(text):
    text = re.sub(r"\s+", "", text)
    return text

# 데이터를 로드하고 전처리하는 함수
def load_and_preprocess_data(filepath):
    # 데이터를 로드
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().split('|', maxsplit=1)
            # 텍스트 정규화 적용
            normalized_text = normalize_text(text)

            if label == '1':  # 혐오 표현인 경우 데이터 증강!
                augmented_texts = augment_text(normalized_text)
                for aug_text in augmented_texts:
                    data.append((aug_text, int(label)))
            else:
                data.append((normalized_text, int(label)))

    df = pd.DataFrame(data, columns=['text', 'label'])
    return df

# 데이터를 로드
df = load_and_preprocess_data('dataset.txt')

# 문자열 기반의 N-gram을 사용한 TF-IDF 벡터화를 수행한다
tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3), max_features=1000)
X = tfidf_vectorizer.fit_transform(df['text']).toarray()
y = df['label']

# 데이터를 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델의 하이퍼파라미터 그리드 설정
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  
    # 로지스틱 회귀의 정규화 강도를 조절하는 파라미터
    # 값이 낮으면 강한 정규화, 높으면 약한 정규화!!

    'max_iter': [100, 500, 1000]  
    # 최대 반복 횟수 -> 데이터를 학습시키는 과정에서 특정 횟수 이상으로 반복하면 과적합 또는 반복 자체가 별 의미가 없어지므로 제일 적합한 반복 횟수를 찾기 위함이다!!!
}

# 로지스틱 회귀 모델 만들기!!
# 그런데 모든 파라미터들에 대해 테스트하기가 어렵기 떄문에 GridSearchCV를 사용해서 객체 생성
# 그럼 테스트 파라미터들에 의한 모델들중에 정확도가 제일 높은 모델을 찾아줌
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')

# 그리드 서치 수행 -> 여기서 정확도 제일 높은 모델을 찾음.
grid_search.fit(X_train, y_train)

# 최적의 파라미터
print("Best Parameters: {}".format(grid_search.best_params_))
# 정확도
print("Best Cross-Validation Score: {:.2f}".format(grid_search.best_score_))

# 테스트 데이터셋에 대해 최적의 모델로 평가!!!
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')


# 모델 저장
from joblib import dump
model_bundle = {
    "vectorizer": tfidf_vectorizer,
    "model": grid_search
}
dump(model_bundle, 'model.joblib')
