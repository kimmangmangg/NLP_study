## 09-01. 워드 임베딩(Word Embedding)<br>


- 워드 임베딩은 단어를 밀집 표현(Dense Representation)으로 변환하는 방법임.
- 단어 의미적 유사성을 반영할 수 있는 벡터 표현 방법임.

### 1) 희소 표현 (Sparse Representation)

- 원-핫 인코딩은 단어 집합 크기만큼 차원을 가지며 해당 단어 인덱스만 1, 나머지는 0으로 표현함
- 차원이 단어 집합 크기와 동일 → 고차원
- 대부분 값이 0인 희소 벡터
- **단어 간 의미적 유사성을 반영하지 못함**

```python
강아지 = [0, 0, 0, 0, 1, 0, 0, ..., 0]   (차원: 10,000)
```

**[문제점]**
- 공간 낭비 (고차원 희소 행렬)
- 의미 반영 불가 (단어 간 관계 고려 없음)
- DTM(문서-단어 행렬)도 희소 표현의 대표적 사례


### 2) 밀집 표현 (Dense Representation)

- 희소 표현과 달리 차원을 작게 설정하고, 값은 실수로 구성됨
- 모든 단어가 동일한 크기의 벡터로 매핑됨 (보통 50~300차원)
- **의미적 정보를 반영 가능**

```python
강아지 = [0.2, 1.8, 1.1, -2.1, 1.1, 2.8, ...]   (차원: 128)
```

### 3) 워드 임베딩 (Word Embedding)

- 단어를 밀집 벡터로 표현하는 기법
- 학습을 통해 얻어진 벡터를 임베딩 벡터(embedding vector)라 함
- 대표적인 방법론: LSA, Word2Vec, FastText, GloVe
- 케라스 `Embedding()`

  - 초기엔 무작위 값으로 임베딩 벡터 생성
  - 신경망 학습 과정에서 가중치처럼 업데이트되어 단어 벡터 학습

### 📌 원-핫 벡터 vs 임베딩 벡터

| 구분    | 원-핫 벡터         | 임베딩 벡터          |
| ----- | -------------- | --------------- |
| 차원    | 고차원 (단어 집합 크기) | 저차원 (보통 50~300) |
| 종류    | 희소 벡터          | 밀집 벡터           |
| 표현 방법 | 수동 (인덱스 기반)    | 학습 기반           |
| 값의 타입 | 0과 1           | 실수              |


### 📌 핵심 정리

- 원-핫 인코딩은 희소 벡터로 의미 반영 불가
- 밀집 표현은 저차원 실수 벡터로 의미 반영 가능
- 워드 임베딩은 단어를 밀집 벡터로 표현하는 표준 기법
- 임베딩 벡터는 학습을 통해 단어 간 관계를 반영

<br><Br>

## 09-02. 워드투벡터(Word2Vec)<br>

* Word2Vec은 단어를 밀집 벡터로 학습하는 대표적인 워드 임베딩 방법임.
* 신경망 기반으로 단어의 의미적 유사성을 반영한 벡터를 학습함.

### 1) Word2Vec의 개념

* 구글(2013)에서 제안한 단어 임베딩 알고리즘
* **목표**: 단어의 의미적/문법적 관계를 벡터 공간에 반영
* CBOW, Skip-gram 두 가지 모델 구조 제공

### 2) CBOW (Continuous Bag of Words)

* **중심 단어 예측 모델**
* 주변 단어(Context word)들을 입력으로 받아 중심 단어(Target word)를 예측함
* 학습 시, 주변 단어들이 주어진 상황에서 중심 단어가 나올 확률을 최대화
* 입력: 주변 단어들
* 출력: 중심 단어
* 작은 데이터셋에서 더 빠르고 효율적

### 3) Skip-gram

* **주변 단어 예측 모델**
* 중심 단어를 입력으로 받아 주변 단어들을 예측함
* 학습 시, 중심 단어가 주어졌을 때 주변 단어가 등장할 확률을 최대화
* 입력: 중심 단어
* 출력: 주변 단어들
* 큰 데이터셋에서 더 좋은 성능

### 📌 CBOW vs Skip-gram 비교

| 구분 | CBOW                  | Skip-gram                   |
| -- | --------------------- | --------------------------- |
| 입력 | 주변 단어들(Context)       | 중심 단어(Target)               |
| 출력 | 중심 단어(Target)         | 주변 단어들(Context)             |
| 장점 | 학습 속도 빠름, 작은 데이터셋에 적합 | 큰 데이터셋에서 성능 우수, 희귀 단어 학습 잘됨 |

### 4) Word2Vec의 의의

* 단어를 저차원 실수 벡터로 표현
* 단어 간 연산이 가능 (king - man + woman ≈ queen)
* 의미적 유사성을 잘 반영하여, 다양한 NLP 태스크에서 기본 표현 기법으로 활용

### 📌 핵심 정리

* Word2Vec은 신경망 기반의 단어 임베딩 기법
* CBOW: 주변 단어로 중심 단어 예측
* Skip-gram: 중심 단어로 주변 단어 예측
* 단어 간 의미적 관계를 벡터 공간에서 학습 가능

<br><Br>

## 09-03. 영어/한국어 Word2Vec 실습<br>

* Word2Vec을 gensim 라이브러리를 사용하여 실습하는 예제임.
* 영어와 한국어 데이터셋을 활용하여 CBOW/Skip-gram 모델을 학습하고 단어 유사도를 확인함.

### 1) 영어 Word2Vec 실습

* **데이터**: NLTK의 `text8` 데이터 사용 (약 1700만 단어)

* **전처리**: 토큰화된 문장 리스트 형태 준비

* **모델 학습**

  ```python
  from gensim.models import Word2Vec
  from nltk.corpus import text8

  dataset = text8.sents()
  model = Word2Vec(sentences=dataset, size=100, window=5, sg=0, min_count=5, workers=4)
  ```

  * `size=100`: 임베딩 차원
  * `window=5`: 학습 시 고려할 주변 단어 크기
  * `sg=0`: CBOW (sg=1이면 Skip-gram)
  * `min_count=5`: 최소 5회 이상 등장 단어만 학습

* **유사도 확인**

  ```python
  print(model.wv.most_similar("man"))
  print(model.wv.similarity("man", "woman"))
  ```

  * 의미적으로 유사한 단어들을 벡터 공간에서 찾을 수 있음

### 2) 한국어 Word2Vec 실습

* **데이터**: 한국어 문장을 토큰화 후 학습
* **전처리**: `Okt` 형태소 분석기로 문장을 토큰화

  ```python
  from konlpy.tag import Okt

  okt = Okt()
  tokenized = [okt.morphs(sentence) for sentence in korean_sentences]
  ```
* **모델 학습**

  ```python
  model_ko = Word2Vec(sentences=tokenized, size=100, window=5, sg=1, min_count=5, workers=4)
  ```

  * `sg=1`: Skip-gram 모델
* **유사도 확인**

  ```python
  print(model_ko.wv.most_similar("강아지"))
  ```

  * 한국어에서도 의미 유사 단어들을 확인 가능

### 📌 핵심 정리

* Word2Vec 실습은 `gensim` 패키지를 주로 활용함
* CBOW(sg=0), Skip-gram(sg=1) 선택 가능
* 영어(NLTK text8), 한국어(Okt 토큰화) 모두 학습 가능
* 학습된 모델은 단어 간 유사도를 계산하여 의미 관계를 파악할 수 있음

<br><Br>
