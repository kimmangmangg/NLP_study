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

## 09-04. 네거티브 샘플링을 이용한 Word2Vec 구현 (SGNS)<br>

* Skip-gram with Negative Sampling(SGNS)은 Word2Vec의 Skip-gram을 개선한 방식임.
* 단어 벡터 학습 시 불필요한 연산을 줄이고 효율적으로 임베딩을 학습할 수 있음.

### 1) Skip-gram 한계

* Skip-gram은 중심 단어로 주변 단어를 예측함
* 소프트맥스 계산 시 **전체 단어 집합 크기만큼 확률 분포 계산** 필요 → 매우 비효율적
* 단어 집합이 수만~수십만 개일 경우 연산량이 급격히 증가

### 2) 네거티브 샘플링 (Negative Sampling)

* 전체 단어에 대한 확률 계산 대신, 일부 단어만 선택하여 학습
* **중심 단어–실제 주변 단어 쌍(positive pair)** 과
  **중심 단어–무작위 샘플링 단어 쌍(negative pair)** 을 함께 학습
* 손실 함수는 긍정/부정 샘플에 대해 로지스틱 회귀(시그모이드) 기반으로 계산

### 3) 학습 구조

* 입력: 중심 단어
* 출력: 주변 단어 (실제 주변 단어는 label=1, 무작위 샘플은 label=0)
* 네거티브 샘플 개수(k)는 일반적으로 5~20개 사용
* 전체 단어 집합에 대한 확률 계산을 피하고 효율적으로 단어 임베딩 학습 가능

### 4) 구현 예시 (gensim)

```python
from gensim.models import Word2Vec
from nltk.corpus import text8

dataset = text8.sents()

# Skip-gram + Negative Sampling
model = Word2Vec(sentences=dataset, size=100, window=5, sg=1, negative=5, min_count=5, workers=4)

print(model.wv.most_similar("man"))
```

* `sg=1`: Skip-gram
* `negative=5`: 네거티브 샘플링 적용 (5개 단어 샘플링)

### 📌 핵심 정리

* Skip-gram은 단어 집합이 크면 연산량이 과도함
* 네거티브 샘플링은 전체 단어 대신 일부 단어만 고려하여 효율적으로 학습
* SGNS는 Word2Vec에서 널리 사용되는 최적화 기법으로 단어 간 의미 관계를 잘 반영함

<br><br>

## 09-05. 글로브(GloVe)<br>

* GloVe(Global Vectors for Word Representation)는 카운트 기반과 예측 기반 방법론을 결합한 워드 임베딩 기법임.
* 전역 통계 정보와 로컬 문맥 정보를 함께 반영하여 단어 벡터를 학습함.

### 1) Word2Vec과의 차이

* **Word2Vec**: 로컬 문맥 정보 활용 (주변 단어 예측)
* **GloVe**: 전역 통계 정보(동시 등장 행렬) + 로컬 문맥 정보 결합

### 2) 동시 등장 행렬(Co-occurrence Matrix)

* 말뭉치 전체에서 단어 i와 j가 동시에 등장하는 빈도를 행렬로 표현
* 특정 단어 w의 의미는 다른 단어들과 함께 등장하는 확률 분포로 설명 가능

### 3) GloVe의 아이디어

* 두 단어의 내적은 두 단어의 동시 등장 확률을 잘 설명해야 함
* 단어 벡터 학습 시, **확률비(PMI, Pointwise Mutual Information)** 를 최소화하는 방향으로 최적화
* 손실 함수는 가중치 함수와 로그 확률 비를 기반으로 설계됨

### 4) 구현 예시 (GloVe 학습)

* 일반적으로 Stanford에서 제공하는 GloVe 사전 학습 모델 사용 가능
* 예: Wikipedia, Gigaword 등의 대규모 코퍼스 기반으로 학습된 벡터 활용

```bash
$ git clone https://github.com/stanfordnlp/GloVe.git
$ cd GloVe
$ make
$ ./demo.sh
```

* Python에서는 `gensim.downloader`를 통해 사전 학습된 GloVe 벡터 불러와 사용 가능

```python
import gensim.downloader as api
glove_vectors = api.load("glove-wiki-gigaword-100")

print(glove_vectors.most_similar("king"))
```

### 📌 핵심 정리

* GloVe는 **카운트 기반 + 예측 기반**을 결합한 임베딩 방법
* 단어 벡터 간 내적이 동시 등장 확률을 잘 설명하도록 설계됨
* Word2Vec보다 전역적 통계 정보를 더 잘 반영함
* 사전 학습된 GloVe 벡터는 다양한 NLP 태스크에서 활용 가능

<br><br>

## 09-06. 패스트텍스트(FastText)<br>

* FastText는 Facebook AI Research(2016)에서 제안한 단어 임베딩 기법임.
* Word2Vec의 한계를 보완하여 **단어 내부의 형태(서브워드)** 를 활용함.

### 1) Word2Vec의 한계

* Word2Vec은 단어 단위 학습이므로, 훈련 시 등장하지 않은 단어(OOV)를 처리할 수 없음
* 한국어, 독일어처럼 형태 변화가 많은 언어에서 단어 간 유사성을 충분히 반영하지 못함

### 2) FastText의 아이디어

* 단어를 글자 단위의 n-gram으로 분해하여 학습
* 각 단어 벡터는 해당 단어의 n-gram 벡터들의 합으로 표현됨
* 따라서 **형태적으로 유사한 단어는 벡터도 유사하게 학습됨**

예: `apple` → `ap`, `app`, `ppl`, `ple`, `le` 등의 subword로 분해

### 3) 장점

* 희귀 단어나 훈련에 등장하지 않은 단어(OOV)도 subword 기반으로 벡터 생성 가능
* 형태 변화가 많은 언어(한국어, 독일어 등)에 강점
* Word2Vec보다 일반화 성능 우수

### 4) 구현 예시 (gensim)

```python
from gensim.models import FastText
from nltk.corpus import text8

dataset = text8.sents()

model = FastText(sentences=dataset, size=100, window=5, min_count=5, workers=4, sg=1)

print(model.wv.most_similar("apple"))
```

* `size=100`: 임베딩 차원
* `sg=1`: Skip-gram 방식
* `min_count=5`: 최소 5회 이상 등장 단어만 학습

### 📌 핵심 정리

* FastText는 Word2Vec을 확장한 기법으로, 단어를 **subword 단위**로 분해하여 학습
* OOV 단어도 처리 가능 → 일반화 성능 우수
* 형태 변화가 많은 언어에서 효과적
* 현재 많은 NLP 작업에서 Word2Vec보다 더 널리 활용됨

<br><br>

## 09-08. 사전 훈련된 워드 임베딩 (Pre-trained Word Embedding)<br>

* 대규모 코퍼스로 미리 학습된 임베딩 벡터를 가져와 사용하는 방법임.
* 작은 데이터셋으로 직접 임베딩 학습 시 한계가 있으므로, 사전 학습된 벡터를 활용하면 성능 향상을 기대할 수 있음.

### 1) 필요성

* 대규모 코퍼스를 직접 학습하기 어렵고 비용이 큼
* 사전 훈련된 임베딩을 사용하면 더 나은 일반화 성능 확보 가능

### 2) 대표적인 사전 훈련 임베딩

* **Word2Vec** (구글 뉴스 1억 단어 기반)
* **GloVe** (Wikipedia + Gigaword 등 대규모 데이터 기반)
* **FastText** (Facebook 제공, subword 기반)

### 3) 활용 방법

* `gensim.downloader`를 통해 손쉽게 불러와 사용 가능

```python
import gensim.downloader as api

# GloVe 임베딩 불러오기 (100차원)
glove_vectors = api.load("glove-wiki-gigaword-100")

print(glove_vectors.most_similar("king"))
```

* 불러온 임베딩을 Keras Embedding Layer와 연동하거나, 모델의 가중치로 초기화 가능

### 4) 장점

* 작은 데이터셋에서도 강력한 임베딩 활용 가능
* 학습 시간 절약 및 성능 향상
* 다양한 언어, 다양한 크기 차원의 사전 훈련 벡터 제공

### 📌 핵심 정리

* 사전 훈련된 워드 임베딩은 대규모 코퍼스로 학습된 임베딩을 가져와 사용하는 방법
* Word2Vec, GloVe, FastText 등이 대표적
* 직접 학습하기 어려운 대규모 데이터의 효과를 간접적으로 활용 가능
* 성능 향상과 학습 시간 절감에 유리함

<br><br>
## 09-09. 엘모(ELMo, Embeddings from Language Model)<br>

* ELMo는 **단어의 문맥(Context)에 따라 동적으로 임베딩을 생성**하는 방법임.
* 기존 Word2Vec, GloVe, FastText 등은 문맥과 무관하게 단어당 하나의 정적 벡터를 제공하지만, ELMo는 문맥에 따라 다른 벡터를 생성함.

### 1) 기존 임베딩의 한계

* "bank" → 금융기관 / 강둑, 문맥에 따라 의미 달라짐
* Word2Vec·GloVe는 단어당 하나의 벡터만 제공 → 다의어 처리 불가

### 2) ELMo의 아이디어

* **양방향 LSTM 기반 언어모델**을 활용하여 문맥 정보를 반영
* 문맥에 따라 단어 벡터가 달라짐 (동적 임베딩)
* 입력 문장에서 단어의 앞뒤 단어 모두 고려

### 3) 구조

* Character-level CNN + BiLSTM
* 문장 전체를 입력으로 받아 각 단어의 문맥적 표현 벡터를 출력
* 사전 훈련된 언어모델에서 얻은 벡터를 다운스트림 태스크(NER, 감성분석 등)에 적용

### 4) 장점

* **다의어 처리 가능**: 문맥에 맞는 의미로 벡터 생성
* **다양한 태스크 성능 향상**: NER, 문장 분류, 질의응답 등
* Word2Vec/GloVe보다 훨씬 풍부한 의미 정보 제공

### 📌 핵심 정리

* ELMo는 **문맥 의존적 동적 임베딩** 기법
* BiLSTM 기반 언어모델로 단어 의미를 문맥에 따라 다르게 표현
* 다의어 처리 및 다양한 NLP 태스크 성능 향상에 효과적

<br><br>
## 09-10. 임베딩 벡터의 시각화 (Embedding Visualization)<br>

* 학습된 임베딩 벡터를 시각화하면 단어 간 의미적 관계를 직관적으로 이해할 수 있음.
* 주로 차원 축소 기법(PCA, t-SNE 등)을 이용하여 2D/3D 공간에 표현함.

### 1) 차원 축소 필요성

* 임베딩 벡터는 보통 100~300차원 → 사람이 직접 해석하기 어려움
* 차원 축소 기법을 활용해 저차원(2D, 3D) 공간으로 투영하여 시각화

### 2) 주요 기법

* **PCA (Principal Component Analysis)**

  * 분산을 최대화하는 방향으로 데이터를 축소
  * 전반적인 분포 확인에 적합

* **t-SNE (t-distributed Stochastic Neighbor Embedding)**

  * 고차원 데이터의 지역적 구조(유사 단어끼리의 군집) 반영에 효과적
  * Word2Vec, GloVe 결과 해석에 자주 사용

### 3) gensim Word2Vec + 시각화 예시

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = model.wv[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:,0], result[:,1])
words = list(model.wv.vocab)
for i, word in enumerate(words[:100]):
    plt.annotate(word, xy=(result[i,0], result[i,1]))
plt.show()
```

* 상위 100개 단어를 2차원 공간에 시각화

### 4) 해석

* 의미적으로 유사한 단어들이 가까이 위치
* 특정 주제(예: 나라, 동물, 음식)별로 자연스럽게 군집 형성

### 📌 핵심 정리

* 임베딩 시각화는 단어 간 의미 관계를 직관적으로 파악하는 도구
* PCA는 전반적 구조, t-SNE는 지역적 구조 파악에 효과적
* 시각화를 통해 학습된 임베딩의 품질을 점검 가능

<br><br>
## 09-11. 문서 벡터를 이용한 추천 시스템<br>

* 문서 임베딩을 활용해 **문서 간 유사도를 계산**하고, 이를 기반으로 추천 시스템을 구현할 수 있음.
* 코사인 유사도를 주로 활용하여 입력 문서와 가장 유사한 문서를 추천함.

### 1) 문서 벡터(Document Embedding)

* 문서 전체를 하나의 벡터로 표현하는 방법
* 방법 예시

  * Word2Vec 임베딩 벡터들의 평균
  * Doc2Vec 등 전용 문서 임베딩 기법

### 2) 문서 유사도 측정

* **코사인 유사도**: 두 벡터의 방향적 유사성을 계산
* 값의 범위: -1 ~ 1

  * 1에 가까울수록 두 문서가 유사

### 3) 추천 시스템 구현 절차

1. 문서를 벡터로 변환
2. 입력 문서 벡터와 다른 문서 벡터 간 코사인 유사도 계산
3. 유사도가 높은 순으로 문서 추천

### 4) 간단한 예시 코드

```python
from sklearn.metrics.pairwise import cosine_similarity

# 임의의 문서 벡터 (예: Word2Vec 평균)
doc_vectors = [
    [0.1, 0.3, 0.5],
    [0.2, 0.4, 0.6],
    [0.9, 0.8, 0.7]
]

# 첫 번째 문서를 기준으로 유사도 계산
similarities = cosine_similarity([doc_vectors[0]], doc_vectors)
print(similarities)
```

* 결과값이 큰 문서일수록 첫 번째 문서와 의미적으로 유사

### 📌 핵심 정리

* 문서 벡터는 문서 단위 의미 표현을 가능하게 함
* 코사인 유사도를 활용해 문서 간 유사성 계산
* 문서 추천 시스템, 검색 엔진, 뉴스 기사 추천 등 다양한 분야에 활용 가능

<br><br>
## 09-12. 문서 임베딩 : 워드 임베딩의 평균 (Average Word Embedding)<br>

* 문서 임베딩은 문서 전체를 벡터로 표현하는 방법임.
* 간단한 방법으로, **문서를 구성하는 단어 임베딩들의 평균**을 구하여 문서 벡터로 사용함.

### 1) 아이디어

* 각 단어를 Word2Vec, GloVe, FastText 등으로 임베딩
* 문서 벡터 = 단어 임베딩들의 평균값
* 단순하지만 문서 간 유사도 계산 및 분류 작업에서 효과적

### 2) 구현 절차

1. 단어별 임베딩 벡터 불러오기 (예: Word2Vec)
2. 문장을 토큰화하여 각 단어 벡터 추출
3. 벡터들의 평균을 계산하여 문서 벡터 생성

### 3) 예시 코드

```python
import numpy as np
from gensim.models import Word2Vec

# 학습된 Word2Vec 모델 불러오기
model = Word2Vec.load("word2vec.model")

def document_vector(doc):
    # 단어 벡터들의 평균으로 문서 벡터 계산
    doc = [word for word in doc if word in model.wv]
    return np.mean(model.wv[doc], axis=0)

sentence = ["강아지", "고양이", "밥"]
print(document_vector(sentence))
```

* 결과: 100차원(또는 설정한 크기)의 문서 벡터 반환

### 4) 장점과 한계

* **장점**

  * 구현 간단, 연산 효율적
  * 소규모 데이터셋에서 빠르게 적용 가능
* **한계**

  * 단순 평균 → 단어 순서·문맥 반영 불가
  * 의미적으로 중요한 단어와 불필요한 단어를 구분하지 못함

### 📌 핵심 정리

* Average Word Embedding은 단어 벡터들의 평균으로 문서 임베딩을 만드는 방법
* 단순하지만 문서 유사도 계산 및 분류에 기본적으로 활용 가능
* 문맥 반영이 안 되는 한계가 있어 Doc2Vec, Transformer 기반 임베딩으로 확장됨

<br><br>
## 09-13. Doc2Vec으로 공시 사업보고서 유사도 계산하기<br>

* Doc2Vec은 문서 단위 임베딩을 학습하는 알고리즘임.
* Word2Vec의 확장으로, 문서를 고정된 크기의 벡터로 표현하여 문서 간 유사도 계산이 가능함.

### 1) Doc2Vec 개념

* Word2Vec은 단어 단위 벡터 학습 → 문서 전체 표현 불가
* Doc2Vec은 문서 태그를 추가하여 문서 벡터도 함께 학습
* 문서 임베딩은 검색, 추천, 분류 등에 활용 가능

### 2) 학습 방식

* **Distributed Memory (DM)**: CBOW와 유사, 문맥 단어 + 문서 벡터로 중심 단어 예측
* **Distributed Bag of Words (DBOW)**: Skip-gram과 유사, 문서 벡터로 단어를 예측

### 3) 구현 예시

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = ["삼성전자는 반도체 사업을...", "LG전자는 가전 사업을..."]
tagged_docs = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(documents)]

model = Doc2Vec(documents=tagged_docs, vector_size=100, window=5, min_count=2, workers=4)

# 문서 벡터 추출
print(model.docvecs["0"])
```

* 각 문서마다 고유 태그(tags) 부여
* 학습 후 `model.docvecs`로 문서 벡터 추출

### 4) 유사도 계산

```python
similarity = model.docvecs.similarity("0", "1")
print("문서0-문서1 유사도:", similarity)
```

* 코사인 유사도를 통해 문서 간 유사도 측정

### 📌 핵심 정리

* Doc2Vec은 문서 전체를 벡터로 표현하는 기법
* DM, DBOW 두 가지 방식 존재
* 문서 간 유사도 계산, 추천 시스템, 문서 검색 등에 활용 가능

<br><br>
## 09-14. 실전! 한국어 위키피디아로 Word2Vec 학습하기<br>

* 대규모 한국어 코퍼스(위키피디아)를 활용해 Word2Vec 모델을 학습하는 실습임.
* 실제 산업·연구 환경에서 자주 사용되는 워드 임베딩 학습 과정 이해를 목표로 함.

### 1) 데이터 준비

* 한국어 위키피디아 덤프 파일(`kowiki-latest-pages-articles.xml.bz2`) 다운로드
* `wikiextractor` 툴을 이용해 텍스트만 추출
* 문장 단위 분리 후 토큰화 작업 수행

### 2) 전처리

* 불필요한 기호·특수문자 제거
* 형태소 분석기(Okt, Mecab 등)로 토큰화
* 학습에 사용할 문장 리스트 형태로 데이터 구성

### 3) Word2Vec 학습

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=tokenized_data,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1  # Skip-gram
)
```

* `vector_size`: 임베딩 차원
* `window`: 주변 단어 크기
* `min_count`: 최소 등장 횟수
* `sg=1`: Skip-gram 방식

### 4) 모델 활용

```python
print(model.wv.most_similar("대한민국"))
print(model.wv.similarity("서울", "도쿄"))
```

* 유사 단어 탐색 및 단어 간 유사도 계산 가능

### 📌 핵심 정리

* 위키피디아 같은 대규모 코퍼스를 사용하면 풍부한 의미 표현 가능
* Word2Vec 학습 시 토큰화·정제 과정이 성능에 큰 영향
* 학습된 벡터는 검색, 분류, 추천 등 다양한 NLP 태스크에 활용 가능

<br><br>
