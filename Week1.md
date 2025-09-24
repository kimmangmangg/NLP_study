## 2-1. 토큰화(Tokenization)<br>
토큰화(Tokenization)는 문장을 토큰(token)이라는 단위로 나누는 작업.  
토큰은 문장을 구성하는 각 단어이며, 단어 토큰화와 문장 토큰화로 구분됨.

---

### 📌 단어 토큰화 (Word Tokenization)

- 문장을 단어 단위로 나누는 작업
- 구두점이나 특수문자 제거 후 → 공백 기준으로 분리
- 구두점, 특수문자 등을 전부 제거하면 토큰이 의미를 잃는 경우도 있음

**[예시]**
```python
문장: Time is an illusion. Lunchtime double so!
결과: ['Time', 'is', 'an', 'illusion', 'Lunchtime', 'double', 'so']
```
**[Don't 와 Jone's의 토큰화?]** <br>

  - NLTK의 `word_tokenize` : Do와 n't로 분리 / Jone과 's로 분리
  - NLTK의 `WordPunctTokenizer` : Don과 '와 t로 분리 / Jone과 '와 s로 분리
  - 케라스의 `text_to_word_sequence` : all소문자화 + don't/jone's으로 분리

```python
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
```
**[주의 사항]**
  - 구두점/특수문자 단순 제거하면 안됨 → AT&T, Ph.D. 등
  - 아포스트로피(') → don't, it's 등
  - 단어 내 띄어쓰기 존재 → rock 'n' roll, New York 등 <br>

**✅ [Penn Treebank Tokenization 예제 코드 및 결과]**
- 규칙 1. 하이푼으로 구성된 단어는 하나로 유지한다.
- 규칙 2. doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리해준다.
```python
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print('트리뱅크 워드토크나이저 :',tokenizer.tokenize(text))
```
```python
트리뱅크 워드토크나이저 : ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']

```

---

### 📌 문장 토큰화 (Sentence Tokenization)

- 문장을 종결 기호(`.`, `?`, `!`) 기준으로 나누는 작업
- 하지만, 마침표가 문장 끝이 아닌 경우도 있기 때문에 단순히 마침표로 문장을 구분지으면 안 됨.
- 마침표(`.`)는 약어, 숫자, 웹 주소 등에 사용되기도 하므로 처리 주의 필요

**[예시]**
```python
문장1: IP 192.168.56.31 서버에 들어가서 로그 파일 저장해서 aaa@gmail.com로 결과 좀 보내줘. 그 후 점심 먹으러 가자.
문장2: Since I'm actively looking for Ph.D. students, I get the same question a dozen times every year.

결과1: 2문장으로 분리 → ~보내줘./ ~가자.
결과2: 1문장으로 분리
```

**✅ [NLTK에서의 sent_tokenize예제 코드 및 결과]**
- NLTK는 단순히 마침표를 구분자로 하여 문장을 구분하지 않음
```python
from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print('문장 토큰화1 :',sent_tokenize(text))
```
```python
문장 토큰화1 : ['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to make sure no one was near.']
```
- NLTK는 Ph.D.를 문장 내의 단어로 인식하여 성공적으로 문장 구분 성공
```python
text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print('문장 토큰화2 :',sent_tokenize(text))
```
```python
문장 토큰화2 : ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
```
**✅ [KSS예제 코드 및 결과]**
```python
import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :',kss.split_sentences(text))
```
```python
한국어 문장 토큰화 : ['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다.', '이제 해보면 알걸요?']
```

---

### 📌 한국어에서의 토큰화

- 한국어는 교착어로서 조사, 어미 등이 단어에 붙는 구조
- 영어처럼 단순 공백으로 나누는 것이 의미 분석에 부적절함
- 어절(띄어쓰기) 기준이 아닌 **형태소 토큰화**를 통해 단어를 더 작은 의미 단위로 분리해야 함
  - 어절로 구분하면 여전히 '단어+조사'로 붙어 있기 때문에, 형태소를 기준으로 나누어야 조사까지 분리할 수 있음

**[형태소란?]**

- 뜻을 가진 가장 작은 말의 단위
- 자립형태소: 단독으로 사용 가능한 형태소 → 체언(명사, 대명사, 수사), 수식언(관형사, 부사), 감탄사
- 의존형태소: 다른 형태소와 함께 사용되는 것 → 접사, 어미, 조사, 어간

**[예시]**
```
문장: 에디가 책을 읽었다
어절 단위: 에디가, 책을, 읽었다
형태소 단위: 에디, -가, 책, -을, 읽-, -었-, -다
```
- '에디'라는 사람 이름과 '책'이라는 명사를 얻어내려면 **형태소 토큰화**가 필요함
- 한국어는 띄어쓰기가 무시되는 경우가 많아 이를 기준으로 자연어 처리가 어려움

---

### 📌 NLTK와 KoNLPy를 이용한 영어, 한국어 토큰화 실습

**[품사 태깅(Part-of-speech tagging)]**
- 단어 토큰화 과정에서 각 단어가 어떤 품사로 쓰였는지를 구분해놓는 작업을 의미
- 단어는 표기는 같지만 품사에 따라서 단어의 의미가 달라지는 경우 존재
- 'fly'는 동사로는 '날다' / 명사로는 '파리'

> ✅ NLTK에서는 Penn Treebank POS Tags라는 기준을 사용하여 품사를 태깅
- PRP는 인칭 대명사, VBP는 동사, RB는 부사, VBG는 현재부사, IN은 전치사, NNP는 고유 명사, NNS는 복수형 명사, CC는 접속사, DT는 관사를 의미
```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print('단어 토큰화 :',tokenized_sentence)
print('품사 태깅 :',pos_tag(tokenized_sentence))
```
```python
단어 토큰화 : ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
품사 태깅 : [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
```
> ✅ KoNLPy(코엔엘파이)의 Okt형태소 분석기를 사용하여 토큰화 수행
- 코엔엘파이를 통해서 사용할 수 있는 형태소 분석기 : Okt(Open Korea Text), 메캅(Mecab), 코모란(Komoran), 한나눔(Hannanum), 꼬꼬마(Kkma)
```python
from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 
```
```python
OKT 형태소 분석 : ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
OKT 품사 태깅 : [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
OKT 명사 추출 : ['코딩', '당신', '연휴', '여행']
```
- 코엔엘파이의 형태소 분석기들은 공통적으로
  - 분석기.morphs : 형태소 추출
  - 분석기.pos : 품사 태깅
  - 분석기.nouns : 명사 추출

> ✅ KoNLPy(코엔엘파이)의 꼬꼬마 형태소 분석기를 사용하여 토큰화 수행
```python
print('꼬꼬마 형태소 분석 :',kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 품사 태깅 :',kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 명사 추출 :',kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  
```
```python
꼬꼬마 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
꼬꼬마 품사 태깅 : [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
꼬꼬마 명사 추출 : ['코딩', '당신', '연휴', '여행']
```
- 각 형태소 분석기는 성능과 결과가 다르게 나오기 때문에 목적에 맞는 분석기를 사용해야 함<br><br><br><br>



## 2-2. 정제(Cleaning) and 정규화(Normalization)

정제(Cleaning)와 정규화(Normalization)는 텍스트에서 불필요한 요소를 제거하고 일관된 형태로 통일하는 과정.  
불필요하거나 노이즈가 많은 데이터를 전처리해 모델 학습에 도움이 되는 형태로 만드는 것이 목적임.

---

### 📌 정제(Cleaning)

- 텍스트 데이터에는 전처리가 필요한 다양한 **노이즈(Noise)** 가 존재함
  - 자연어가 아니면서 아무 의미도 갖지 않는 글자들(특수 문자 등)
  - 혹은, 분석하고자 하는 목적에 맞지 않는 불필요 단어
- 이 노이즈를 제거하는 작업이 **정제**임
  - 등장 빈도가 적은 단어 : 보통 도움이 되지 않아 정제 가능
  - 길이가 짧은 단어 : 영어권 언어에서는 길이가 짧은 단어를 삭제하는 것이 어느정도 정제의 효과 있음
- 영어 단어의 평균 길이는 6-7 정도, 한국어 단어의 평균 길이는 2-3 정도로 추정

**[예시]**
```
- 특수 문자, HTML 태그 등 제거
- 중복 문자 → ‘ㅋㅋㅋㅋ’ → ‘ㅋㅋ’처럼 축약
- 무의미한 기호, 이메일 주소, URL 등 제거
```
 
**✅ [정규 표현식을 활용한 정제 예시 코드 및 결과]**
```python
import re
text = "I was wondering if anyone out there could enlighten me on this car."

# 길이가 1~2인 단어들을 정규 표현식을 이용하여 삭제
shortword = re.compile(r'\W*\b\w{1,2}\b')
print(shortword.sub('', text))
```
```python
was wondering anyone out there could enlighten this car.
```

---

### 📌 정규화(Normalization)

- 서로 다른 표현을 같은 단어로 통일하는 작업
  - USA와 US / uh-huh와 uhhuh 등
  - 정규화를 거치면 US를 찾아도 USA도 함께 찾을 수 있을 것
- 대소문자 → 많은 경우 소문자로 통합하나, 고유명사 등은 대문자 유지
- 표현 방식이 다른 단어들을 통일하여 모델이 일관되게 학습할 수 있도록 만듦

**[예시]**
```
- ‘예뻐’, ‘이뻐’ → 같은 의미로 통일
- ‘그랬다’ → ‘그렇다’, ‘먹었다’ → ‘먹다’
- 단어의 통일된 표기를 위한 작업
```
<br><br><br>

## 2-3. 어간 추출(Stemming) and 표제어 추출(Lemmatization)

- 정규화 기법 중 코퍼스에 있는 단어의 개수를 줄일 수 있는 기법의 일종
- 단어를 표준화된 형태로 변환하여 처리 단위를 줄이고, 단어 간 중복을 제거하는 작업임.  
- 두 기법 모두 단어를 **기본형**이나 **원형**에 가깝게 변형시켜서 의미는 유지하면서 단어 수를 줄이는 데 사용됨.

---

### 📌 어간 추출 (Stemming)

- 단어의 **어간(stem)** 을 추출하는 작업
- 어간은 단어에서 **어미(접미사)** 를 제거한 나머지 부분
- 어간 추출 후에 나오는 결과 단어는 사전에 존재하지 않는 단어일 수도 있음
- 정확한 문법적 분석 없이 규칙에만 집중하여 잘라내므로 **의미가 훼손될 수 있음**
- 표제어 추출에 비해 빠르지만 정확도는 낮을 수 있음

**[예시]**
```
어간 추출 전 : ['formalize', 'allowance', 'electricical']
어간 추출 후 : ['formal', 'allow', 'electric']
```

**✅ [NLTK의 PorterStemmer 예제 코드 및 결과]**

- 포터 어간 추출기는 정밀하게 설계되어 정확도가 높으므로 영어 자연어 처리에서 어간을 추출하고자 하면 적절한 선택임

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

sentence = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
tokenized_sentence = word_tokenize(sentence)

print('어간 추출 전 :', tokenized_sentence)
print('어간 추출 후 :',[stemmer.stem(word) for word in tokenized_sentence])
```
```python
어간 추출 전 : ['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']
어간 추출 후 : ['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', 'with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']
```

**✅ [Porter와 Lancaster 비교 예제 코드 및 결과]**
```python
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print('어간 추출 전 :', words)
print('포터 스테머의 어간 추출 후:',[porter_stemmer.stem(w) for w in words])
print('랭커스터 스테머의 어간 추출 후:',[lancaster_stemmer.stem(w) for w in words])
```
```python
어간 추출 전 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
포터 스테머의 어간 추출 후: ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
랭커스터 스테머의 어간 추출 후: ['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']
```

---

### 📌 표제어 추출 (Lemmatization)

- 단어의 **표제어(lemma)** 를 찾아가는 작업
- 단어들이 다른 형태를 가지더라도, 그 뿌리 단어(=표제어)를 찾아가서 단어의 개수를 줄임
- 문맥과 품사를 고려하여 변형된 단어를 그 **기본형**으로 복원
- 어간 추출보다 정확도 높지만, 속도는 느림

**[예시]**
```python
am, are, is → be
```

**[형태학적 파싱]**
- 어간과 접사를 분리하는 작업
  - cats : cat(어간)와 -s(접사)를 분리
  - fox : 더 이상 분리할 수 없음

**✅ [NLTK의 WordNetLemmatizer 예제 코드 및 결과]**

- 단어의 형태가 적절히 보존되지만, 
- dy나 ha와 같이 의미를 알 수 없는 적절하지 못한 단어를 출력
- 본래 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있음 → **품사 명시 필요**

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('표제어 추출 전 :',words)
print('표제어 추출 후 :',[lemmatizer.lemmatize(word) for word in words])
```
```python
표제어 추출 전 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
표제어 추출 후 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']
```

**✅ [WordNetLemmatizer의 품사 명시 예제]**
```python
lemmatizer.lemmatize('doing', 'v') → do
lemmatizer.lemmatize('fly', 'v') → fly
```

| 입력 문장     | Stemming 결과 | Lemmatization 결과 |
| --------- | ----------- | ---------------- |
| am        | am          | be               |
| the going | the go      | the going        |
| having    | hav         | have             |

---

### 📌 한국어에서의 어간 추출

- 용언(동사, 형용사)의 경우 어간+어미 결합
- 규칙활용 : 어간은 활용시에도 불변, 어미는 활용시 형태가 변화 → 분리 용이
- 불규칙활용 : 어간/어미 분리를 위해 보다 복잡한 규칙을 적용해야 함..


<br><br><br>

## 2-4. 불용어(Stopword)

- 불용어(stopword)란 문장에서 자주 등장하지만 문맥상 큰 의미를 가지지 않는 단어를 말함.  
- 예: 영어의 a, the, in, I, my, me, over / 한국어의 조사, 접속사, 접미사 등

---

### 📌 불용어의 필요성

- 텍스트 데이터에는 의미를 크게 담고 있지 않은 단어가 다수 존재함
- 이들을 제거하면:
  - 데이터 크기 감소
  - 연산 효율 향상
  - 불필요한 노이즈 제거 가능
- 하지만 특정 작업에서는 불용어가 의미를 가질 수도 있으므로 **목적에 따라 제거 여부 결정** 필요함

---

### 📌 불용어 제거 방법 (영어)

- NLTK는 영어 불용어 사전을 제공함.(내장된 불용어 사전이 있음)  
- 불용어 리스트를 불러오고, 토큰화된 단어 중 불용어를 제거할 수 있음.

**✅ [NLTK 불용어 제거 예시]**
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
example = "Family is not an important thing. It's everything."
word_tokens = word_tokenize(example)

result = []
for w in word_tokens:
    if w not in stop_words:
        result.append(w)

print(result)
```
```python
['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
```

---

### 📌 불용어 제거 방법 (한국어)

- 불용어는 개발자가 직접 정의할 수도 있음
- 한국어는 불용어 사전이 내장되어 있지 않음
- 한국어의 경우 불용어 사전을 직접 구축한 후 제거해야 함
- 예: 의, 가, 이, 은, 들, 는, 좀, 잘, 걍, 과, 도, 를, 으로, 자, 에, 와, 한, 하다 등

**✅ [한국어 불용어 제거 예시]**
```python
from konlpy.tag import Okt

okt = Okt()

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "를 아무렇게나 구 우려 고 안 돼 같은 게 구울 때 는"

stop_words = set(stop_words.split(' '))
word_tokens = okt.morphs(example)

result = [word for word in word_tokens if not word in stop_words]

print('불용어 제거 전 :',word_tokens) 
print('불용어 제거 후 :',result)

```
```python
불용어 제거 전 : ['고기', '를', '아무렇게나', '구', '우려', '고', '하면', '안', '돼', '.', '고기', '라고', '다', '같은', '게', '아니거든', '.', '예컨대', '삼겹살', '을', '구울', '때', '는', '중요한', '게', '있지', '.']
불용어 제거 후 : ['고기', '하면', '.', '고기', '라고', '다', '아니거든', '.', '예컨대', '삼겹살', '을', '중요한', '있지', '.']
```

<br><br><br>

## 2-5. 정규 표현식(Regular Expression)

1. 정규 표현식은 문자열에서 **특정 패턴을 찾고, 추출, 치환**하는 데 사용하는 강력한 도구임.  
2. 토큰화, 정제, 정규화 과정에서 불필요한 기호 제거, 특정 단어 추출 등에 활용됨.  
3. 메타문자(`. ^ $ * + ? { } [ ] \ | ( )`)를 사용하여 다양한 패턴 정의 가능함.  
4. 파이썬의 `re` 모듈을 사용해 `match`, `search`, `findall`, `sub` 등의 함수로 정규 표현식을 적용할 수 있음.  
5. 자연어 처리에서는 텍스트에서 **이메일, URL, 숫자, 특수문자 제거** 등 전처리에 자주 사용됨.  
> 자세한 내용은 링크(https://wikidocs.net/21703)

<br><br><br>

## 2-6. 정수 인코딩(Integer Encoding)

단어를 숫자로 변환하여 컴퓨터가 처리할 수 있도록 하는 과정.  
자연어 처리를 위해 텍스트 데이터를 수치화하는 가장 기초적인 방법 중 하나임.

---

### 📌 정수 인코딩 필요성

- 기계 학습 모델은 텍스트가 아닌 **숫자 벡터**만 입력으로 처리 가능함
- 단어를 고유한 정수에 매핑해 문장을 수치화
- 이후 원-핫 인코딩, 워드 임베딩 등 더 발전된 기법의 기반이 됨

---

### 📌 단어 집합(Vocabulary) 구축

- 코퍼스의 모든 단어를 중복 제거하여 리스트화 → 단어 집합(vocabulary)
- 각 단어에 고유한 인덱스를 부여
- 빈도수 기반으로 정렬하여 인덱스를 부여하기도 함


**✅ [단어 집합 만들기 예시 코드 및 결과]**
```python
from nltk.tokenize import word_tokenize

sentence = "The earth is an awesome place live"
tokens = word_tokenize(sentence)
print(tokens)
```
```python
['The', 'earth', 'is', 'an', 'awesome', 'place', 'live']
```


**✅ [정수 인코딩 예시 코드 및 결과]**
```python
vocab = {t: i for i, t in enumerate(tokens)}
print(vocab)
```
```python
{'The': 0, 'earth': 1, 'is': 2, 'an': 3, 'awesome': 4, 'place': 5, 'live': 6}
```


**✅ [케라스(Keras) 활용 예시 코드 및 결과]**
- `Tokenizer` 객체를 이용해 단어 집합 자동 생성 및 정수 인코딩 가능
```python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ["The earth is an awesome place live", "The earth is great place live"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

print(tokenizer.word_index)  # 단어 집합
print(tokenizer.texts_to_sequences(sentences))  # 정수 인코딩 결과
```

<br><br><br>

## 2-7. 패딩(Padding)

자연어 문장은 길이가 다르기 때문에 기계 학습 모델에 입력하기 위해서는 길이를 맞춰야 함.  
짧은 문장은 뒤에 **패딩(padding) 토큰**을 붙여 동일한 길이로 맞추는 과정을 거침.

---

### 📌 패딩의 필요성

- 딥러닝 모델 입력은 동일한 크기의 행렬이어야 함
- 문장마다 길이가 다르면 학습 불가
- 따라서 가장 긴 문장의 길이에 맞춰 나머지를 채움

---

### 📌 패딩 방식

- 주로 숫자 `0`을 패딩 토큰으로 사용
- 패딩 위치:
  - 앞에 붙이는 경우 (pre-padding)
  - 뒤에 붙이는 경우 (post-padding)

**✅ [케라스(Keras)에서의 패딩 예시 코드 및 결과]**

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [[1, 2, 3], [4, 5], [6]]
padded = pad_sequences(sentences)
print(padded)
```
```python
[[1 2 3]
 [0 4 5]
 [0 0 6]]
```
- 기본은 pre-padding 방식
- 옵션 padding='post' 로 설정하면 `post-padding` 가능

```python
padded = pad_sequences(sentences, padding='post')
print(padded)
```
```python
[[1 2 3]
 [4 5 0]
 [6 0 0]]
```

---

### 📌 패딩시 주의사항

- 패딩으로 추가된 0은 실제 단어가 아니므로 모델 학습 시 무시되도록 처리 필요
- RNN, LSTM 등에서는 masking 기법을 활용하여 패딩 위치를 무시함

<br><br><br>


## 2-8. 원-핫 인코딩(One-Hot Encoding)

단어를 고유한 정수로 변환한 뒤, 그 정수를 다시 벡터로 표현하는 방법.  
각 단어는 단어 집합의 크기만큼의 차원을 가진 벡터로 표현되며, 해당 인덱스만 1이고 나머지는 모두 0임.

---

### 📌 원-핫 인코딩 과정

1. 단어 집합(vocabulary) 구축
2. 각 단어에 정수 인덱스 부여
3. 인덱스를 기준으로 원-핫 벡터 생성

**[예시]**
```python
문장: "나는 자연어 처리를 배운다"
단어 집합: {'나는':0, '자연어':1, '처리':2, '를':3, '배운다':4}

'자연어' → [0, 1, 0, 0, 0]
'배운다' → [0, 0, 0, 0, 1]
```

**✅ [구현 예시 코드 및 결과]**
```python
from konlpy.tag import Okt

okt = Okt()
tokens = okt.morphs("나는 자연어 처리를 배운다")
vocab = {t: i for i, t in enumerate(tokens)}

def one_hot_encoding(word, vocab):
    one_hot = [0] * len(vocab)
    idx = vocab[word]
    one_hot[idx] = 1
    return one_hot

print(one_hot_encoding("자연어", vocab))
```
```python
[0, 1, 0, 0, 0]
```

**✅ [케라스dml to_categorical 함수 예시 코드 및 결과]**
```python
from tensorflow.keras.utils import to_categorical

encoded = [0, 1, 2, 3]
one_hot = to_categorical(encoded)
print(one_hot)
```
```python
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
```

---

### 📌 원-핫 인코딩의 한계
- 벡터 차원이 단어 집합 크기와 같음 → 단어 수가 많아지면 차원 폭발 발생
- 단어 간 유사도 표현 불가능
  - 예: '강아지'와 '개'는 유사하지만 원-핫 벡터는 전혀 다름
- 이후 분산 표현(Distributed Representation), 즉 임베딩 기법으로 발전 및 보완됨


<br><br><br>

## 2-9. 데이터의 분리(Splitting Data)

- 머신 러닝이나 딥러닝 모델을 학습할 때 데이터는 보통 **훈련용 데이터와 평가용 데이터**로 나눔.  
- 모델의 성능을 공정하게 측정하고, 과적합을 방지하기 위해 데이터 분리는 필수 과정임.

---

### 📌 데이터 분리 목적

- 훈련 데이터(train set): 모델 학습에 사용
- 테스트 데이터(test set): 모델의 일반화 성능 평가에 사용
- 일부는 검증 데이터(validation set)로도 나누어 하이퍼파라미터 조정에 활용

**✅ [데이터 분리 예시 코드 및 결과]**

- 사이킷런의 `train_test_split` 함수를 활용하여 데이터를 쉽게 분리 가능함.

```python
from sklearn.model_selection import train_test_split

X = [i for i in range(10)]
y = [i*2 for i in range(10)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("훈련 데이터:", X_train, y_train)
print("테스트 데이터:", X_test, y_test)
```

- 일반적으로 훈련:테스트 = 8:2 또는 7:3
- 데이터가 충분하다면 훈련:검증:테스트 = 6:2:2
  - 데이터 크기와 문제 특성에 따라 적절히 조정

<br><br><br>

## 2-10. 한국어 전처리 패키지(Text Preprocessing Tools for Korean Text)

- 한국어는 교착어로 조사, 어미 등이 단어에 붙어 있기 때문에 영어보다 전처리가 복잡함.  
- 이를 위해 다양한 형태소 분석기와 전처리 패키지를 활용함.

---

### 📌 KoNLPy

- 파이썬에서 한국어 형태소 분석기를 사용할 수 있게 만든 패키지
- 지원 분석기: Okt, Mecab, Komoran, Hannanum, Kkma
- 주요 기능:
  - `morphs(text)` : 형태소 추출
  - `pos(text)` : 형태소 + 품사 태깅
  - `nouns(text)` : 명사만 추출

**✅ [KoNLPy 예시 코드 및 결과]**
```python
from konlpy.tag import Okt

okt = Okt()
print(okt.morphs("나는 자연어 처리를 배운다"))
print(okt.pos("나는 자연어 처리를 배운다"))
print(okt.nouns("나는 자연어 처리를 배운다"))
```

---

### 📌 soynlp
- 비지도학습 기반의 한국어 형태소 분석 패키지
- 띄어쓰기, 단어 점수 계산, 토큰화 등 기능 지원
- 학습 코퍼스 기반으로 단어의 응집도(cohesion)와 브랜칭 엔트로피(branching entropy) 계산 가능

---

### 📌 khaiii
- 카카오에서 공개한 한국어 형태소 분석기
- 딥러닝 기반으로 높은 정확도 제공
- 다른 분석기 대비 설치와 환경 설정이 복잡할 수 있음

---

### 📌 kss
- 한국어 문장 토큰화에 특화된 패키지
- 종결 어미, 문장 부호 등을 활용해 문장 단위 분리
- 한국어의 문장 경계 인식에 자주 활용됨

