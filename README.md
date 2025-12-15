# 🐾 애완동물(개·고양이) 품종 분류 AI 시스템  
## 인공지능 개발 프로젝트 최종보고서

**프로젝트명:** AI Pet Breed Classifier (37종)  
**과목명:** 인공지능개발프로젝트  
**학번 / 이름:** 2021143021 김병수
**제출일:** 2025년 12월 15일

---

## 1. 프로젝트 개요

### 1.1 프로젝트명

**AI Pet Breed Classifier (37종)**  
Oxford-IIIT Pet Dataset 기반 개·고양이 품종 분류 시스템

### 1.2 프로젝트 목표

이미지 한 장만으로 **개·고양이 37종의 품종을 자동으로 분류**하고,  
웹 화면에서 품종 이름·예측 확률·상위 3개 후보를 직관적으로 보여주는 AI 시스템을 구축하는 것이 목표이다.

구체적으로는 다음을 달성하고자 한다.

- 개·고양이 품종 37종에 대한 이미지 분류 모델 구축
- 희귀하거나 생소한 품종도 사람보다 빠르게 인식하는 보조 도구 역할 수행
- 딥러닝 모델을 **Streamlit 웹 애플리케이션**과 연결하여  
  사용자가 코드 지식 없이도 브라우저에서 바로 테스트할 수 있는 형태로 제공

### 1.3 개발 동기 및 필요성

반려동물 문화가 확산되면서, SNS나 입양 공고 등에서 다양한 품종의 개와 고양이를 접하게 된다.  
하지만 **일반 사용자가 사진만 보고 정확한 품종을 구분하기는 쉽지 않다.**

특히 다음과 같은 상황에서 품종 인식 보조 도구가 유용할 수 있다.

- 보호소/입양 공고에서 잘 알려지지 않은 품종의 특징을 알고 싶을 때  
- 반려동물을 처음 보는 사람에게 “어떤 품종인지”를 설명해주고 싶을 때  
- 비슷하게 생긴 견종(예: pit bull 계열)이나 고양이 단모종을 구분하고 싶을 때  

이에 따라, **사진 한 장만으로 품종 후보를 빠르게 제안해주는 AI 서비스**를 만들고,  
그 과정에서 딥러닝 모델 학습부터 웹 서비스 배포 직전 단계까지의 개발 과정을 모두 경험하고자 했다.

---

## 2. 개발 환경 및 도구

| 항목 | 내용 |
| --- | --- |
| **언어** | Python 3.10 |
| **딥러닝 프레임워크** | TensorFlow 2.x / Keras |
| **모델 백본** | MobileNetV2 (ImageNet 사전 학습) |
| **웹 프레임워크(UI)** | Streamlit |
| **데이터셋** | Oxford-IIIT Pet Dataset (개·고양이 37종) |
| **이미지 처리** | Pillow(PIL), NumPy |
| **개발 환경** | Google Colab(모델 학습), VSCode + Windows(로컬 실행) |
| **버전 관리** | Git / GitHub (보고서 및 코드 관리) |

---

## 3. 시스템 구조

### 3.1 전체 시스템 구성

본 프로젝트는 크게 **모델 학습 파트**와 **웹 서비스 파트**로 나눌 수 있다.

1. **모델 학습 파이프라인 (Colab)**  
   - Oxford-IIIT Pet Dataset 로드  
   - 이미지 전처리 (리사이즈, 정규화, 라벨 인코딩)  
   - MobileNetV2를 활용한 Transfer Learning  
   - 학습된 모델을 `.h5` 포맷으로 저장  

2. **웹 애플리케이션 파이프라인 (Streamlit)**  
   - 사용자가 웹에서 이미지를 업로드  
   - 업로드된 이미지를 224×224로 전처리 후 정규화  
   - 로컬에 저장된 `pet_breed_classifier_finetuned.h5` 로 예측 수행  
   - 예측 결과(품종 이름, 확률, Top-3 후보)를 시각적으로 출력  

### 3.2 흐름도

```text
사용자 이미지 업로드
      ↓
이미지 전처리 (224×224, 정규화)
      ↓
Fine-Tuned MobileNetV2 분류 모델
      ↓
품종별 확률 벡터 (37차원 Softmax 출력)
      ↓
가장 높은 확률의 품종 + 상위 3개 품종 선택
      ↓
Streamlit UI에 결과 시각화 (메시지 + Top-3 리스트)
```

## 4. 모델 학습 및 성능

### 4.1 모델 구조

| 항목 | 내용 |
|:---:|:---|
| **Base Model** | MobileNetV2 (ImageNet 사전 학습, `include_top=False`) |
| **추가 레이어** | Global Average Pooling → Dense(ReLU) → Dropout → Dense(37, Softmax) |
| **입력 크기** | 224 × 224 × 3 (RGB 이미지) |
| **출력 클래스** | 개·고양이 37개 품종 |
| **손실 함수** | Categorical CrossEntropy |
| **옵티마이저** | Adam (학습률은 코드에서 조정) |

### 4.2 학습 전략

* **데이터 분할**
  * 학습용 / 검증용 데이터로 분리하여 과적합 여부를 확인하였다.
* **전처리**
  * 모든 이미지를 `224×224` 크기로 리사이즈
  * 픽셀 값 / `255.0` 으로 정규화하여 0~1 범위로 맞춤
* **Fine-Tuning**
  * MobileNetV2의 상단 일부 레이어는 고정하거나, 후반부 레이어부터 점진적으로 학습을 허용하여 성능과 속도의 균형을 맞추었다.
* **모델 저장**
  * 학습이 끝난 모델을 `.h5` 형식으로 저장(`pet_breed_classifier_finetuned.h5`), 로컬에서 Streamlit 앱이 직접 불러 사용할 수 있도록 구성하였다.

### 4.3 성능 요약

별도의 정량적인 테스트 셋(정답 라벨 포함)을 따로 구축하지는 못했지만, 여러 품종(37종)에 대해 직접 이미지를 넣어본 결과 다음과 같은 특징을 확인했다.

* **Miniature Pinscher, Samoyed, Scottish Terrier** 등 외형이 뚜렷한 견종은 95~99% 수준의 높은 확률로 일관되게 예측되었다.
* **Sphynx**와 같은 독특한 고양이 품종에서도 높은 확률(98% 이상)을 보여주었다.
* 유사한 외형을 가진 종들(예: **Staffordshire Bull Terrier vs American Pit Bull Terrier**)은 1, 2위 확률이 서로 비슷하게 나와 모델도 애매해하는 모습을 보였다.
* 평균 추론 시간은 로컬 CPU 환경 기준으로 약 **1~2초** 수준으로, Streamlit 웹 앱에서 실시간으로 사용하기에 충분한 속도이다.


## 5. 문제점 및 해결 과정 🛠️

프로젝트 개발 과정에서 발생한 주요 기술적 이슈와 해결 과정을 정리하였다. "문제 → 원인 → 해결" 구조로 기록하여 향후 유사한 문제 발생 시 참고할 수 있도록 구성했다.

### 5.1 TensorFlow & Keras 버전 호환성 문제

* **문제**
  * Colab(TF 2.16+)에서 학습된 모델(`.h5`)을 로컬 환경에서 로드할 때 `InputLayer` 설정 오류 및 `keras` 속성 참조 에러 발생.
* **원인**
  * Keras 3.0 업데이트 이후 저장 포맷이 변경되었으나, 로컬에는 구버전 TensorFlow/Keras가 설치되어 있어 라이브러리 충돌 발생.
* **해결**
  * 로컬 라이브러리를 최신 버전으로 업그레이드.
  * 코드 내 모델 로드 방식을 변경하여 호환성 확보.
  ```python
  # 변경 전
  # model = tf.keras.models.load_model(...)

  # 변경 후 (Keras 3 API 호환)
  import keras
  model = keras.models.load_model("pet_breed_classifier_finetuned.h5")

 ### 5.2 Streamlit 실행 컨텍스트 오류

* **문제**
  * IDE(VSCode 등)의 'Run' 버튼으로 `app.py` 실행 시 `missing ScriptRunContext` 경고와 함께 웹앱 구동 실패.
* **원인**
  * Streamlit은 일반 Python 스크립트 실행 방식이 아닌, 자체 CLI 명령어를 통해 웹 서버 위에서 동작하는 구조.
* **해결**
  * 터미널에서 명령어로 실행하는 방식을 표준으로 정립하고 `README`에 명시.
  ```bash
  streamlit run app.py
  ```
### 5.3 유사 품종 혼동에 대한 판단 보조 시스템 구축

#### **문제**
* **유사 품종 간 구분 난이도:** `Russian Blue` vs `British Shorthair` 또는 `Staffordshire Bull Terrier` vs `American Pit Bull Terrier` 등 외형이 생물학적으로 매우 유사한 품종의 경우, AI 모델조차 100% 확신하지 못하고 확률이 분산되는 현상이 발생했다.
* **단일 결과의 위험성:** 확률 차이가 근소함에도 불구하고 가장 높은 확률(Top-1)만 보여줄 경우, 2순위가 된 실제 정답을 사용자가 놓칠 위험이 있었다.

#### **해결**
* 모델의 구조를 무리하게 변경하여 과적합(Overfitting)을 유발하기보다, **사용자에게 판단 근거를 투명하게 제공하는 '의사결정 보조(Decision Support)' 방식**으로 접근하여 문제를 해결했다.

* **1. 다중 후보 제안 (Top-3 Candidates):**
    * 예측 확률이 가장 높은 1순위 결과뿐만 아니라, **2순위와 3순위 후보를 함께 리스트 형태로 시각화**했다.
    * 이를 통해 사용자는 AI가 헷갈려하는 경쟁 후보들을 한눈에 확인하고, 최종적으로 올바른 판단을 내릴 수 있다.

* **2. 확률의 투명한 공개 (Probability Exposure):**
    * 단순히 "이 품종입니다"라고 단정 짓지 않고, **"71.96%"** 와 같이 모델의 **확신 정도(Confidence Level)**를 수치로 명시했다.
    * 이를 통해 사용자는 결과가 '확실한 정답'인지, 혹은 '참고용 추론'인지를 구분할 수 있게 되었다.

### 5.4 가상환경 및 명령어 혼동

* **문제**
  * Python REPL 내부에서 쉘 명령어(`pip show`)를 입력해 `SyntaxError` 발생.
  * PowerShell에서 가상환경 활성화 시 경로 오류(`&` 누락)로 "명령을 인식할 수 없다"는 오류 발생.
* **원인**
  * Python 인터프리터와 OS 쉘(PowerShell) 환경의 문법 및 실행 컨텍스트를 혼동함.
* **해결**
  * 프로젝트용 가상환경(`.venv`)을 생성하고, 활성화 및 설치 패턴을 고정하여 사용.
  ```powershell
  # 가상환경 활성화 (PowerShell)
  & .\.venv\Scripts\Activate

  # 패키지 설치
  pip install streamlit tensorflow keras pillow numpy
  ```

### 5.5 class_names.txt 로딩 문제

* **문제**
  * `class_names.txt` 파일을 읽을 때 `FileNotFoundError` 발생.
* **원인**
  * 실행 위치와 파일 위치 불일치 또는 상대 경로 설정 오류.
* **해결**
  * 파일을 `app.py`와 동일 경로에 배치하고, `st.cache_data`를 적용한 로더 함수로 처리.
  ```python
  @st.cache_data
  def load_class_names():
      try:
          with open("class_names.txt", "r", encoding="utf-8") as f:
              return [line.strip() for line in f]
      except FileNotFoundError:
          st.error("class_names.txt 파일이 같은 폴더에 있는지 확인해주세요.")
          return []
  ```

### 5.6 Top-3 예측 번호 출력 오류

* **문제**
  * 상위 3개 예측 결과를 리스트로 출력할 때, 모든 번호가 `1.`로 표시되어 가독성 저하.
* **원인**
  * 마크다운 출력 문자열 포맷팅 시 리스트 번호를 하드코딩했기 때문.
* **해결**
  * `enumerate` 함수를 사용하여 순위를 동적으로 할당.
  ```python
  top_3_indices = np.argsort(prediction[0])[-3:][::-1]

  for rank, idx in enumerate(top_3_indices, start=1):
      name = format_breed_name(class_names[idx])
      prob = prediction[0][idx] * 100
      st.write(f"{rank}. **{name}**: {prob:.2f}%")
  ```

### 5.7 이미지 전처리 차원(Dimension) 불일치 (New)

* **문제**
  * 사용자가 업로드한 이미지는 `(224, 224, 3)` 형태의 3차원 배열이나, 모델은 `(1, 224, 224, 3)` 형태의 4차원 배치(Batch) 데이터를 요구하여 `ValueError: Shape mismatch` 발생.
* **원인**
  * Keras 모델은 기본적으로 한 번에 여러 장의 이미지를 처리하도록(Batch Processing) 설계되어 있기 때문.
* **해결**
  * `numpy.expand_dims` 함수를 사용하여 이미지 데이터에 배치 차원을 추가.
  ```python
  # (Height, Width, Channel) -> (Batch, Height, Width, Channel)
  img_array = np.expand_dims(img_array, axis=0)
  ```

## 5.8 개발 중 느낀 점 (Retrospective)

### 1. End-to-End 개발 파이프라인의 직접 경험
단순히 단일 모델 코드를 작성하는 것에 그치지 않고, **[환경 설정 → 모델 학습 → 모델 저장/로드 → 웹 실행 → UX 설계]**로 이어지는 개발의 전 과정을 직접 수행했다. 이 과정에서 데이터와 모델, 그리고 서비스가 어떻게 유기적으로 연결되는지 깊이 이해할 수 있었다.

### 2. 디테일과 트러블 슈팅의 중요성 체감
라이브러리 버전 충돌, 파일 경로 설정, 실행 명령어 등 사소해 보이는 요소들이 실제 개발 단계에서는 **서비스 구동을 막는 치명적인 장애물**이 될 수 있음을 체감했다. 이를 통해 코딩 능력뿐만 아니라 개발 환경을 통제하고 문제를 해결하는 능력이 필수적임을 깨달았다.

### 3. AI 모델에 대한 관점 변화
유사 품종을 완벽히 구분하는 것은 인간에게도 매우 어려운 문제였다. 따라서 AI를 무조건적인 정답을 내놓는 **'정답 기계(Answer Machine)'**로 인식하기보다, 확률적 근거를 바탕으로 사용자의 판단을 돕는 **'의사결정 보조 도구(Decision Support Tool)'**로 설계하는 관점이 서비스 개발에 있어 훨씬 현실적이고 중요하다는 점을 배웠다.


## 6. 결과 분석 (Result Analysis)

### 6.1 확신도 높은 견종·묘종 (High Confidence Cases)

**1) Scottish Terrier (스코티시 테리어)**
* **결과:** 예측 확률 **99.91%** 기록.
* **분석:** 입력된 이미지가 두 마리의 강아지를 포함하고 있는 복잡한 구도였음에도 불구하고, 품종 고유의 외형적 특징(검은 털, 입 주변 수염 등)을 정확히 포착하여 거의 **100%에 가까운 확신도**를 보였다.
<img width="594" height="648" alt="111" src="https://github.com/user-attachments/assets/c3b3fb99-a2ae-41a3-ac4b-5a4c07312d3f" />

*
*<img width="571" height="730" alt="111111" src="https://github.com/user-attachments/assets/0bae7aa4-1764-4219-82b9-286093a1d98d" />



**2) Miniature Pinscher (미니어처 핀셔)**
* **결과:** 예측 확률 **99.81%** 기록.
* **분석:** 짧은 털과 뾰족한 귀 등 특징이 뚜렷한 견종에 대해서는 배경이나 조명과 관계없이 매우 **안정적인 성능**을 발휘함을 확인했다.


**3) Sphynx (스핑크스 고양이)**
* **결과:** 예측 확률 **98.81%** 기록.
* **분석:** 털이 없는 독특한 피부 질감과 큰 귀라는 명확한 특징 덕분에, 고양이 품종 중에서도 **매우 높은 정확도**로 분류되었다.

### 6.2 유사 품종 혼동 사례 (Confusion Cases)

**1) Newfoundland vs Leonberger (대형견 - 실험 결과)**
* **현상:** **Newfoundland(63.08%)**가 1순위로 예측되었으나, 2순위인 **Leonberger(34.11%)**와 확률 차이가 크지 않았다. 수치상으로는 1위(63%)와 2위(34%)의 격차가 커 보일 수 있으나, 타 품종들이 99% 이상의 확신도를 보인 것과 비교하면 이는 현저히 낮은 신뢰도이다. 이는 Newfoundland와 Leonberger의 외형이 매우 흡사하여 모델이 명확한 판단을 유보했음을 의미한다. 즉, 1순위 확신도가 낮게 측정된 것은 모델이 2순위 후보(Leonberger) 역시 유력한 정답으로 인지하여 확률을 분산시킨 합리적인 결과로 해석된다.
* **분석:** 두 견종은 모두 체구가 거대하고 풍성한 갈색/검정 털을 가지고 있어 실제 육안으로도 구별이 쉽지 않다. AI가 압도적인 확률을 내놓지 않고 확률을 분산시킨 것은, 단순히 오답을 낸 것이 아니라 **유사한 특징을 가진 후보군 사이에서 합리적인 추론**을 수행했음을 시사한다.
* 
<img width="513" height="803" alt="515" src="https://github.com/user-attachments/assets/86ecd8d0-a69f-44d8-a622-4cf691ce2439" />


**2) Russian Blue vs British Shorthair (회색 묘종 - 심층 분석)**

* **실험 결과 (Phenomenon):**
    * **Russian Blue(71.96%)**로 정답을 맞혔으나, 2순위인 **British Shorthair(20.47%)**가 비교적 높은 확률로 뒤따랐다.
    * 앞선 **Scottish Terrier(99.9%)** 사례와 달리, 정답임에도 불구하고 확신도가 약 70%대로 낮게 조정된 모습이다.

* **상세 분석 (Core Insight):**
    * 두 품종은 모두 **'청회색 단모'**라는 강력한 시각적 공통점을 가지고 있어 육안으로도 구분이 어렵다.
    * 이때 모델이 확신도를 99%가 아닌 71%로 낮춘 것은 성능 저하가 아니라, **입력 이미지의 난이도를 정확히 인지(Calibration)**하고 있음을 보여준다.
    * 특히 외형이 매우 흡사한 British Shorthair를 2순위로 지목했다는 점은, 모델이 단순히 이미지를 암기한 것이 아니라 **'회색 단모'와 '체형' 같은 구체적인 특징(Feature)을 학습**했음을 증명한다.

* **결과:**
    * 결과적으로 본 모델은 쉬운 케이스와 어려운 케이스를 명확히 구분하는 **변별력(Discriminability)**을 갖추고 있음을 확인한다.

<img width="493" height="795" alt="777777" src="https://github.com/user-attachments/assets/a2b8dcd2-6b14-4d9c-a384-786a842f3e3c" />



### 6.3 종합 평가 (Overall Evaluation)

* **성과:** 명확한 외형 특징을 가진 품종에 대해서는 거의 **100%에 가까운 신뢰도**를 보이며, 희귀 고양이 품종에서도 기대 이상의 성능을 달성했다.
* **보완:** 유사 품종 간의 세밀한 구분(Fine-grained Classification)에 존재하는 한계를 극복하기 위해, **'확률 기반 경고 메시지'**와 **'Top-3 후보 표시'** 기능을 도입하여 **사용자 경험(UX) 측면에서 완성도**를 높였다.

## 7. 프로젝트 종합 평가 (Comprehensive Evaluation)

### 7.1 잘된 점 (Successes)

**1) 모델 측면: 효율성과 성능 확보**
* **전이 학습(Transfer Learning) 적용:** 사전 학습된 **MobileNetV2**를 기반으로 하여, 상대적으로 적은 데이터셋으로도 우수한 학습 효율과 성능을 달성했다.
* **안정적인 분류:** 흔히 볼 수 있는 품종뿐만 아니라 희귀종과 유사종에서도 전반적으로 균형 잡힌 분류 성능을 확인했다.

**2) 서비스 측면: 사용자 편의성 극대화**
* **Streamlit 웹 UI 구현:** 복잡한 설치 과정 없이 웹 브라우저에서 바로 실행 가능한 직관적인 인터페이스를 제공했다.
* **간결한 워크플로우:** [사진 업로드] → [즉시 결과 확인]으로 이어지는 단순한 흐름을 통해 사용자가 쉽게 서비스를 이용할 수 있도록 설계했다.

**3) 개발 경험 측면: 풀사이클(Full-Cycle) 경험**
* **파이프라인 구축:** [데이터 전처리 → 모델 학습 → 저장 → 웹 연동]에 이르는 개발의 전 과정을 직접 구현하며 실전 감각을 익혔다.
* **실전 문제 해결:** TensorFlow/Keras 버전 호환성 문제, 가상환경(venv) 설정, Streamlit 구동 오류 등 실제 개발 현장에서 빈번한 이슈들을 직접 트러블 슈팅하며 해결 능력을 길렀다.

### 7.2 아쉬운 점 및 한계 (Limitations)

**1) 데이터셋의 다양성 부족**
* **데이터 편향:** 학술용 데이터인 'Oxford-IIIT Pet Dataset'에 의존하다 보니, 실제 보호소나 입양 환경(복잡한 배경, 조명, 각도)에서 촬영된 "야생의 데이터(Wild Data)" 특성을 충분히 반영하지 못했다.
**2) 정량적 평가 지표의 부재**
* **검증의 한계:** 라벨링이 완벽히 검증된 별도의 테스트셋(Test Set)을 확보하지 못해, **정확도(Accuracy)**나 **F1-score**와 같은 객관적인 수치로 모델 성능을 공식화하지 못한 점이 아쉬웠다.

**3) 배포 및 확장성 미비**
* **로컬 환경의 한계:** 모델 경량화(Quantization 등)나 AWS/Heroku 등을 통한 클라우드 배포까지 진행하지 못하여, 현재는 개인 PC에서만 구동 가능한 **'로컬 실험용 서비스'** 단계에 머물러있다.

## 8. 한계점 및 향후 개선 방향 (Limitations & Future Works)

### 8.1 한계점 요약 (Limitations)

* **세밀한 분류의 어려움:** 유사 품종 사이의 미세한 특징 차이를 구분하는 것은 여전히 난이도가 높다.
* **실전 검증 부족:** 조명 변화, 촬영 각도, 피사체의 부분 가림 등 극단적인 상황을 포함한 **'실전 사진(Wild Data)'**에 대한 충분한 검증이 이루어지지 않았다.
* **단일 객체 가정:** 현재 모델은 이미지 한 장에 **'동물 한 마리'**만 존재한다는 가정 하에 동작하므로, 여러 마리가 동시에 있는 사진은 처리가 어렵다.

### 8.2 향후 개선 방향 (Future Improvements)

**1) 데이터 확장 및 정량 지표 확보**
* 인터넷, 보호소, 입양 사이트 등 다양한 소스에서 이미지를 추가 수집하여, 특히 **희귀·유사 품종 위주로 데이터셋을 강화**한다.
* 라벨링이 된 별도의 테스트셋(Test Set)을 구축하여, 품종별 **정확도(Accuracy)**와 **F1-score** 등을 공식적으로 측정하고 성능을 객관화한다.

**2) Detection + Classification 구조 도입**
* **YOLO(You Only Look Once)** 등 객체 탐지 모델과 결합하여 2-Stage 구조로 확장한다.
* 한 장의 사진에서 **여러 마리의 동물을 동시에 검출(Detection)**하고, 검출된 각 개체에 대해 **품종 분류(Classification)**를 수행하도록 개선한다.

**3) 모델 경량화 및 배포**
* **ONNX, TensorRT, TensorFlow Lite** 등을 활용해 모델을 경량화하여, 저사양 PC나 모바일 환경에서도 **실시간 추론(Real-time Inference)**이 가능하도록 만든다.
* **Streamlit Cloud**나 **Hugging Face Spaces** 등에 배포하여, 설치 없이 웹 브라우저만으로 누구나 서비스를 사용할 수 있도록 접근성을 높인다.

**4) 설명 가능한 AI (Explainable AI) 적용**
* **Grad-CAM** 등의 기법을 활용하여, 모델이 이미지의 **어떤 부분(귀, 털, 무늬 등)을 보고 해당 품종으로 판단했는지 시각화**한다.
* 이를 통해 사용자의 신뢰도를 높이고, 모델이 엉뚱한 배경을 보고 판단하지 않는지 확인하는 **디버깅 도구**로 활용한다.

## 9. 개인 소감 (Personal Reflection)

이번 프로젝트를 통해 단순히 "이미지 분류 모델을 만들었다"는 결과에 그치지 않고, 다음과 같은 개발의 **전 과정을 직접 경험**해 볼 수 있었다.

* **데이터셋 선택과 전처리**
* **Transfer Learning**을 활용한 모델 설계
* **Colab–로컬 간 환경 차이**로 인한 트러블 슈팅
* **Streamlit 기반 웹 UI** 구현
* 사용자가 이해하기 쉬운 **결과 표현과 UX 설계**

특히 TensorFlow/Keras 버전 충돌, 가상환경 설정, Streamlit 실행 방식, 파일 경로 문제 등으로 여러 번 막히며 많은 시간이 소요되었다. 하지만 그 과정에서 겪은 시행착오는 **"실제 개발에서 부딪히는 문제를 스스로 해결하는 연습"**으로서 값진 경험이 되었다.

또한, 희귀 종이나 유사 종에서 모델이 제시하는 확률과 Top-3 후보를 확인하며, AI를 무조건 정답만 내놓는 시스템이 아니라 **"사람에게 후보와 단서를 제시하는 도구"**로 디자인하는 관점이 중요하다는 점을 깊이 느꼈다.


## 10. 제출 형식

📺 **시연 영상 **  

- **제출 형태:** [YouTube 링크](https://www.youtube.com/watch?v=w0NXnkMiuMI)  
- **영상 길이:** 약 1~2분  
- **영상 내용 구성:**  
  - Streamlit 앱 실행 화면 소개  
  - 반려동물(개·고양이) 이미지 업로드 과정 시연  
  - 예측 결과(품종 이름, 예측 확률, Top-3 후보) 출력 화면 확인  
  - 확률에 따라 Success / Warning / Error 메시지가 달라지는 모습  
  - Egyptian Mau, Russian Blue 등 희귀 품종 예시 테스트  
  - Staffordshire Bull Terrier 등 유사 품종에서 모델이 애매하게 예측하고  
    경고 메시지로 안내하는 장면 포함  

### 1. Miniature Pinscher (미니어처 핀셔)
> **결과:** 99.8% (가장 완벽한 정답 사례)

<p align="center">
<img width="488" height="873" alt="image" src="https://github.com/user-attachments/assets/17128cce-dadf-40a4-b69c-18e99e584c1e" />

</p>

<br>

### 2. Sphynx (스핑크스)
> **결과:** 98.8% (털 없는 특징 인식 성공 사례)

<p align="center">
<img width="473" height="777" alt="스크린샷 2025-12-15 105038" src="https://github.com/user-attachments/assets/9f128a12-4270-4a36-b948-52a6b072aaf4" />


</p>

<br>

### 3. Samoyed (사모예드)
> **결과:** 98.9% (흰색 털의 특징을 잘 잡아냄)

<p align="center">
<img width="480" height="785" alt="99999" src="https://github.com/user-attachments/assets/e727a28d-2941-4446-b840-c2ae8501d9ed" />

</p>

<br>

### 4. Soft Coated Wheaten Terrier (휘튼 테리어)
> **결과:** 98.6% (복잡한 배경에서도 인식 성공)

<p align="center">
<img width="572" height="846" alt="898988" src="https://github.com/user-attachments/assets/434101b8-2682-4274-a74c-a34e46477576" />
</p>

## 11. 참고 문헌 (References)

* **Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. (2012).**
    * *Cats and Dogs – Oxford-IIIT Pet Dataset.*

* **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).**
    * *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.

* **TensorFlow & Keras 공식 문서**
    * [https://www.tensorflow.org/](https://www.tensorflow.org/)
    * [https://keras.io/](https://keras.io/)

* **Streamlit 공식 문서**
    * [https://streamlit.io/](https://streamlit.io/)




