# pet-project
pet project

🐾 AI Pet Breed Classifier (37종)

Oxford-IIIT Pet Dataset 기반 반려동물 품종 분류 시스템
MobileNetV2 Fine-Tuning + Streamlit Web Service

1. 프로젝트 개요

이 프로젝트는 개·고양이 총 37종을 이미지로 분류하는 AI 웹 애플리케이션입니다.
사용자가 사진을 업로드하면 AI가 품종 이름 + 예측 확률 + 상위 3개 후보를 실시간으로 제공합니다.

🎯 목표

반려동물 품종을 자동으로 인식하는 실생활형 AI 서비스 개발

**희귀종, 유사종(Pit Bull / Staffordshire 등)**을 정확히 구분하도록 모델 학습

가벼운 모델(MobileNetV2)로 빠른 실시간 예측 기반 Streamlit UI 구현

🧰 기술 스택
구성 요소	기술
모델	MobileNetV2 (ImageNet pre-trained)
학습 데이터	Oxford-IIIT Pet Dataset (37종)
학습 환경	Google Colab (TensorFlow 2.16+)
웹 UI	Streamlit
프레임워크	TensorFlow / Keras
배포 방식	로컬 실행 (streamlit run app.py)
2. 프로젝트 구조
Pet_Classifier/
├─ app.py                            
├─ class_names.txt                 
├─ pet_breed_classifier_finetuned.h5  
├─ docs/                    # 결과 스크린샷 및 보고서 이미지
└─ README.md

3. 실행 방법
3.1 환경 설치
pip install streamlit tensorflow keras pillow numpy

3.2 실행
streamlit run app.py


브라우저에서 웹 인터페이스가 열리며,
이미지를 업로드하면 실시간으로 품종 예측 결과가 표시됩니다.

4. 시스템 동작 원리
4.1 전체 흐름
이미지 업로드
     ↓
224×224 리사이즈 + 정규화
     ↓
MobileNetV2 Fine-tuned 모델 예측
     ↓
Top-3 품종 및 확률 계산
     ↓
확률에 따라 적절한 메시지 출력 (성공/경고/불확실)

4.2 핵심 코드 요약

모델 로드

import keras

model = keras.models.load_model("pet_breed_classifier_finetuned.h5")

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f]


전처리

from PIL import Image, ImageOps
import numpy as np

def preprocess_image(image):
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img = np.asarray(image).astype("float32") / 255.0
    return img[np.newaxis, ...]  # (1, 224, 224, 3)


예측 & Top-3

prediction = model.predict(processed_image)
top3 = np.argsort(prediction[0])[-3:][::-1]

for rank, i in enumerate(top3, start=1):
    name = class_names[i]
    prob = prediction[0][i] * 100

🛠️ 5. Trouble Shooting & Challenges

📌 교수님이 특히 좋아하는 형식: “문제 → 원인 → 해결”

5.1 TensorFlow & Keras 버전 호환성 문제
🔍 문제

Colab(TF 2.16)에서 학습한 .h5 모델을 로컬에서 로드할 때
InputLayer mismatch, Keras attribute error 발생

⚠ 원인

Keras 3.0 업데이트 이후 .h5 포맷 호환성 변화

로컬 환경이 구버전 TensorFlow/Keras

🛠 해결

로컬 TensorFlow 및 Keras 최신 버전으로 업그레이드

모델 로드 방식을
tf.keras.models.load_model() → keras.models.load_model() 로 교체

5.2 Streamlit 실행 오류 (ScriptRunContext)
🔍 문제

VSCode Run 버튼으로 실행하면 Streamlit 앱이 구동되지 않음

missing ScriptRunContext 경고 출력

🛠 해결

Streamlit은 반드시 터미널에서 실행해야 함:

streamlit run app.py

5.3 유사 품종 오분류에 대한 UX 개선
🔍 문제

Russian Blue vs British Shorthair

American Pit Bull Terrier vs Staffordshire Bull Terrier
같이 사람도 헷갈리는 외형에서 모델 확률이 낮게 나오는 문제(40~60%)

🛠 해결

확률 기반 UX 레이어링 적용:

80% 이상 → 확신 메시지

50~80% → 경고 메시지(“약간 애매함”)

50% 미만 → 불확실 메시지 + Top-3 안내

사용자가 AI 결과를 “맹신하지 않도록” UX 구조를 명확히 함

5.4 Top-3 결과가 "1, 1, 1"로 출력되는 문제
🔍 문제

Top-3 번호가 증가하지 않고 모두 1번으로 표시됨.

🛠 해결
for rank, i in enumerate(top3, start=1):

5.5 이미지 출력 비율 깨짐
🔍 문제

이미지가 Streamlit 화면에서 너무 크게 보이거나 비율이 깨짐

🛠 해결
st.image(image, use_container_width=True)

5.6 모델 파일 및 class_names.txt 경로 오류
🔍 문제

Streamlit 실행 시 파일을 찾지 못해 FileNotFoundError 발생

🛠 해결

app.py, class_names.txt, .h5를 반드시 같은 폴더에 배치

디버깅용 현재 경로 출력:

import os
print(os.getcwd())

5.7 모델 파일 용량이 커서 GitHub에 업로드 불가
🔍 문제

150MB 이상의 .h5 파일은 GitHub 일반 push 불가

🛠 해결

.zip으로 압축하여 업로드

또는 다운로드 링크 제공

README에 “압축 해제 후 app.py와 함께 두세요”로 안내

5.8 Pillow(PIL) Warning 출력 문제
🔍 문제

Pillow 이미지 로딩 시 경고 메시지가 너무 많이 뜸

🛠 해결
import warnings
warnings.filterwarnings("ignore")

📊 6. 결과 분석 (대표 7종 기반)

아래는 네가 보내준 37종 결과 중 대표 성능을 보여주는 품종들이다.

6.1 💚 확신도 매우 높은 품종
✔ Basset Hound

예측 확률: 99%+

귀·머즐 등의 특징이 뚜렷해 모델이 매우 안정적으로 분류

“명확한 외형 → 높은 정확도” 패턴 확인

6.2 🔥 희귀 고양이 품종 성능
✔ Egyptian Mau

예측 확률: 96~99%

점무늬 패턴을 정확히 인식

희귀종임에도 매우 높은 정확도

✔ Russian Blue

예측 확률: 93% 이상

British Shorthair와 혼동 가능성이 있지만 대부분 정확히 분류

회색 단모·얼굴 비율 등 외형 특징 잘 학습

6.3 ⚠ 혼동이 발생한 품종 (Hard Cases)
⚠ Pit Bull / Staffordshire Bull Terrier

외형이 실제로도 거의 동일

예측 확률이 55% vs 40% 등 매우 비슷하게 나옴

→ 모델 한계와 데이터셋 특성이 동시에 드러나는 사례

6.4 🐶 기타 안정적으로 분류된 품종

Yorkshire Terrier

Scottish Terrier

Samoyed

장모/단모·크기·형태(ears/legs) 등을 잘 파악하여 안정적으로 분류됨.

📌 7. 프로젝트 종합 평가
✅ 강점

37종에 대한 전반적으로 높은 분류 성능

희귀종에서도 강한 퍼포먼스(Egyptian Mau, Russian Blue)

유사 품종 혼동 시 사용자 경고 UX 제공

Streamlit 기반의 실시간 AI 서비스 형태로 구현됨

데이터 전처리 → 학습 → 모델 저장 → 웹 UI 연결까지 완전한 AI 파이프라인 구축

⚠ 약점

유사 종 구분은 현재 MobileNetV2 수준에서는 한계 존재

조명·각도·배경 변화가 크면 확률이 떨어지는 경향

일부 품종은 Oxford-IIIT Pet Dataset의 데이터양 자체가 부족

🚀 8. 아쉬웠던 점 & 향후 개선
😿 아쉬웠던 점

학습 데이터가 많지 않아 희귀종에서 편차가 존재

웹 배포(Streamlit Cloud)까지 진행하지 못함

모델 경량화(ONNX, TensorRT)를 하지 못함

🚀 향후 개선

ONNX 변환으로 모바일/웹 실시간 추론 속도 개선

YOLO + Breed Classifier 2-stage 시스템으로 확장

품종 분류 → 나이/건강상태 추정 등 멀티태스크 모델 확대

희귀종 전용 추가 데이터셋 수집 후 성능 향상

🎉 9. 마무리

본 프로젝트는 반려동물 37종 품종 분류 AI를 개발하며
데이터 전처리 → 모델 학습 → 추론 → 웹UI 구현까지
AI 서비스 전체 파이프라인을 직접 실습한 프로젝트입니다.

높은 품종 분류 성능

희귀 종에서도 일관된 퍼포먼스

UX 관점에서의 확률 기반 메시지 처리

Streamlit 웹앱으로 실제 서비스 수준 구현

단순 코드 작성이 아니라
실제로 “사용 가능한 AI 제품”을 만드는 전체 과정을 경험했습니다.
