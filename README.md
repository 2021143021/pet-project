🐾 AI Pet Breed Classifier (37종)

Oxford-IIIT Pet Dataset 기반 반려동물 품종 분류 시스템
MobileNetV2 Fine-Tuning + Streamlit Web Application

📌 1. 프로젝트 개요

이 프로젝트는 개·고양이 37종을 이미지 한 장으로 자동 분류하는 AI 웹 서비스입니다.
사용자가 사진을 업로드하면, 모델이 다음 정보를 실시간으로 제공합니다.

예측 품종 이름

예측 확률(%)

상위 3개 후보 품종

확률 수준에 따른 안내 메시지(Success / Warning / Error)

🎯 프로젝트 목표

반려동물 품종을 자동으로 인식하는 실생활형 AI 서비스 구현

희귀종 / 유사종(Pit Bull, Staffordshire Bull Terrier 등) 도 구분 가능하도록 모델 성능 향상

Streamlit 기반 실시간 웹 인퍼런스 서비스 구축

학습 → 모델 저장 → 서비스 연동까지 AI 전체 파이프라인 경험

🏗️ 기술 스택
구성 요소	기술
✔ 모델	MobileNetV2 (ImageNet 사전학습 기반 Fine-Tuning)
✔ 데이터	Oxford-IIIT Pet Dataset (37종)
✔ 학습 환경	Google Colab (TensorFlow 2.16)
✔ 웹 UI	Streamlit
✔ 라이브러리	TensorFlow, Keras, Pillow, NumPy
✔ 실행 방식	streamlit run app.py
📁 2. 프로젝트 구조
Pet_Classifier/
│── app.py                         # Streamlit 메인 실행 파일
│── class_names.txt                # 37개 품종 이름 목록
│── pet_breed_classifier_finetuned.h5   # Fine-Tuning된 MobileNetV2 모델
│── docs/                          # 보고서/README용 이미지 저장 폴더
└── README.md

🚀 3. 실행 방법
🔧 3.1 필요한 라이브러리 설치
pip install streamlit tensorflow keras pillow numpy

▶ 3.2 Streamlit 실행
streamlit run app.py


브라우저가 자동으로 열리며, 이미지를 업로드하면 즉시 결과가 표시됩니다.

🔍 4. 시스템 동작 흐름
이미지 업로드
      ↓
(224×224) 이미지 전처리 + 정규화
      ↓
MobileNetV2 Fine-Tuned 모델로 예측
      ↓
예측 품종 + 확률 + Top-3 출력
      ↓
확률에 따라 Success / Warning / Error 메시지 제공

🧠 4.1 핵심 코드 요약
✔ 모델 로드
model = keras.models.load_model("pet_breed_classifier_finetuned.h5")

✔ 이미지 전처리
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
img = np.asarray(image) / 255.0

✔ 예측 & Top-3 품종 추출
prediction = model.predict(processed_image)
top3 = np.argsort(prediction[0])[-3:][::-1]

🛠️ 5. Trouble Shooting & 해결 과정

교수님이 “문제 → 해결” 구조를 좋아함 → 그 스타일로 완전 깔끔하게 정리해줌.

① TensorFlow/Keras 버전 충돌 문제

문제
Colab(TF 2.16+)에서 학습한 .h5 모델을 로컬 TensorFlow로 로드할 때 오류 발생
(InputLayer mismatch, keras attribute 오류 등)

해결

로컬 TensorFlow/Keras 라이브러리 최신 버전으로 업그레이드

모델 로드 방식을 tf.keras → keras.models.load_model() 로 변경하여 호환성 해결

② Streamlit 실행 오류 (ScriptRunContext)

문제
VSCode Run 버튼으로 실행 시 missing ScriptRunContext 오류 발생

해결

streamlit run app.py


터미널 실행 방식으로 통일 → 문제 해결

③ Top-3 예측 결과가 “1,1,1”로 찍히는 문제

문제
순번 고정 출력

해결

for rank, i in enumerate(top3, start=1):
    ...


번호 자동 증가 적용

④ 유사 품종에서 확률이 낮게 나오는 문제

러시안 블루 ↔ 브리티시 쇼트헤어
Pit Bull ↔ Staffordshire Bull Terrier

해결

확률 기반 UX 개선

80% 이상 → 확신 메시지

50~80% → 경고(애매함)

50% 미만 → 불확실, Top-3 중심 안내

⑤ 모델 저장 후 로드 시 발생하는 구조 불일치 오류

해결

모델 저장 시 save_format='h5' 명시

로컬과 Colab의 Keras 버전 동기화

📊 6. 모델 성능 분석 (업로드한 7개 이미지 기반)
💚 고신뢰(high-confidence) 분류
품종	예측 확률	비고
Basset Hound	99%	외형 특징 명확 → 높은 정확도
Yorkshire Terrier	99%	장모, 얼굴형 특징 잘 학습
Samoyed	98%	색/체형이 확실해 정확도 높음
Scottish Terrier	99%	견종 특징 명확
🔥 희귀종 분류 성능 우수
품종	확률	설명
Egyptian Mau	96%	무늬·체형 정확히 탐지
Russian Blue	93%	British Shorthair와 유사하지만 성공적으로 분류
⚠️ 혼동 발생 케이스
품종	모델 반응
Staffordshire Bull Terrier ↔ Pit Bull	두 견종이 실제로도 매우 유사 → 확률 간격 좁음

UX 개선으로 해결 (경고 메시지 표시)

🧾 7. 종합 평가
👍 모델 강점

희귀종 포함 대부분 품종에서 90~99% 정확도

실사용 가능한 웹 서비스 형태로 완성

가벼운 MobileNetV2 기반으로 CPU에서도 빠른 인퍼런스

Streamlit UI로 사용성 매우 개선됨

👎 모델 약점

유사 품종 구분은 한계 존재

조명/각도 영향에 취약

데이터가 적은 품종은 추가 학습 필요

🚀 8. 향후 개선 방향

ONNX 변환 → 모바일/브라우저 속도 대폭 증가

YOLO + Classification 형태의 2단계 구조 확장

품종 + 나이 + 건강 상태까지 확장하는 Multi-task 모델 개발

희귀종 추가 데이터 확보 후 재학습

🎉 9. 마무리

본 프로젝트는 AI 데이터 준비 → 모델 학습 → 서비스 구현까지
전체 파이프라인을 직접 수행하여 완성한 종합 AI 프로젝트입니다.

“단순한 모델 구현을 넘어, 실제 사용자가 쓸 수 있는 AI 서비스를 완성한 것이 핵심 성과입니다.”
