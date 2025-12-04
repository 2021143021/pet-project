# 🐾 AI Pet Breed Classifier (37종)

Oxford-IIIT Pet Dataset 기반 반려동물(개·고양이) 품종 분류 시스템  
**MobileNetV2 Fine-Tuning + Streamlit Web Application**

---

## 0. 목차

1. [프로젝트 개요](#1-프로젝트-개요)  
2. [데이터셋 및 클래스 구성](#2-데이터셋-및-클래스-구성)  
3. [모델 구조 및 학습 방법](#3-모델-구조-및-학습-방법)  
4. [웹 애플리케이션 구조 및 동작](#4-웹-애플리케이션-구조-및-동작)  
5. [Trouble Shooting & Challenges](#5-trouble-shooting--challenges)  
6. [모델 성능 및 결과 분석](#6-모델-성능-및-결과-분석)  
7. [프로젝트 종합 평가](#7-프로젝트-종합-평가)  
8. [한계점 및 향후 개선 방향](#8-한계점-및-향후-개선-방향)  
9. [프로젝트 구조 및 실행 방법](#9-프로젝트-구조-및-실행-방법)  

---

## 1. 프로젝트 개요

이 프로젝트는 **개·고양이 37종을 하나의 이미지로 자동 분류하는 AI 웹 서비스**이다.  
사용자가 사진을 업로드하면, 모델이 다음 정보를 실시간으로 반환한다.

- 예측 품종 이름  
- 예측 확률(%)  
- 상위 3개 후보 품종  
- 확률 수준에 따른 안내 메시지(Success / Warning / Error)

### 🎯 프로젝트 목표

- 반려동물 품종을 자동으로 인식하는 **실생활형 AI 서비스 구축**
- 사람이 구분하기 어려운 **희귀종·유사종(Pit Bull / Staffordshire Bull Terrier 등)** 에서도 의미 있는 결과 도출
- 학습된 딥러닝 모델을 **Streamlit 웹 애플리케이션**으로 연결해, 실시간 추론 환경 제공
- 데이터 준비 → 모델 학습 → 저장 → 배포까지 **엔드 투 엔드(End-to-End) AI 파이프라인 경험**

---

## 2. 데이터셋 및 클래스 구성

### 2.1 사용 데이터셋

- **Oxford-IIIT Pet Dataset**
  - 개와 고양이 품종으로 구성된 이미지 데이터셋
  - 총 **37개 품종** (개 25종, 고양이 12종)
  - 각 품종별 여러 장의 이미지(다양한 배경·각도·조명 포함)

> 이 프로젝트에서는 Kaggle/공식 배포본을 기반으로 전처리 후,  
> **학습/검증 세트로 분리 및 MobileNetV2 입력 형식(224×224 RGB)으로 변환**하였다.

### 2.2 클래스 예시

- 개 품종 예시  
  - Basset Hound, Boxer, Yorkshire Terrier, Scottish Terrier, Samoyed,  
    Staffordshire Bull Terrier, Wheaten Terrier 등
- 고양이 품종 예시  
  - Egyptian Mau, Russian Blue, Abyssinian, Bengal 등

클래스 이름은 `class_names.txt` 파일로 관리하며,  
모델의 출력 인덱스를 다시 사람 읽을 수 있는 품종 이름으로 매핑한다.

---

## 3. 모델 구조 및 학습 방법

### 3.1 기본 아이디어

- **MobileNetV2 (ImageNet 사전 학습)** 을 기반으로 한 **Transfer Learning**
- 마지막 분류기 부분을 37클래스에 맞게 교체 후 Fine-Tuning

### 3.2 모델 구조(개략)

- Base Model: `MobileNetV2(weights="imagenet", include_top=False)`
- Global Average Pooling
- Fully Connected Layer (ReLU)
- Dropout (과적합 완화)
- Output Layer: `Dense(37, activation="softmax")`

### 3.3 학습 전략

- 입력 크기: **224 × 224 × 3**
- 전처리:  
  - Resize + Center Crop → 224×224  
  - 0~1 정규화(`/255.0`)  
- 손실 함수: Categorical Cross-Entropy
- 옵티마이저: Adam (learning rate는 코드 내에서 설정)
- 학습 환경: Google Colab (TensorFlow 2.16 기준)
- 학습 완료 후, 모델을 **`.h5` 포맷으로 저장** → 로컬로 다운로드 후 Streamlit 앱에서 사용

> 세부 하이퍼파라미터(에폭 수, 배치 크기 등)는 학습 코드에 명시되어 있으며,  
> 프로젝트의 핵심은 **사전학습 모델 기반 Fine-Tuning + 웹 서비스 연결**이다.

---

## 4. 웹 애플리케이션 구조 및 동작

### 4.1 전체 동작 흐름

```text
이미지 업로드
      ↓
이미지 전처리 (224×224 리사이즈 + 정규화)
      ↓
Fine-Tuned MobileNetV2 모델로 추론
      ↓
가장 높은 확률의 품종 + Top-3 후보 계산
      ↓
확률 구간에 따라 다른 메시지(Success / Warning / Error) 출력
