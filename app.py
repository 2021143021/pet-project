import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import keras  # 최신 버전 호환성을 위해 추가

# ---
# 1. 모델과 클래스 이름 로드
@st.cache_resource
def load_my_model():
    # 최신 TensorFlow 호환성을 위해 keras.models 사용
    # 파일 이름이 'pet_breed_classifier_finetuned.h5'인지 꼭 확인하세요!
    model = keras.models.load_model('pet_breed_classifier_finetuned.h5')
    print("모델 로드 완료.")
    return model


@st.cache_data
def load_class_names():
    # 여기가 아까 오류났던 부분입니다. 깔끔하게 수정했습니다!
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f]
    print("클래스 이름 로드 완료.")
    return class_names


# 예외 처리: 파일이 없으면 에러 메시지 표시
try:
    model = load_my_model()
    class_names = load_class_names()
except Exception as e:
    st.error(f"파일 로드 오류: {e}")
    st.error("'pet_breed_classifier_finetuned.h5' 파일과 'class_names.txt' 파일이 같은 폴더에 있는지 확인해주세요.")
    st.stop()


# ---
# 2. 이미지 전처리 함수
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # 0~1 사이로 정규화
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    # 차원 확장 (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return data


# ---
# 3. 품종 이름 포맷팅 함수
def format_breed_name(breed_name):
    return ' '.join([word.capitalize() for word in breed_name.split('_')])


# ---
# 4. 화면 구성 (기본 레이아웃)
st.title("🐾 AI 반려동물 품종 분류기 (37종)")
st.write("AI가 사진 속 동물의 품종을 맞혀 드립니다!")
st.write("(Oxford-IIIT Pet Dataset 기반, MobileNetV2 미세 조정 학습)")

# 파일 업로드 버튼
uploaded_file = st.file_uploader("이미지 파일을 선택하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 표시 (use_container_width 적용됨)
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드된 이미지', use_container_width=True)
    st.write("")

    # 예측 수행
    with st.spinner('AI가 품종을 분석 중입니다...'):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        class_index = np.argmax(prediction)
        probability = np.max(prediction)
        breed_name = class_names[class_index]

    # 결과 표시
    st.subheader("🤖 AI 분석 결과")

    formatted_name = format_breed_name(breed_name)
    percentage = probability * 100

    if probability > 0.5:
        st.success(f"이 동물은 **{percentage:.2f}%** 확률로 **{formatted_name}** 입니다!")
    elif probability > 0.2:
        st.warning(f"**{formatted_name}**일 확률이 **{percentage:.2f}%**로 가장 높지만, AI도 확신하지 못하고 있습니다.")
    else:
        st.error(f"AI가 이 이미지를 판별하기 어렵습니다. (가장 높은 확률: {formatted_name}, {percentage:.2f}%)")

    # 상위 3개 예측 결과 보여주기
    st.write("---")
    st.write("상위 3개 예측 결과:")
    top_3_indices = np.argsort(prediction[0])[-3:][::-1]
    for i in top_3_indices:
        name = format_breed_name(class_names[i])
        prob = prediction[0][i] * 100
        st.write(f"1. **{name}**: {prob:.2f}%")

else:
    st.info("먼저 반려동물 이미지를 업로드해주세요.")
