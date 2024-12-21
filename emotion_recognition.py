import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import os
st.set_page_config(layout='wide',page_title="Emotion Recognition")

emotions = ['angry','fear','happy','neutral','sad','surprise']
emotions_img = ['angry.png','fear.png','happy.png','neutral.png','sad.png','surprise.png']

st.title("I am Emotion Recognizer")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict(img):
    img1 = cv2.resize(img, (48, 48))
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    final_img = gray_img.reshape(-1, 48, 48, 1)
    pred_prob = model.predict(final_img)
    pred = np.argmax(pred_prob)
    return pred,pred_prob[0]

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(48, 48))
    pred_prob = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    pred = 3
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        face = vid[y:y + h, x:x + w]
        pred,pred_prob = predict(face)
        cv2.putText(vid,emotions[pred],(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    return pred_prob,pred

col1,col2 = st.columns(2,gap="small")
with col1:
    start_button_pressed = st.button("Start")
with col2:
    stop_button_pressed = st.button("Stop")
if start_button_pressed:
    model = tf.keras.models.load_model('best_CNN_model.keras')
    cap = cv2.VideoCapture(0)
    col1,col2,col3 = st.columns([1.5,2,1.5],gap="small")
    with col2:
        frame_placeholder = st.empty()
    with col3:
        row1 = st.empty()
    with col1:
        row2 = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        prob_arr,pred_index = detect_bounding_box(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame,channels="RGB")
        source = pd.DataFrame(
                    {
                        "emotion": emotions,
                        "probability": prob_arr,
                    }
                    )
        row1.bar_chart(source, x="emotion", y="probability", horizontal=True,height=300,width=300)
        row2.image(os.path.join('emotions_img',emotions_img[pred_index]))
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    st.write("Press Start and let me guess what you feeling !!!")

