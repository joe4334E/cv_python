import cv2
import numpy as np
import time
import pygame
import os

# Rutas a los clasificadores y al archivo de sonido
face_cascade_path = os.path.join('..', 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join('..', 'haarcascade_eye.xml')
alarm_sound_path = os.path.join('..', 'alarm', 'alarm.wav')

# Inicializar clasificadores
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Inicializar pygame mixer para el sonido
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(alarm_sound_path)

cap = cv2.VideoCapture(0)

blink_count = 0
blink_history = []
is_blinking = False
start_time = 0
alarm_triggered = False

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        if len(eyes) == 0:
            # Eyes are closed
            if not is_blinking:
                start_time = time.time()
                is_blinking = True
        else:
            # Eyes are open
            is_blinking = False

        # Check if the eyes have been closed for 3 seconds
        if is_blinking and time.time() - start_time > 3 and not alarm_triggered:
            print("Eyes closed for 3 seconds. Triggering alarm.")
            alarm_sound.play()
            alarm_triggered = True
            blink_count += 1
            blink_history.append(time.strftime("%Y-%m-%d %H:%M:%S"))

        # Reset alarm_triggered if the eyes are open
        if not is_blinking:
            alarm_triggered = False

    # Display blink count and history on the frame
    cv2.putText(frame, f"Blink Count: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Blink History: {', '.join(blink_history)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

