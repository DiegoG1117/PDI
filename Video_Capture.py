import cv2

cap = cv2.VideoCapture(0)  # Cambiado para abrir la cámara del computador

while cap.isOpened():
    ret, frame = cap.read()

    frame_borroso = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("frame", frame)
    #cv2.imshow("frame_borroso", frame_borroso)

    key = cv2.waitKey(1)
    if key == 27 or not ret:
        break

cap.release()
cv2.destroyAllWindows()