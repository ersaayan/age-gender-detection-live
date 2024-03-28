from deepface import DeepFace
import cv2

# Haar Cascade yüz dedektörünü yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video yakalama nesnesi oluştur
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Görüntüyü yakala
    if not ret:
        break

    # Gri tonlamaya çevir (yüz algılama için)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri algıla
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Algılanan her yüz için
    for (x, y, w, h) in faces:
        # Yüzü çerçevele
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Yüzü DeepFace analizi için kes
        face = frame[y:y+h, x:x+w]

        try:
            # DeepFace ile yüz analizi yap
            analysis = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)
            
            if 'age' in analysis and 'gender' in analysis:
                # Sonuçları ekranda göster
                cv2.putText(frame, f"Age: {int(analysis['age'])}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Gender: {analysis['gender']}", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                print("DeepFace analizi beklenen sonuçları döndürmedi.")
        
        except Exception as e:
            print(f"Yüz analizinde hata: {e}")

    # Sonuçları göster
    cv2.imshow('Frame', frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Her şey bittiğinde kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
