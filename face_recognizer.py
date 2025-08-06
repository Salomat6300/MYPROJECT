import cv2
import face_recognition
import numpy as np
from datetime import datetime
from deepGPUFolder.face_orientation import FaceOrientationDetector

class YuzTanibOlovchi:
    def __init__(self, face_db):
        self.face_db = face_db
        self.video_kamera = None
        self.malum_yuz_ids = []
        self.malum_yuz_kodlari = []
        self.orient_detector = FaceOrientationDetector()

    def ishga_tushirish(self):
        self.malum_yuz_ids, self.malum_yuz_kodlari = self.face_db.barcha_yuzlarni_olish()
        self.video_kamera = cv2.VideoCapture(0)
        if not self.video_kamera.isOpened():
            print("Kamerani ochib bolmadi!")
            return False
        return True

    def get_main_face(self, face_locations, frame_shape):
        """Diqqat markazidagi asosiy yuzni aniqlaydi"""
        if not face_locations:
            return None
            
        height, width = frame_shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Eng yaqin yuzni topish (markazga eng yaqin)
        min_distance = float('inf')
        main_face = None
        
        for (top, right, bottom, left) in face_locations:
            face_center_x = (left + right) // 2
            face_center_y = (top + bottom) // 2
            
            distance = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                main_face = (top, right, bottom, left)
        
        return main_face

    def yuzlarni_tanib_olish(self):
        oxirgi_log_vaqtlari = {}
        while True:
            joriy_vaqt = datetime.now()
            muvaffaqiyatli, kadr = self.video_kamera.read()
            kadr = cv2.flip(kadr, 1)
            if not muvaffaqiyatli:
                break

            rgb_kadr = cv2.cvtColor(kadr, cv2.COLOR_BGR2RGB)
            try:
                yuz_joylari = face_recognition.face_locations(rgb_kadr)
                
                # Faqat diqqat markazidagi asosiy yuzni olish
                main_face = self.get_main_face(yuz_joylari, kadr.shape)
                if not main_face:
                    cv2.imshow("Facial recognition", kadr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Faqat asosiy yuz uchun kodlarni olish
                yuz_kodlari = face_recognition.face_encodings(
                    rgb_kadr, 
                    known_face_locations=[main_face]
                )
                
                if not yuz_kodlari:
                    cv2.imshow("Facial recognition", kadr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                yuz_kodi = yuz_kodlari[0]
                yuqori, ong, past, chap = main_face
                sub_frame = kadr[yuqori:past, chap:ong]
                orientation = self.orient_detector.detect(sub_frame)
                
                if not orientation:
                    ism = "Yuz holati notogri"
                else:
                    solishtirish = face_recognition.compare_faces(
                        self.malum_yuz_kodlari, yuz_kodi, tolerance=0.6
                    )

                    ism = "Nomalum shaxs!"
                    yuz_id = None

                    if True in solishtirish:
                        idx = solishtirish.index(True)
                        yuz_id = self.malum_yuz_ids[idx]
                        ism = f"ID-{yuz_id}"

                        log_kerak = False
                        if yuz_id not in oxirgi_log_vaqtlari:
                            log_kerak = True
                        else:
                            vaqt_farqi = (joriy_vaqt - oxirgi_log_vaqtlari[yuz_id]).total_seconds()
                            if vaqt_farqi >= 30:
                                log_kerak = True

                        if log_kerak:
                            self.face_db.kirishni_loglash(yuz_id)
                            oxirgi_log_vaqtlari[yuz_id] = joriy_vaqt

                    else:
                        yuz_rasmi = kadr[yuqori:past, chap:ong]
                        yuz_id = self.face_db.yuz_qoshish(yuz_rasmi, yuz_kodi)
                        if yuz_id:
                            ism = f"ID-{yuz_id}"
                            self.malum_yuz_ids.append(yuz_id)
                            self.malum_yuz_kodlari.append(yuz_kodi)
                            oxirgi_log_vaqtlari[yuz_id] = joriy_vaqt

                cv2.rectangle(kadr, (chap, yuqori), (ong, past), (0, 255, 0), 2)
                cv2.putText(
                    kadr, ism, (chap + 6, past - 6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

                cv2.imshow("Facial recognition", kadr)

            except Exception as e:
                print(f"Xatolik: {e}")
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def tozalash(self):
        if self.video_kamera:
            self.video_kamera.release()
        cv2.destroyAllWindows()