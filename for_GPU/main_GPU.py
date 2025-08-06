# deepGPUFolder/main.py
from MYPROJECT.for_GPU.database_GPU import FaceDB
from MYPROJECT.for_GPU.face_recognizer_GPU import YuzTanibOlovchi

if __name__ == "__main__":
    DB_NAME = "face_db"
    DB_USER = "postgres"
    DB_PASSWORD = "123"
    DB_HOST = "localhost"
    DB_PORT = "5432"

    face_db = FaceDB(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)
    if not face_db.jadvallarni_yaratish():
        print("Jadvallar yaratilmadi. To'xtatildi.")
        exit(1)

    recognizer = YuzTanibOlovchi(face_db)
    if recognizer.ishga_tushirish():
        recognizer.yuzlarni_tanib_olish()
    else:
        print("Kamera ishga tushmadi.")
