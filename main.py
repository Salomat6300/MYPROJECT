from deepGPUFolder.database import FaceDB
from deepGPUFolder.face_recognizer import YuzTanibOlovchi

if __name__ == "__main__":
    DB_NAME = "face_db"
    DB_USER = "postgres"
    DB_PASSWORD = "123"
    DB_HOST = "localhost"
    DB_PORT = "5432"

    face_db = FaceDB(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)

    if face_db.jadvallarni_yaratish():
        tanib_oluvchi = YuzTanibOlovchi(face_db)
        if tanib_oluvchi.ishga_tushirish():
            tanib_oluvchi.yuzlarni_tanib_olish()
            tanib_oluvchi.tozalash()
    else:
        print("Jadvallar yaratilmadi. Dastur toxtatildi.")
