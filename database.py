import psycopg2
from psycopg2 import sql
import cv2
import io
import numpy as np

class FaceDB:
    def __init__(self, dbname, user, password, host, port):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connection = None

    def ulanish(self):
        try:
            self.connection = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            return True
        except Exception as e:
            print(f"Bazaga ulanib bolmadi: {e}")
            return False

    def ulanishni_yopish(self):
        if self.connection:
            self.connection.close()

    def jadvallarni_yaratish(self):
        if not self.ulanish():
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_data (
                    id SERIAL PRIMARY KEY,
                    img BYTEA NOT NULL,
                    encoding FLOAT[] NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_log_data (
                    id SERIAL PRIMARY KEY,
                    id_name INTEGER REFERENCES face_data(id),
                    kirish_vaqti TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.connection.commit()
            print("Jadvallar yaratildi.")
            return True
        except Exception as e:
            print(f"Xatolik: {e}")
            return False
        finally:
            self.ulanishni_yopish()

    def yuz_qoshish(self, yuz_rasmi, yuz_kodi):
        if not self.ulanish():
            return None
        try:
            cursor = self.connection.cursor()
            _, buffer = cv2.imencode(".jpg", yuz_rasmi)
            io_buf = io.BytesIO(buffer)
            kod_royxati = yuz_kodi.tolist()
            cursor.execute(
                sql.SQL("INSERT INTO face_data (img, encoding) VALUES (%s, %s) RETURNING id"),
                (io_buf.read(), kod_royxati)
            )
            yuz_id = cursor.fetchone()[0]
            cursor.execute(
                sql.SQL("INSERT INTO face_log_data (id_name) VALUES (%s)"),
                (yuz_id,)
            )
            self.connection.commit()
            return yuz_id
        except Exception as e:
            print(f"Yuz qoshishda xatolik: {e}")
            return None
        finally:
            self.ulanishni_yopish()

    def barcha_yuzlarni_olish(self):
        if not self.ulanish():
            return [], []
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT id, encoding FROM face_data")
            qatorlar = cursor.fetchall()
            yuz_ids = []
            yuz_kodlari = []
            for qator in qatorlar:
                yuz_ids.append(qator[0])
                yuz_kodlari.append(np.array(qator[1]))
            return yuz_ids, yuz_kodlari
        except Exception as e:
            print(f"Xatolik: {e}")
            return [], []
        finally:
            self.ulanishni_yopish()

    def kirishni_loglash(self, yuz_id):
        if not self.ulanish():
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                sql.SQL("INSERT INTO face_log_data (id_name) VALUES (%s)"),
                (yuz_id,)
            )
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Log yozishda xatolik: {e}")
            return False
        finally:
            self.ulanishni_yopish()
