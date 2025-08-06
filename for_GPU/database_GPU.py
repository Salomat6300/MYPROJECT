# deepGPUFolder/database.py
import psycopg2
from psycopg2 import sql
import io
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class FaceDB:
    def __init__(self, dbname, user, password, host, port, max_workers=4):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _connect(self):
        return psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )

    def jadvallarni_yaratish(self):
        conn = None
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_data (
                    id SERIAL PRIMARY KEY,
                    img BYTEA NOT NULL,
                    encoding FLOAT8[] NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_log_data (
                    id SERIAL PRIMARY KEY,
                    id_name INTEGER REFERENCES face_data(id),
                    kirish_vaqti TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            print("Jadvallar yaratildi yoki mavjud.")
            return True
        except Exception as e:
            print(f"Jadvallarni yaratishda xatolik: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def yuz_qoshish(self, yuz_rasmi_bgr, yuz_kodi_np):
        # Asinxron DB yozish uchun executor ishlatiladi
        return self.executor.submit(self._yuz_qoshish_sync, yuz_rasmi_bgr, yuz_kodi_np)

    def _yuz_qoshish_sync(self, yuz_rasmi_bgr, yuz_kodi_np):
        conn = None
        try:
            conn = self._connect()
            cur = conn.cursor()
            _, buf = cv2.imencode('.jpg', yuz_rasmi_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            img_bytes = buf.tobytes()
            kod_list = yuz_kodi_np.astype(float).tolist()
            cur.execute(
                sql.SQL("INSERT INTO face_data (img, encoding) VALUES (%s, %s) RETURNING id"),
                (psycopg2.Binary(img_bytes), kod_list)
            )
            yuz_id = cur.fetchone()[0]
            cur.execute("INSERT INTO face_log_data (id_name) VALUES (%s)", (yuz_id,))
            conn.commit()
            return yuz_id
        except Exception as e:
            print(f"Yuz qoshishda xatolik: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def barcha_yuzlarni_olish(self):
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT id, encoding FROM face_data")
            rows = cur.fetchall()
            ids = []
            encs = []
            for r in rows:
                ids.append(r[0])
                encs.append(np.array(r[1], dtype=np.float32))
            return ids, encs
        except Exception as e:
            print(f"DBdan olish xatolik: {e}")
            return [], []
        finally:
            try:
                conn.close()
            except:
                pass

    def kirishni_loglash(self, yuz_id):
        # Asinxron log yozish
        return self.executor.submit(self._kirishni_loglash_sync, yuz_id)

    def _kirishni_loglash_sync(self, yuz_id):
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("INSERT INTO face_log_data (id_name) VALUES (%s)", (yuz_id,))
            conn.commit()
            return True
        except Exception as e:
            print(f"Log yozishda xatolik: {e}")
            return False
        finally:
            try:
                conn.close()
            except:
                pass
