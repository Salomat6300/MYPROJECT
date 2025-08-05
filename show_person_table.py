import cv2
import numpy as np
import psycopg2
from psycopg2 import sql
from datetime import datetime
import io
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

class FaceDBViewer:
    def __init__(self, dbname, user, password, host, port):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connection = None
        
        # GUI oynasini yaratish
        self.root = tk.Tk()
        self.root.title("Face Recognition System")
        self.root.geometry("1200x800")
        
        # Asosiy frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Rasm ko'rsatish uchun frame
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Tanlangan shaxs")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Rasm label
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(expand=True)
        
        # Ma'lumotlar jadvali
        self.table_frame = ttk.LabelFrame(self.main_frame, text="Barcha shaxslar")
        self.table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview (jadval) yaratish
        self.tree = ttk.Treeview(self.table_frame, columns=("ID", "Created At", "Last Entry"), show="headings")
        self.tree.heading("ID", text="ID")
        self.tree.heading("Created At", text="Yaratilgan vaqti")
        self.tree.heading("Last Entry", text="Oxirgi kirish")
        self.tree.column("ID", width=100, anchor=tk.CENTER)
        self.tree.column("Created At", width=200, anchor=tk.CENTER)
        self.tree.column("Last Entry", width=200, anchor=tk.CENTER)
        
        # Scrollbar qo'shish
        scrollbar = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Jadvaldagi elementga bosilganda
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        
        # Tugmalar uchun frame (Qo'shildi)
        self.buttons_frame = ttk.Frame(self.main_frame)
        self.buttons_frame.pack(pady=5)
        
        # Yangi ma'lumot qo'shish tugmasi
        self.add_button = ttk.Button(self.buttons_frame, text="Yangi shaxs qo'shish", command=self.add_new_face)
        self.add_button.pack(side=tk.LEFT, padx=5)
        
        # Yangilash tugmasi (Qo'shildi)
        self.refresh_button = ttk.Button(self.buttons_frame, text="Yangilash", command=self.refresh_data)
        self.refresh_button.pack(side=tk.LEFT, padx=5)
        
        # Ma'lumotlarni yangilash
        self.refresh_data()
    
    def connect_db(self):
        """Bazaga ulanish"""
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
            messagebox.showerror("Xatolik", f"Bazaga ulanib bo'lmadi: {e}")
            return False
    
    def close_connection(self):
        """Ulanishni yopish"""
        if self.connection:
            self.connection.close()
    
    def refresh_data(self):
        """Ma'lumotlarni yangilash"""
        if not self.connect_db():
            return
        
        try:
            cursor = self.connection.cursor()
            
            # Jadvalni tozalash
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # face_data va face_log_data dan ma'lumotlarni olish
            query = """
            SELECT fd.id, fd.created_at, MAX(fl.kirish_vaqti) as last_entry
            FROM face_data fd
            LEFT JOIN face_log_data fl ON fd.id = fl.id_name
            GROUP BY fd.id, fd.created_at
            ORDER BY fd.id DESC
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Jadvalga ma'lumotlarni qo'shish
            for row in rows:
                self.tree.insert("", tk.END, values=row)
            
        except Exception as e:
            messagebox.showerror("Xatolik", f"Ma'lumotlarni olishda xatolik: {e}")
        finally:
            self.close_connection()
    
    def on_select(self, event):
        """Jadvaldagi element tanlanganda"""
        selected_item = self.tree.focus()
        if not selected_item:
            return
        
        item_data = self.tree.item(selected_item)
        face_id = item_data['values'][0]
        
        if not self.connect_db():
            return
        
        try:
            cursor = self.connection.cursor()
            
            # Tanlangan ID bo'yicha rasmni olish
            cursor.execute("SELECT img FROM face_data WHERE id = %s", (face_id,))
            img_data = cursor.fetchone()[0]
            
            # Rasmni ko'rsatish
            if img_data:
                # Bytedan rasmga o'tkazish
                img = Image.open(io.BytesIO(img_data))
                img = img.resize((300, 300), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Labelga rasmni joylash
                self.image_label.configure(image=photo)
                self.image_label.image = photo  # Garbage collectiondan saqlash uchun
            else:
                self.image_label.configure(image=None)
                self.image_label.image = None
            
        except Exception as e:
            messagebox.showerror("Xatolik", f"Rasmni ko'rsatishda xatolik: {e}")
        finally:
            self.close_connection()
    
    def add_new_face(self):
        """Yangi shaxs qo'shish"""
        # Soddalik uchun kamera orqali rasm olishni simulyatsiya qilamiz
        # Haqiqiy loyihada bu kamera orqali rasm olish bo'lishi kerak
        
        # Simulyatsiya uchun fayl tanlash oynasi
        file_path = filedialog.askopenfilename(
            title="Shaxs rasmini tanlang",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        try:
            # Rasmni o'qish
            with open(file_path, "rb") as f:
                img_data = f.read()
            
            # Bazaga qo'shish
            if not self.connect_db():
                return
            
            cursor = self.connection.cursor()
            
            # Rasmni face_data jadvaliga qo'shish
            cursor.execute(
                "INSERT INTO face_data (img) VALUES (%s) RETURNING id",
                (img_data,)
            )
            face_id = cursor.fetchone()[0]
            
            # face_log_data ga ham yozish
            cursor.execute(
                "INSERT INTO face_log_data (id_name) VALUES (%s)",
                (face_id,)
            )
            
            self.connection.commit()
            messagebox.showinfo("Muvaffaqiyat", "Yangi shaxs muvaffaqiyatli qo'shildi!")
            
            # Ma'lumotlarni yangilash
            self.refresh_data()
            
        except Exception as e:
            messagebox.showerror("Xatolik", f"Yangi shaxs qo'shishda xatolik: {e}")
        finally:
            self.close_connection()
    
    def run(self):
        """Dasturni ishga tushirish"""
        self.root.mainloop()

if __name__ == "__main__":
    # Bazaga ulanish parametrlari
    DB_NAME = "face_db"
    DB_USER = "postgres"
    DB_PASSWORD = "123"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    
    # Dasturni ishga tushirish
    app = FaceDBViewer(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)
    app.run()
