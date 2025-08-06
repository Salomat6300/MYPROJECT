class YuzTanibOlovchi:
    def __init__(self, face_db, device=None, tolerance=0.8):
        self.face_db = face_db
        self.device = device or ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
        self.model = FaceModel(device=self.device)
        self.orient_detector = FaceOrientationDetector()
        self.cam = None
        self.known_ids = []
        self.known_embs = np.empty((0,512), dtype=np.float32)
        self.tolerance = tolerance
        self.last_log_times = {}  # {id: datetime}

    def ishga_tushirish(self, cam_index=0):
        # load known embeddings
        ids, embs = self.face_db.barcha_yuzlarni_olish()
        if embs:
            self.known_ids = ids
            self.known_embs = np.vstack(embs).astype(np.float32)
        else:
            self.known_ids = []
            self.known_embs = np.empty((0,512), dtype=np.float32)
        self.cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cam.isOpened():
            print("Kamerani ochib bo'lmadi.")
            return False
        return True

    def yuzlarni_tanib_olish(self):
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                # detect faces (returns list of torch tensors already aligned)
                faces = self.model.detect_and_align(frame)
                if not faces:
                    cv2.imshow("Facial recognition", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Get embeddings for all faces found (batch on GPU)
                emb_batch = self.model.embeddings(faces)  # (N, 512) numpy
                for i, emb in enumerate(emb_batch):
                    # Convert face tensor -> crop image for orientation and DB save
                    # facenet-pytorch's MTCNN crop returns torch tensors in RGB float [0,1],
                    # but we need BGR uint8 for cv2 display / DB save. Convert:
                    face_tensor = faces[i].cpu().permute(1,2,0).numpy()  # HWC RGB float
                    face_img = (face_tensor * 255).astype(np.uint8)[:,:,::-1]  # BGR
                    # Orientation check on the crop
                    orientation_ok = self.orient_detector.detect(face_img)
                    if not orientation_ok:
                        label = "Yuz holati notog'ri"
                    else:
                        label = "Nomalum shaxs!"
                        yuz_id = None
                        if self.known_embs.shape[0] > 0:
                            # Cosine similarity via cdist (euclidean or cosine). We compute cosine distance.
                            dists = cdist(self.known_embs, emb.reshape(1, -1), metric='cosine').reshape(-1)
                            idx = np.argmin(dists)
                            if dists[idx] <= (1 - (1 - self.tolerance)):  # convert tolerance intuition
                                # simpler: treat dists[idx] < 0.4 as match (experimentally)
                                if dists[idx] < 0.4:
                                    yuz_id = self.known_ids[idx]
                                    label = f"ID-{yuz_id}"
                                    # log every 30 seconds max
                                    now = datetime.now()
                                    last = self.last_log_times.get(yuz_id)
                                    if (last is None) or ((now - last).total_seconds() >= 30):
                                        self.face_db.kirishni_loglash(yuz_id)
                                        self.last_log_times[yuz_id] = now

                        if yuz_id is None:
                            # not matched: save to DB (async)
                            fut = self.face_db.yuz_qoshish(face_img, emb)
                            # fut is Future: get result in background but update local known list when done
                            def _on_done(f):
                                new_id = f.result()
                                if new_id:
                                    self.known_ids.append(new_id)
                                    if self.known_embs.size == 0:
                                        self.known_embs = emb.reshape(1,-1)
                                    else:
                                        self.known_embs = np.vstack([self.known_embs, emb.reshape(1,-1)])
                                    self.last_log_times[new_id] = datetime.now()
                            fut.add_done_callback(_on_done)

                    # draw label and box. We don't have face bbox here — MTCNN can return boxes,
                    # but for simplicity show label at top-left
                    cv2.putText(frame, label, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                cv2.imshow("Facial recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.tozalash()

    def tozalash(self):
        if self.cam:
            self.cam.release()
        cv2.destroyAllWindows()
