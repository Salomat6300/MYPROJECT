import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np

class FaceModel:
    def __init__(self, device=None, mtcnn_thresholds=[0.6,0.7,0.7], keep_all=False):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # MTCNN for detection (GPU-accelerated if CUDA available)
        self.mtcnn = MTCNN(margin=14, keep_all=keep_all, device=self.device)
        # InceptionResnetV1 embedding model
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    @torch.no_grad()
    def detect_and_align(self, bgr_image):
        # expecting BGR (OpenCV) -> convert to RGB internally in facenet_pytorch
        # returns list of PIL-like cropped face tensors or empty list
        # mtcnn returns cropped face tensors (RGB, float) on device if keep_all True -> list
        faces = self.mtcnn(bgr_image)  # facenet-pytorch accepts cv2 image (BGR or PIL)
        # If keep_all=False and one face, faces is tensor or None
        if faces is None:
            return []
        # Normalize to CPU numpy arrays or keep as tensors for embedding
        if isinstance(faces, torch.Tensor):
            faces = [faces]
        return faces

    @torch.no_grad()
    def embeddings(self, face_tensors):
        # face_tensors: list of torch.Tensor on CPU or device. Convert and batch.
        if not face_tensors:
            return np.array([])
        # Make a single batch
        batch = torch.stack([f.to(self.device) if not isinstance(f, torch.nn.Parameter) else f for f in face_tensors])
        # Ensure float32 and proper shape
        batch = batch.float()
        emb = self.resnet(batch)  # (N, 512)
        emb = emb.cpu().numpy()
        return emb
