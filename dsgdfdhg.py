import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU mavjudmi: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"CUDA versiyasi: {tf.sysconfig.get_build_info()['cuda_version']}")
print(f"cuDNN versiyasi: {tf.sysconfig.get_build_info()['cudnn_version']}")

# import torch
# print(torch.cuda.is_available())  # True chiqishi kerak
# print(torch.version.cuda)         # 11.8 chiqishi kerak