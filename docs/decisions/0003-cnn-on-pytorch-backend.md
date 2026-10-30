# 0003 Run the CNN with Keras on the PyTorch backend

**Context.** EasyOCR needs PyTorch. The script CNN was trained with TensorFlow. Loading both took about 2 GB
at peak during a scan, and the app crashed on the host.

**Decision.** The app loads the saved `.keras` model with `KERAS_BACKEND=torch`; TensorFlow is only in
`requirements-dev.txt` for retraining.

**Evidence.** Same probabilities on 53 test images (largest difference 3.4e-6, no changed answer); prediction
about twice as fast. Peak scan memory 2.04 GB -> 1.46 GB together with the shared OCR detector.

**Consequences.** One deep-learning runtime in production; a 546 MB smaller install.
