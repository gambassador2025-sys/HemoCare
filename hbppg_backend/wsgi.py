import os
# Force CPU-only before TensorFlow is imported anywhere.
# Use "-1" to reliably disable CUDA device visibility.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hbppg_backend.settings')
application = get_wsgi_application()
