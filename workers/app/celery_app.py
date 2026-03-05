"""Celery configuration for workers"""

from celery import Celery
from kombu import Queue
import os

# Redis connection
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
app = Celery("ml-tagger-workers", broker=redis_url, backend=redis_url)

# Queue configuration
app.conf.update(
    # Task routing
    task_routes={
        'app.sampler.extract_frames': {'queue': 'sampling'},
        'app.embeddings.generate_embeddings': {'queue': 'embeddings'},
        'app.asr_ocr.process_audio_text': {'queue': 'asr_ocr'},
        'app.fusion.generate_suggestions': {'queue': 'fusion'},
    },
    
    # Queue definitions with priorities
    task_queues=[
        Queue('sampling', priority=0),
        Queue('embeddings', priority=1),
        Queue('asr_ocr', priority=1),
        Queue('fusion', priority=2),
    ],
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Retry settings
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Result settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Performance settings
    worker_log_color=False,
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
)

# Import task modules
app.autodiscover_tasks([
    'app.sampler',
    'app.embeddings',
    'app.asr_ocr', 
    'app.fusion'
])