"""Pydantic models for API request/response schemas"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

class JobStatus(str, Enum):
    QUEUED = "queued"
    SAMPLING = "sampling" 
    EMBEDDINGS = "embeddings"
    ASR_OCR = "asr_ocr"
    FUSION = "fusion"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SuggestionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPLIED = "auto_applied"

class JobPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

# Health Check Models
class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall system health")
    database: str = Field(..., description="Database connection status")
    queue: str = Field(..., description="Queue system status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Ingest Models
class IngestRequest(BaseModel):
    video_id: str = Field(..., description="Video ID to process")
    priority: JobPriority = Field(default=JobPriority.NORMAL, description="Processing priority")
    force_reprocess: bool = Field(default=False, description="Force reprocessing even if recently processed")
    clean_process: bool = Field(default=False, description="Delete all old jobs for video before processing")
    # Video metadata (optional, for creating video record)
    title: Optional[str] = Field(None, description="Video title")
    path: Optional[str] = Field(None, description="Path to video file")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    frame_rate: Optional[float] = Field(None, description="Video frame rate")
    # Job options
    max_frames: Optional[int] = Field(None, description="Maximum number of frames to process (overrides settings)")
    auto_approve_threshold: Optional[float] = Field(None, description="Auto-approve suggestions above this confidence (0.0-1.0)")
    auto_delete_threshold: Optional[float] = Field(None, description="Auto-delete suggestions below this confidence (0.0-1.0)")
    sample_fps: Optional[float] = Field(None, description="Frames per second to sample (overrides settings)")
    
class IngestResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Human-readable status message")
    estimated_duration_minutes: Optional[int] = Field(None, description="Estimated processing time")

# Evidence Models
class EvidenceFrame(BaseModel):
    frame_number: int = Field(..., description="Frame number in scene")
    timestamp_seconds: float = Field(..., description="Timestamp in scene")
    confidence: float = Field(..., description="Confidence score for this frame")
    thumbnail_url: Optional[str] = Field(None, description="URL to frame thumbnail")
    signals: Dict[str, Any] = Field(default_factory=dict, description="Signal details (vision, asr, ocr)")

class ConfidenceBreakdown(BaseModel):
    vision_confidence: float = Field(..., description="Visual similarity confidence")
    asr_confidence: Optional[float] = Field(None, description="Audio transcription confidence") 
    ocr_confidence: Optional[float] = Field(None, description="Text detection confidence")
    temporal_consistency: float = Field(..., description="Multi-frame agreement score")
    calibrated_confidence: float = Field(..., description="Final calibrated confidence")
    
class TagContext(BaseModel):
    tag_id: str = Field(..., description="Tag ID")
    tag_name: str = Field(..., description="Tag name")
    parent_tags: List[str] = Field(default_factory=list, description="Parent tag names")
    child_tags: List[str] = Field(default_factory=list, description="Child tag names") 
    synonyms: List[str] = Field(default_factory=list, description="Tag synonyms used in matching")
    
# Suggestion Models
class Suggestion(BaseModel):
    model_config = {"populate_by_name": True}
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique suggestion ID")
    video_id: str = Field(..., alias="scene_id", description="Video ID")
    tag_name: str = Field(..., description="Suggested tag name")
    confidence: float = Field(..., description="Overall confidence score")
    status: SuggestionStatus = Field(default=SuggestionStatus.PENDING, description="Suggestion status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
class SuggestionResponse(BaseModel):
    """Full suggestion details with evidence"""
    id: str
    video_id: str
    video_title: Optional[str] = Field(None, description="Video title")
    tag_context: TagContext
    confidence: float
    confidence_breakdown: ConfidenceBreakdown
    status: SuggestionStatus
    evidence_frames: List[EvidenceFrame] = Field(default_factory=list, description="Supporting evidence frames")
    reasoning: Optional[str] = Field(None, description="Human-readable explanation")
    created_at: datetime
    reviewed_at: Optional[datetime] = Field(None, description="Review timestamp")
    reviewed_by: Optional[str] = Field(None, description="Reviewer identifier")
    notes: Optional[str] = Field(None, description="Review notes")
    is_backup: bool = Field(default=False, description="True if this is a backup (parent) tag")
    
# Approval Models
class ApprovalRequest(BaseModel):
    approved_by: Optional[str] = Field(None, description="User identifier (optional for anonymous mode)")
    notes: Optional[str] = Field(None, description="Optional review notes")

# Text-based tag suggestions
class TextBasedSuggestionsRequest(BaseModel):
    use_description: bool = Field(default=True, description="Whether to use scene description")
    use_title: bool = Field(default=True, description="Whether to use scene title")
    use_ocr: bool = Field(default=False, description="Whether to include OCR text if available")
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_suggestions: int = Field(default=50, ge=1, le=200, description="Maximum number of suggestions to return")

class TextBasedSuggestion(BaseModel):
    tag_id: str = Field(..., description="Tag ID")
    tag_name: str = Field(..., description="Tag name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(default="text", description="Source of suggestion")
    text_type: Optional[str] = Field(None, description="Type of text used (description, title, ocr)")

class TextBasedSuggestionsResponse(BaseModel):
    video_id: str = Field(..., description="Video ID")
    suggestions: List[TextBasedSuggestion] = Field(..., description="List of tag suggestions")
    text_used: Dict[str, bool] = Field(..., description="Which text sources were used")
    total_tags_checked: int = Field(..., description="Total number of tags checked")
    
# Job Models
class Job(BaseModel):
    model_config = {"populate_by_name": True}
    
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    video_id: str = Field(..., alias="scene_id")
    status: JobStatus = Field(default=JobStatus.QUEUED)
    priority: JobPriority = Field(default=JobPriority.NORMAL)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: Dict[str, Any] = Field(default_factory=dict, description="Step-by-step progress")
    
class JobResponse(BaseModel):
    """Job details with progress information"""
    job_id: str
    video_id: str
    status: JobStatus
    priority: JobPriority
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    progress: Dict[str, Any]
    estimated_completion: Optional[datetime]
    suggestions_created: int = Field(default=0, description="Number of suggestions generated")
    
# Statistics Models
class SuggestionStats(BaseModel):
    total_suggestions: int = 0
    pending_count: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    auto_applied_count: int = 0
    average_confidence: Optional[float] = None
    accuracy_rate: Optional[float] = Field(None, description="Approval rate for reviewed suggestions")
    
class JobStats(BaseModel):
    total_jobs: int = 0
    queued_count: int = 0
    processing_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    average_processing_time_minutes: Optional[float] = None
    
class SystemStats(BaseModel):
    suggestions: SuggestionStats
    jobs: JobStats
    system: Dict[str, Any] = Field(default_factory=dict)
    
# Video Scene Models
class VideoScene(BaseModel):
    id: str
    title: Optional[str]
    description: Optional[str] = None
    path: str
    screenshot_path: Optional[str] = None
    duration: Optional[float]  # seconds
    frame_rate: Optional[float]
    width: Optional[int]
    height: Optional[int]
    tags: List[str] = Field(default_factory=list)
    
class VideoTag(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    parent_ids: List[str] = Field(default_factory=list)
    child_ids: List[str] = Field(default_factory=list)