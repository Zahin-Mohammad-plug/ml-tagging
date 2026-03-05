import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Stack,
  Alert,
  Collapse,
  IconButton,
  Button,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Sync as SyncIcon,
  Cancel as CancelIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';

interface JobProgress {
  sampling?: { 
    status: string; 
    frames_extracted?: number;
    current_step?: string;
  };
  embeddings?: { 
    status: string; 
    embeddings_count?: number;
    frames_total?: number;
    frames_processed?: number;
    current_step?: string;
  };
  asr_ocr?: { 
    status: string; 
    text_segments?: number;
    asr_segments?: number;
    ocr_texts?: number;
    current_step?: string;
  };
  fusion?: { 
    status: string; 
    suggestions_generated?: number;
    high_confidence?: number;
    medium_confidence?: number;
    low_confidence?: number;
    tags_analyzed?: number;
    tags_total?: number;
    current_tag?: string;
    tag_progress?: string;
    current_step?: string;
  };
}

interface Job {
  job_id: string;
  video_id: string;
  status: string;
  progress: JobProgress;
  created_at: string;
  error_message?: string;
}

interface JobMonitorProps {
  jobId?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
  onComplete?: () => void;
}

const JobMonitor: React.FC<JobMonitorProps> = ({
  jobId,
  autoRefresh = true,
  refreshInterval = 5000, // Reduced from 2000ms to 5000ms to reduce server load
  onComplete,
}) => {
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(true);
  const [cancelling, setCancelling] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:9898';

  const fetchJob = useCallback(async (isInitialLoad: boolean = false) => {
    if (!jobId) return;

      try {
      // Only set loading state on initial load, not during polling
      if (isInitialLoad) {
        setLoading(true);
      }
      const response = await fetch(`${apiUrl}/jobs/${jobId}`);
        if (!response.ok) {
          throw new Error('Failed to fetch job status');
        }
        const data = await response.json();
        setJob(data);
        setError(null);

      // Stop polling if job is complete
      if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
        // Call onComplete callback if job completed successfully
        if (data.status === 'completed' && onComplete) {
          onComplete();
        }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
      if (isInitialLoad) {
        setLoading(false);
      }
    }
  }, [jobId, apiUrl, onComplete]);

  useEffect(() => {
    if (!jobId) return;

    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Initial fetch with loading state
    fetchJob(true);

    // Set up polling if autoRefresh is enabled
    if (autoRefresh) {
      intervalRef.current = setInterval(() => {
        fetchJob(false); // Don't show loading during polling
      }, refreshInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [jobId, autoRefresh, refreshInterval, fetchJob]);

  const handleCancel = async () => {
    if (!window.confirm('Are you sure you want to cancel this job?')) {
      return;
    }

    try {
      setCancelling(true);
      const response = await fetch(`${apiUrl}/jobs/${jobId}/cancel`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to cancel job');
      }
      
      // Refresh job status after cancellation
      await fetchJob();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel job');
    } finally {
      setCancelling(false);
    }
  };

  const handleReprocess = async () => {
    if (!job) return;
    
    if (!window.confirm(`Are you sure you want to reprocess video ${job.video_id}? This will delete existing suggestions and create a new job.`)) {
      return;
    }

    try {
      setReprocessing(true);
      setError(null);
      const response = await fetch(`${apiUrl}/videos/${job.video_id}/reprocess`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to reprocess video');
      }
      
      const data = await response.json();
      // The reprocess endpoint returns a new job_id, but we'll keep showing the old one
      // until the parent component updates the job list
      await fetchJob();
      
      // Show success message
      if (onComplete) {
        onComplete();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reprocess video');
    } finally {
      setReprocessing(false);
    }
  };

  const handleManualRefresh = async () => {
    setRefreshing(true);
    await fetchJob();
    setRefreshing(false);
  };

  const handleDelete = async () => {
    if (!window.confirm('Are you sure you want to delete this job? This will permanently delete the job and all related data (suggestions, frames, embeddings). This action cannot be undone.')) {
      return;
    }

    try {
      setDeleting(true);
      const response = await fetch(`${apiUrl}/jobs/${jobId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to delete job');
      }
      
      // Job is deleted, call onComplete to remove from parent's list
      if (onComplete) {
        onComplete();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete job');
    } finally {
      setDeleting(false);
    }
  };

  if (!jobId) {
    return null;
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Error loading job status: {error}
      </Alert>
    );
  }

  if (!job && loading) {
    return (
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Stack direction="row" spacing={2} alignItems="center">
            <SyncIcon className="rotating" />
            <Typography>Loading job status...</Typography>
          </Stack>
        </CardContent>
      </Card>
    );
  }

  if (!job) {
    return null;
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'cancelled':
        return 'default';
      case 'queued':
        return 'default';
      default:
        return 'primary';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckIcon color="success" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      case 'cancelled':
        return <CancelIcon color="action" />;
      default:
        return <SyncIcon className="rotating" color="primary" />;
    }
  };

  const calculateProgress = (): number => {
    // ASR/OCR is not currently used, so exclude it from progress calculation
    const steps = ['sampling', 'embeddings', 'fusion'];
    let completedSteps = 0;
    let partialProgress = 0;

    steps.forEach((step) => {
      const stepProgress = job.progress?.[step as keyof JobProgress];
      if (stepProgress?.status === 'completed') {
        completedSteps++;
        } else if (stepProgress?.status === 'processing') {
          // Add partial progress for processing steps
          if (step === 'embeddings') {
            const embeddingsProgress = stepProgress as JobProgress['embeddings'];
            if (embeddingsProgress?.frames_total && embeddingsProgress.frames_processed !== undefined) {
              partialProgress += (embeddingsProgress.frames_processed / embeddingsProgress.frames_total) * (1 / steps.length);
            } else {
              // Default to 50% for processing steps without specific progress
              partialProgress += 0.5 * (1 / steps.length);
            }
          } else if (step === 'fusion') {
            const fusionProgress = stepProgress as JobProgress['fusion'];
            if (fusionProgress?.tags_total && fusionProgress.tags_analyzed !== undefined) {
              partialProgress += (fusionProgress.tags_analyzed / fusionProgress.tags_total) * (1 / steps.length);
            } else {
              // Default to 50% for processing steps without specific progress
              partialProgress += 0.5 * (1 / steps.length);
            }
          } else {
            // Default to 50% for processing steps without specific progress
            partialProgress += 0.5 * (1 / steps.length);
          }
        }
    });

    return ((completedSteps + partialProgress) / steps.length) * 100;
  };

  const getStepStatus = (step: string) => {
    const stepProgress = job.progress?.[step as keyof JobProgress];
    return stepProgress?.status || 'pending';
  };

  const getStepDetails = (step: string) => {
    const stepProgress = job.progress?.[step as keyof JobProgress];
    if (!stepProgress) return null;

    switch (step) {
      case 'sampling':
        const sampling = stepProgress as any;
        if (sampling.status === 'completed' && sampling.frames_extracted) {
          return `${sampling.frames_extracted} frames extracted`;
        }
        return sampling.current_step || null;
      case 'embeddings':
        const embeddings = stepProgress as any;
        if (embeddings.status === 'completed') {
          return `${embeddings.embeddings_count || 0} embeddings generated`;
        }
        if (embeddings.status === 'processing') {
          if (embeddings.frames_total && embeddings.frames_processed !== undefined) {
            return `${embeddings.frames_processed}/${embeddings.frames_total} frames processed`;
          }
          return embeddings.current_step || 'Processing...';
        }
        return embeddings.current_step || null;
      case 'asr_ocr':
        const asrOcr = stepProgress as any;
        if (asrOcr.status === 'completed') {
          const parts = [];
          if (asrOcr.asr_segments) parts.push(`${asrOcr.asr_segments} ASR segments`);
          if (asrOcr.ocr_texts) parts.push(`${asrOcr.ocr_texts} OCR texts`);
          return parts.length > 0 ? parts.join(', ') : `${asrOcr.text_segments || 0} segments`;
        }
        return asrOcr.current_step || 'Processing...';
      case 'fusion':
        const fusion = stepProgress as any;
        if (fusion.status === 'completed') {
          const parts = [];
          if (fusion.suggestions_generated) {
            parts.push(`${fusion.suggestions_generated} suggestions`);
          }
          if (fusion.high_confidence || fusion.medium_confidence || fusion.low_confidence) {
            const confParts = [];
            if (fusion.high_confidence) confParts.push(`High: ${fusion.high_confidence}`);
            if (fusion.medium_confidence) confParts.push(`Med: ${fusion.medium_confidence}`);
            if (fusion.low_confidence) confParts.push(`Low: ${fusion.low_confidence}`);
            if (confParts.length > 0) parts.push(`(${confParts.join(', ')})`);
          }
          return parts.length > 0 ? parts.join(' ') : 'Completed';
        }
        if (fusion.status === 'processing') {
          // Show detailed progress with tag name and progress
          const stepParts = [];
          if (fusion.current_step) {
            stepParts.push(fusion.current_step);
          }
          if (fusion.tag_progress && fusion.current_tag) {
            stepParts.push(`${fusion.tag_progress} - ${fusion.current_tag}`);
          } else if (fusion.tag_progress) {
            stepParts.push(fusion.tag_progress);
          }
          return stepParts.length > 0 ? stepParts.join(' | ') : 'Analyzing...';
        }
        return fusion.current_step || 'Analyzing...';
      default:
        return null;
    }
  };

  const progress = calculateProgress();

  return (
    <Card sx={{ mb: 2, border: job.status === 'completed' ? '2px solid #4caf50' : undefined }}>
      <CardContent>
        <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
          <Stack direction="row" spacing={2} alignItems="center">
            {getStatusIcon(job.status)}
            <Box>
              <Typography variant="h6">
                Processing Video {job.video_id}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Job ID: {job.job_id.slice(0, 8)}...
              </Typography>
            </Box>
          </Stack>
          <Stack direction="row" spacing={1} alignItems="center">
            <Chip
              label={job.status.toUpperCase()}
              color={getStatusColor(job.status)}
              size="small"
            />
            {job.status !== 'completed' && job.status !== 'failed' && job.status !== 'cancelled' && (
              <Button
                size="small"
                variant="outlined"
                color="error"
                startIcon={<CancelIcon />}
                onClick={handleCancel}
                disabled={cancelling}
              >
                {cancelling ? 'Cancelling...' : 'Cancel'}
              </Button>
            )}
            {(job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') && (
              <>
                <Button
                  size="small"
                  variant="outlined"
                  color="primary"
                  startIcon={<RefreshIcon />}
                  onClick={handleReprocess}
                  disabled={reprocessing}
                >
                  {reprocessing ? 'Reprocessing...' : 'Reprocess'}
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteIcon />}
                  onClick={handleDelete}
                  disabled={deleting}
                >
                  {deleting ? 'Deleting...' : 'Delete'}
                </Button>
              </>
            )}
            <IconButton
              size="small"
              onClick={handleManualRefresh}
              disabled={refreshing}
              title="Refresh status"
            >
              <SyncIcon className={refreshing ? 'rotating' : ''} />
            </IconButton>
            <IconButton
              size="small"
              onClick={() => setExpanded(!expanded)}
              sx={{
                transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.3s',
              }}
            >
              <ExpandMoreIcon />
            </IconButton>
          </Stack>
        </Stack>

        <Box mb={2}>
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{ height: 8, borderRadius: 1 }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
            {Math.round(progress)}% complete
          </Typography>
        </Box>

        <Collapse in={expanded}>
          <Stack spacing={1.5}>
            {/* Sampling Step */}
            <Stack direction="row" spacing={2} alignItems="center">
              <Chip
                label="Sampling"
                size="small"
                color={
                  getStepStatus('sampling') === 'completed' 
                    ? 'success' 
                    : getStepStatus('sampling') === 'processing'
                    ? 'primary'
                    : 'default'
                }
                sx={{ minWidth: 100 }}
              />
              <Box sx={{ flex: 1 }}>
                {getStepDetails('sampling') && (
                  <Typography variant="caption" color="text.secondary">
                    {getStepDetails('sampling')}
                  </Typography>
                )}
              </Box>
            </Stack>

            {/* Embeddings Step */}
            <Stack direction="row" spacing={2} alignItems="center">
              <Chip
                label="Embeddings"
                size="small"
                color={
                  getStepStatus('embeddings') === 'completed' 
                    ? 'success' 
                    : getStepStatus('embeddings') === 'processing'
                    ? 'primary'
                    : 'default'
                }
                sx={{ minWidth: 100 }}
              />
              <Box sx={{ flex: 1 }}>
                {getStepDetails('embeddings') && (
                  <Typography variant="caption" color="text.secondary">
                    {getStepDetails('embeddings')}
                  </Typography>
                )}
              </Box>
            </Stack>

            {/* ASR/OCR Step - Commented out as not currently used */}
            {/* <Stack direction="row" spacing={2} alignItems="center">
              <Chip
                label="ASR/OCR"
                size="small"
                color={
                  getStepStatus('asr_ocr') === 'completed' 
                    ? 'success' 
                    : getStepStatus('asr_ocr') === 'processing'
                    ? 'primary'
                    : 'default'
                }
                sx={{ minWidth: 100 }}
              />
              <Box sx={{ flex: 1 }}>
                {getStepDetails('asr_ocr') && (
                  <Typography variant="caption" color="text.secondary">
                    {getStepDetails('asr_ocr')}
                  </Typography>
                )}
              </Box>
            </Stack> */}

            {/* Fusion Step */}
            <Stack direction="row" spacing={2} alignItems="center">
              <Chip
                label={
                  job.progress?.fusion?.tag_progress 
                    ? `Fusion (${job.progress.fusion.tag_progress.split('/')[0]}/${job.progress.fusion.tag_progress.split('/')[1] || '?'})`
                    : "Fusion"
                }
                size="small"
                color={
                  getStepStatus('fusion') === 'completed' 
                    ? 'success' 
                    : getStepStatus('fusion') === 'processing'
                    ? 'primary'
                    : 'default'
                }
                sx={{ minWidth: 100 }}
              />
              <Box sx={{ flex: 1 }}>
                {getStepDetails('fusion') && (
                  <Typography variant="caption" color="text.secondary">
                    {getStepDetails('fusion')}
                  </Typography>
                )}
              </Box>
            </Stack>
          </Stack>

          {job.error_message && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {job.error_message}
            </Alert>
          )}
        </Collapse>
      </CardContent>

      <style>{`
        @keyframes rotate {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
        .rotating {
          animation: rotate 2s linear infinite;
        }
      `}</style>
    </Card>
  );
};

export default JobMonitor;
