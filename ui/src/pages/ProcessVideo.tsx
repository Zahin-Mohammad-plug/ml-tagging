import React, { useState, useEffect } from 'react';
import {
  Container,
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Alert,
  CircularProgress,
  Paper,
  Divider,
  Switch,
  FormControlLabel,
  Chip,
  TextField,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  Checkbox
} from '@mui/material';
import { ExpandMore } from '@mui/icons-material';
import { PlayArrow, Info, Cancel } from '@mui/icons-material';
import axios from 'axios';
import VideoGallery from '../components/VideoGallery';
import JobMonitor from '../components/JobMonitor';

const ProcessVideo: React.FC = () => {
  const [videoId, setVideoId] = useState('');
  const [selectedVideoIds, setSelectedVideoIds] = useState<string[]>([]);
  const [multiSelect, setMultiSelect] = useState(true); // Default to multi-select
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [activeJobs, setActiveJobs] = useState<string[]>([]);
  
  // Job options
  const [maxFrames, setMaxFrames] = useState<number | null>(null);
  const [sampleFps, setSampleFps] = useState<number | null>(null);
  const [autoApproveThreshold, setAutoApproveThreshold] = useState<number | null>(null);
  const [autoDeleteThreshold, setAutoDeleteThreshold] = useState<number | null>(null);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [cancellingAll, setCancellingAll] = useState(false);
  const [cleanProcess, setCleanProcess] = useState(false);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:9898';
  
  // Load default settings
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const response = await axios.get(`${apiUrl}/settings`);
        if (response.data.max_frames_per_scene || response.data.max_frames_per_video) {
          setMaxFrames(response.data.max_frames_per_video || response.data.max_frames_per_scene);
        }
        if (response.data.sample_fps) {
          setSampleFps(response.data.sample_fps);
        }
        if (response.data.default_auto_threshold) {
          setAutoApproveThreshold(response.data.default_auto_threshold);
        }
      } catch (err) {
        // Use defaults if settings fail
        setMaxFrames(100);
        setSampleFps(0.5);
        setAutoApproveThreshold(0.8);
      }
    };
    loadSettings();
  }, [apiUrl]);

  // Load active jobs on mount and periodically - use single endpoint for all jobs
  useEffect(() => {
    const fetchActiveJobs = async () => {
      try {
        const response = await axios.get(`${apiUrl}/jobs/active`);
        const jobs = response.data.jobs || [];
        // Get job IDs for active jobs
        const active = jobs.map((job: any) => job.job_id);
        setActiveJobs(active);
      } catch (err) {
        // Silently fail - jobs might not be available yet
        console.error('Failed to fetch active jobs:', err);
      }
    };

    fetchActiveJobs();
    // Refresh every 5 seconds
    const interval = setInterval(fetchActiveJobs, 5000);
    return () => clearInterval(interval);
  }, [apiUrl]);

  const handleSelectVideo = (id: string) => {
    setVideoId(id);
    setError(null);
  };

  const handleProcess = async () => {
    const videosToProcess = multiSelect ? selectedVideoIds : (videoId ? [videoId] : []);
    
    if (videosToProcess.length === 0) {
      setError('Please select at least one video from the gallery');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const jobOptions: any = {
        priority: 'normal',
        force_reprocess: false,
        clean_process: cleanProcess
      };
      
      // Add job options if set
      if (maxFrames !== null && maxFrames > 0) {
        jobOptions.max_frames = maxFrames;
      }
      if (sampleFps !== null && sampleFps > 0) {
        jobOptions.sample_fps = sampleFps;
      }
      if (autoApproveThreshold !== null && autoApproveThreshold >= 0 && autoApproveThreshold <= 1) {
        jobOptions.auto_approve_threshold = autoApproveThreshold;
      }
      if (autoDeleteThreshold !== null && autoDeleteThreshold >= 0 && autoDeleteThreshold <= 1) {
        jobOptions.auto_delete_threshold = autoDeleteThreshold;
      }
      
      const jobPromises = videosToProcess.map(id =>
        axios.post(`${apiUrl}/ingest`, {
          video_id: id,
          ...jobOptions
        })
      );
      
      const responses = await Promise.all(jobPromises);
      const newJobIds = responses.map((r: any) => r.data.job_id);
      
      setSuccess(`Processing started for ${videosToProcess.length} video(s)!`);
      
      // Add to active jobs list
      setActiveJobs((prev: string[]) => [...prev, ...newJobIds]);
      
      // Clear selection
      if (multiSelect) {
        setSelectedVideoIds([]);
      } else {
        setVideoId('');
      }
      
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to start processing';
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Process Video
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Select a video to analyze with AI. The system will extract frames, analyze content with computer vision and audio,
        then suggest relevant tags for your review.
      </Typography>

      {/* Job Monitor - Shows live processing status for all active jobs */}
      {activeJobs.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Active Processing Jobs ({activeJobs.length})
            </Typography>
            <Button
              variant="outlined"
              color="error"
              startIcon={cancellingAll ? <CircularProgress size={20} /> : <Cancel />}
              onClick={async () => {
                if (!window.confirm(`Are you sure you want to cancel ALL ${activeJobs.length} active job(s)? This action cannot be undone.`)) {
                  return;
                }

                try {
                  setCancellingAll(true);
                  const response = await axios.delete(`${apiUrl}/jobs/active`);
                  setSuccess(response.data.message || `Cancelled ${response.data.cancelled_count || 0} job(s)`);
                  setActiveJobs([]);
                } catch (err: any) {
                  const errorMsg = err.response?.data?.detail || 'Failed to cancel all jobs';
                  setError(errorMsg);
                } finally {
                  setCancellingAll(false);
                }
              }}
              disabled={cancellingAll}
              size="small"
            >
              {cancellingAll ? 'Cancelling...' : 'Cancel All Jobs'}
            </Button>
          </Box>
          {/* Fetch all active jobs in one call and display them */}
          {activeJobs.map((jobId) => (
            <JobMonitor
              key={jobId}
              jobId={jobId}
              autoRefresh={true}
              refreshInterval={5000}
              onComplete={() => {
                // Remove from active jobs when completed
                setActiveJobs((prev: string[]) => prev.filter((id: string) => id !== jobId));
                setSuccess('Processing completed! Check the Review Queue for suggestions.');
              }}
            />
          ))}
        </Box>
      )}

      {/* Alerts */}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      {/* Processing Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <PlayArrow /> Processing Settings
          </Typography>

          <Box sx={{ display: 'flex', gap: 2, mb: 2, alignItems: 'center', flexWrap: 'wrap' }}>
            <FormControlLabel
              control={
                <Switch
                  checked={multiSelect}
                  onChange={(e) => {
                    setMultiSelect(e.target.checked);
                    if (!e.target.checked) {
                      setSelectedVideoIds([]);
                    } else {
                      setVideoId('');
                    }
                  }}
                />
              }
              label="Multi-select mode"
            />
            {multiSelect && selectedVideoIds.length > 0 && (
              <Button
                size="small"
                variant="outlined"
                onClick={() => setSelectedVideoIds([])}
              >
                Clear Selection
              </Button>
            )}
            <Box sx={{ flex: 1 }}>
              {multiSelect ? (
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                  <Typography variant="body2" color="text.secondary">
                    {selectedVideoIds.length > 0 ? `${selectedVideoIds.length} video(s) selected` : 'Select videos from the gallery below'}
                  </Typography>
                  {selectedVideoIds.length > 0 && (
                    <Chip label={`${selectedVideoIds.length} selected`} size="small" color="primary" />
                  )}
                </Box>
              ) : (
            <Typography variant="body2" color="text.secondary">
              {videoId ? `Selected Video: ${videoId}` : 'Select a video from the gallery below'}
            </Typography>
              )}
            </Box>
            <Button
              variant="contained"
              onClick={handleProcess}
              disabled={loading || (multiSelect ? selectedVideoIds.length === 0 : !videoId.trim())}
              size="large"
              startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
            >
              {loading ? 'Starting...' : multiSelect ? `Process ${selectedVideoIds.length || 0} Video(s)` : 'Process Selected Video'}
            </Button>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Advanced Job Options */}
          <Accordion expanded={showAdvancedOptions} onChange={(_, expanded) => setShowAdvancedOptions(expanded)}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="subtitle1">Advanced Job Options</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={cleanProcess}
                      onChange={(e) => setCleanProcess(e.target.checked)}
                    />
                  }
                  label="Clean Process (delete all old jobs for video before processing)"
                />
                <TextField
                  type="number"
                  label="Max Frames Per Video"
                  value={maxFrames || ''}
                  onChange={(e) => setMaxFrames(e.target.value ? parseInt(e.target.value) : null)}
                  helperText="Maximum number of frames to extract and analyze (leave empty to use default from settings)"
                  inputProps={{ min: 1, max: 1000 }}
                  fullWidth
                />
                
                <TextField
                  type="number"
                  label="Sample FPS"
                  value={sampleFps || ''}
                  onChange={(e) => setSampleFps(e.target.value ? parseFloat(e.target.value) : null)}
                  helperText="Frames per second to sample from video (leave empty to use default from settings)"
                  inputProps={{ min: 0.1, max: 10, step: 0.1 }}
                  fullWidth
                />
                
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Auto-Approve Threshold: {autoApproveThreshold !== null ? `${(autoApproveThreshold * 100).toFixed(0)}%` : 'Use default'}
              </Typography>
            <Slider
                    value={autoApproveThreshold !== null ? autoApproveThreshold : 0.8}
                    onChange={(_, value) => setAutoApproveThreshold(value as number)}
              min={0}
              max={1}
              step={0.05}
              marks={[
                { value: 0.7, label: '70%' },
                      { value: 0.8, label: '80%' },
                { value: 0.9, label: '90%' }
              ]}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
            />
                  <Typography variant="caption" color="text.secondary">
                    Auto-approve suggestions above this confidence (leave at default to use settings)
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Auto-Delete Threshold: {autoDeleteThreshold !== null ? `${(autoDeleteThreshold * 100).toFixed(0)}%` : 'Disabled'}
                  </Typography>
                  <Slider
                    value={autoDeleteThreshold !== null ? autoDeleteThreshold : 0.1}
                    onChange={(_, value) => setAutoDeleteThreshold(value as number)}
                    min={0}
                    max={0.5}
                    step={0.05}
                    marks={[
                      { value: 0.05, label: '5%' },
                      { value: 0.1, label: '10%' },
                      { value: 0.2, label: '20%' }
                    ]}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                  />
                  <Typography variant="caption" color="text.secondary">
                    Auto-delete suggestions below this confidence (set to 0 to disable)
            </Typography>
                  <Button
                    size="small"
                    onClick={() => setAutoDeleteThreshold(null)}
                    sx={{ mt: 1 }}
                  >
                    Disable Auto-Delete
                  </Button>
                </Box>
              </Box>
            </AccordionDetails>
          </Accordion>

        </CardContent>
      </Card>

      {/* Video Gallery */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Select Video from Library
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Browse your recent videos and click to select one for processing
          </Typography>
          <VideoGallery
            onVideoSelect={handleSelectVideo}
            selectedVideoId={videoId}
            selectedVideoIds={selectedVideoIds}
            onMultiSelect={setSelectedVideoIds}
            multiSelect={multiSelect}
            limit={20}
            showProcessed={true}
            onSelectAll={(videoIds: string[]) => {
              setSelectedVideoIds(videoIds);
            }}
          />
        </CardContent>
      </Card>

      {/* Info Panel */}
      <Paper sx={{ p: 3, bgcolor: 'info.dark' }}>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Info sx={{ color: 'info.light' }} />
          <Box>
            <Typography variant="h6" color="info.light" gutterBottom>
              How it works
            </Typography>
            <Typography variant="body2" color="info.light">
              <strong>1. Frame Sampling:</strong> Extracts key frames from your video<br />
              <strong>2. Vision Analysis:</strong> Uses CLIP to identify visual content<br />
              <strong>3. Audio Transcription:</strong> Whisper analyzes dialogue and audio<br />
              <strong>4. Text Detection:</strong> OCR extracts on-screen text<br />
              <strong>5. Fusion:</strong> Combines signals and generates tag suggestions<br />
              <strong>6. Review:</strong> You approve or reject suggestions with evidence frames
            </Typography>
          </Box>
        </Box>
      </Paper>
    </Container>
  );
};

export default ProcessVideo;
