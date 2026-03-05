import React, { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  Chip,
  CircularProgress,
  Grid,
  Card,
  CardContent
} from '@mui/material';
import { ArrowBack, CheckCircle, Cancel, Refresh } from '@mui/icons-material';
import axios from 'axios';
import VideoPlayer from '../components/VideoPlayer';

interface SuggestionDetailData {
  id: string;
  video_id: string;
  video_title?: string;
  tag_context: {
    tag_name: string;
    parent_tags?: string[];
    child_tags?: string[];
    synonyms?: string[];
  };
  confidence: number;
  confidence_breakdown: {
    vision_confidence: number;
    asr_confidence?: number;
    ocr_confidence?: number;
    temporal_consistency: number;
    calibrated_confidence: number;
  };
  status: string;
  created_at: string;
  reasoning?: string;
  evidence_frames?: Array<{
    frame_number: number;
    timestamp_seconds: number;
    confidence: number;
    thumbnail_url?: string;
    signals: any;
  }>;
}

const SuggestionDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [suggestion, setSuggestion] = useState<SuggestionDetailData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reprocessing, setReprocessing] = useState(false);
  const [seekTo, setSeekTo] = useState<number | undefined>(undefined);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:9898';

  // Check for seekTo parameter in URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const seekToParam = params.get('seekTo');
    if (seekToParam) {
      const timestamp = parseFloat(seekToParam);
      if (!isNaN(timestamp) && timestamp >= 0) {
        setSeekTo(timestamp);
      }
    }
  }, []);

  const fetchSuggestion = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${apiUrl}/suggestions/${id}`);
      setSuggestion(response.data);
      
      setError(null);
    } catch (err) {
      console.error('Failed to fetch suggestion:', err);
      setError('Failed to load suggestion details');
    } finally {
      setLoading(false);
    }
  }, [id, apiUrl]);

  useEffect(() => {
    if (id) {
      fetchSuggestion();
    }
  }, [id, fetchSuggestion]);

  const handleApprove = async () => {
    try {
      await axios.post(`${apiUrl}/suggestions/${id}/approve`, {
        approved_by: 'ui_user',
        notes: 'Approved from detail view'
      });
      navigate('/review');
    } catch (err) {
      console.error('Failed to approve:', err);
      alert('Failed to approve suggestion');
    }
  };

  const handleReject = async () => {
    try {
      await axios.post(`${apiUrl}/suggestions/${id}/reject`, {
        approved_by: 'ui_user',
        notes: 'Rejected from detail view'
      });
      navigate('/review');
    } catch (err) {
      console.error('Failed to reject:', err);
      alert('Failed to reject suggestion');
    }
  };

  const handleReprocess = async () => {
    if (!suggestion?.video_id) {
      alert('Video ID not available');
      return;
    }
    
    if (!window.confirm(`Are you sure you want to reprocess video ${suggestion.video_id}? This will delete all existing suggestions and jobs for this video and create a new processing job.`)) {
      return;
    }
    
    try {
      setReprocessing(true);
      const response = await axios.post(`${apiUrl}/videos/${suggestion.video_id}/reprocess`);
      alert(`Video is being reprocessed. New job ID: ${response.data.job_id}`);
      navigate('/review');
    } catch (err: any) {
      console.error('Failed to reprocess video:', err);
      alert(`Failed to reprocess video: ${err.response?.data?.detail || err.message}`);
    } finally {
      setReprocessing(false);
    }
  };

  if (loading) {
    return (
      <Container sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
        <CircularProgress />
      </Container>
    );
  }

  if (error || !suggestion) {
    return (
      <Container sx={{ mt: 4 }}>
        <Paper sx={{ p: 3, bgcolor: 'error.dark' }}>
          <Typography variant="h6" color="error.light">
            {error || 'Suggestion not found'}
          </Typography>
          <Button onClick={() => navigate('/suggestions')} sx={{ mt: 2 }}>
            Back to Suggestions
          </Button>
        </Paper>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Button
        startIcon={<ArrowBack />}
        onClick={() => navigate('/suggestions')}
        sx={{ mb: 3 }}
      >
        Back to Suggestions
      </Button>

      <Paper sx={{ p: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="h4">
                Tag Suggestion Details
              </Typography>
              <Chip
                label={suggestion.status}
                color={
                  suggestion.status === 'approved' ? 'success' :
                  suggestion.status === 'pending' ? 'warning' : 'error'
                }
              />
            </Box>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="overline" display="block" gutterBottom>
                  Suggested Tag
                </Typography>
                <Chip label={suggestion.tag_context.tag_name} size="medium" sx={{ fontSize: '1.1rem', py: 2.5, px: 2, mb: 2 }} />
                {(suggestion.tag_context.parent_tags && suggestion.tag_context.parent_tags.length > 0) || 
                 (suggestion.tag_context.child_tags && suggestion.tag_context.child_tags.length > 0) ? (
                  <Box sx={{ mt: 2 }}>
                    {suggestion.tag_context.parent_tags && suggestion.tag_context.parent_tags.length > 0 && (
                      <Box sx={{ mb: 1 }}>
                        <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                          Parent tags:
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                          {suggestion.tag_context.parent_tags.map((tag, idx) => (
                            <Chip key={idx} label={tag} size="small" variant="outlined" color="primary" />
                          ))}
                        </Box>
                      </Box>
                    )}
                    {suggestion.tag_context.child_tags && suggestion.tag_context.child_tags.length > 0 && (
                      <Box>
                        <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                          Child tags:
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                          {suggestion.tag_context.child_tags.map((tag, idx) => (
                            <Chip key={idx} label={tag} size="small" variant="outlined" color="secondary" />
                          ))}
                        </Box>
                      </Box>
                    )}
                  </Box>
                ) : null}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Confidence Score
                </Typography>
                <Typography variant="h4" color={
                  suggestion.confidence >= 0.8 ? 'success.main' :
                  suggestion.confidence >= 0.6 ? 'warning.main' : 'error.main'
                }>
                  {(suggestion.confidence * 100).toFixed(1)}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Video Information
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Video ID: {suggestion.video_id}
                </Typography>
                {suggestion.video_title && (
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Title: {suggestion.video_title}
                  </Typography>
                )}
                <Typography variant="body2" color="text.secondary">
                  Created: {new Date(suggestion.created_at).toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Video Player */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Video Preview
                </Typography>
                <VideoPlayer 
                  sceneId={suggestion.video_id}
                  sceneTitle={suggestion.video_title}
                  seekTo={seekTo}
                />
              </CardContent>
            </Card>
          </Grid>

          {suggestion.reasoning && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    ML Reasoning
                  </Typography>
                  <Typography variant="body2">
                    {suggestion.reasoning}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {suggestion.evidence_frames && suggestion.evidence_frames.length > 0 && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Evidence Frames ({suggestion.evidence_frames.length})
                  </Typography>
                  <Grid container spacing={2}>
                    {suggestion.evidence_frames.map((frame, idx) => (
                      <Grid item xs={6} sm={4} md={3} key={idx}>
                        <Box sx={{ position: 'relative' }}>
                          {(() => {
                            // Use relative URL so it goes through the proxy
                            // thumbnail_url should already be set to /api/frames/{frame_id}/image
                            const imageUrl = frame.thumbnail_url || 
                              (frame.signals?.frame_id ? `/api/frames/${frame.signals.frame_id}/image` : null);
                            
                            return imageUrl ? (
                              <img
                                src={imageUrl}
                                alt={`Frame ${frame.frame_number} at ${frame.timestamp_seconds.toFixed(1)}s`}
                                style={{ width: '100%', borderRadius: 8, minHeight: '120px', objectFit: 'cover', cursor: 'pointer' }}
                                onClick={() => setSeekTo(frame.timestamp_seconds)}
                                onError={(e: any) => {
                                  // Show error placeholder instead of hiding
                                  const target = e.target as HTMLImageElement;
                                  target.style.display = 'none';
                                  const parent = target.parentElement;
                                  if (parent && !parent.querySelector('.error-placeholder')) {
                                    const placeholder = document.createElement('div');
                                    placeholder.className = 'error-placeholder';
                                    placeholder.style.cssText = 'padding: 20px; text-align: center; background: #333; color: #fff; border-radius: 8px;';
                                    placeholder.textContent = 'Image not available';
                                    parent.appendChild(placeholder);
                                  }
                                }}
                                onLoad={(e: any) => {
                                  // Remove error placeholder if image loads successfully
                                  const parent = e.target.parentElement;
                                  const placeholder = parent?.querySelector('.error-placeholder');
                                  if (placeholder) {
                                    placeholder.remove();
                                  }
                                }}
                              />
                            ) : (
                            <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'grey.800' }}>
                              <Typography variant="caption">No thumbnail available</Typography>
                            </Paper>
                          );
                          })()}
                          <Chip
                            label={`${(frame.confidence * 100).toFixed(0)}%`}
                            size="small"
                            sx={{
                              position: 'absolute',
                              top: 8,
                              right: 8,
                              bgcolor: 'rgba(0,0,0,0.7)',
                              color: 'white'
                            }}
                          />
                          <Typography
                            variant="caption"
                            display="block"
                            onClick={() => setSeekTo(frame.timestamp_seconds)}
                            sx={{
                              mt: 0.5,
                              cursor: 'pointer',
                              color: 'primary.light',
                              textAlign: 'center',
                              '&:hover': {
                                textDecoration: 'underline'
                              }
                            }}
                            title={`Click to jump to ${frame.timestamp_seconds.toFixed(1)}s`}
                          >
                            Frame {frame.frame_number} • {frame.timestamp_seconds.toFixed(1)}s
                          </Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          )}

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end', flexWrap: 'wrap' }}>
              {suggestion.status === 'pending' && (
                <>
                  <Button
                    variant="contained"
                    color="error"
                    size="large"
                    startIcon={<Cancel />}
                    onClick={handleReject}
                  >
                    Reject
                  </Button>
                  <Button
                    variant="contained"
                    color="success"
                    size="large"
                    startIcon={<CheckCircle />}
                    onClick={handleApprove}
                  >
                    Approve
                  </Button>
                </>
              )}
              <Button
                variant="outlined"
                color="primary"
                size="large"
                startIcon={<Refresh />}
                onClick={handleReprocess}
                disabled={reprocessing}
              >
                {reprocessing ? 'Reprocessing...' : 'Reprocess Video'}
              </Button>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
};

export default SuggestionDetail;
