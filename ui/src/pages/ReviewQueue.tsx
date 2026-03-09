import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  CardMedia,
  Grid,
  Chip,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Alert,
  LinearProgress,
  Tooltip,
  Checkbox,
  FormControlLabel,
  Pagination,
  Stack,
  Slider,
  TextField,
  RadioGroup,
  Radio,
  FormLabel,
} from '@mui/material';
import { CheckCircle, Cancel, Visibility, FilterAlt, Delete, ArrowUpward, ArrowDownward, DeleteSweep } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'react-hot-toast';

interface SuggestionItem {
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
  status: 'pending' | 'approved' | 'rejected' | 'auto_applied';
  evidence_frames: Array<{
    frame_number: number;
    timestamp_seconds: number;
    confidence: number;
    thumbnail_url?: string;
    signals?: {
      frame_id?: string;
      file_path?: string;
    };
  }>;
  created_at: string;
}

interface VideoOption {
  video_id: string;
  title: string;
}

const ReviewQueue: React.FC = () => {
  const [suggestions, setSuggestions] = useState<SuggestionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState<string>('pending');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.5);
  const [videoFilter, setVideoFilter] = useState<string>('all');
  const [availableVideos, setAvailableVideos] = useState<VideoOption[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [sortBy, setSortBy] = useState<string>('confidence');
  const [sortOrder, setSortOrder] = useState<string>('desc');
  const [page, setPage] = useState<number>(1);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [itemsPerPage, setItemsPerPage] = useState<number>(20);
  const [frameIndices, setFrameIndices] = useState<Map<string, number>>(new Map());
  const [hoverTimers, setHoverTimers] = useState<Map<string, NodeJS.Timeout>>(new Map());
  const hoverTimersRef = useRef<Map<string, NodeJS.Timeout>>(new Map());
  const [deletingAll, setDeletingAll] = useState(false);
  const navigate = useNavigate();

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:9898';

  useEffect(() => {
    // Load default review threshold from settings
    const loadSettings = async () => {
      try {
        const response = await axios.get(`${apiUrl}/settings`);
        const defaultThreshold = response.data.default_review_threshold || 0.3;
        setConfidenceThreshold(defaultThreshold);
        // Save to localStorage for persistence
        localStorage.setItem('confidenceThreshold', defaultThreshold.toString());
      } catch (err) {
        // Fallback to localStorage if settings fail
        const saved = localStorage.getItem('confidenceThreshold');
        if (saved) {
          setConfidenceThreshold(parseFloat(saved));
        }
      }
    };
    loadSettings();
    
    // Load items per page from localStorage
    const savedItemsPerPage = localStorage.getItem('reviewItemsPerPage');
    if (savedItemsPerPage) {
      setItemsPerPage(parseInt(savedItemsPerPage, 10));
    }
    
    // Load available videos for filter
    const loadVideos = async () => {
      try {
        const response = await axios.get(`${apiUrl}/suggestions/videos`);
        const videos = response.data || [];
        console.log('Loaded videos for filter:', videos.length, videos);
        setAvailableVideos(videos);
      } catch (err) {
        console.error('Failed to load videos for filter:', err);
      }
    };
    loadVideos();
  }, [apiUrl]);

  const fetchSuggestions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const offset = (page - 1) * itemsPerPage;
      const params: any = { 
        limit: itemsPerPage,
        offset: offset
      };
      if (statusFilter !== 'all') {
        params.status = statusFilter;
      }
      
      if (videoFilter !== 'all') {
        params.video_id = videoFilter;
      }
      
      // Use min_confidence as a display filter (not a processing filter)
      // This filters what's shown, but all suggestions are stored in the database
      params.min_confidence = confidenceThreshold;
      
      // Add sorting parameters
      if (sortBy) {
        params.sort_by = sortBy;
        params.sort_order = sortOrder;
      }
      
      const response = await axios.get(`${apiUrl}/suggestions`, { params });
      
      setSuggestions(response.data);
      
      // Estimate total count - if we get fewer items than requested, we're at the end
      if (response.data.length < itemsPerPage) {
        setTotalCount(offset + response.data.length);
      } else {
        // Estimate: assume there are more pages
        setTotalCount(offset + response.data.length + 1);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load suggestions');
    } finally {
      setLoading(false);
    }
  }, [apiUrl, statusFilter, videoFilter, confidenceThreshold, sortBy, sortOrder, page, itemsPerPage]);

  useEffect(() => {
    fetchSuggestions();
  }, [fetchSuggestions]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      // Cleanup all active timers when component unmounts
      hoverTimersRef.current.forEach(timer => {
        if (timer) clearInterval(timer);
      });
      hoverTimersRef.current.clear();
    };
  }, []); // Only run on mount/unmount

  const handleQuickApprove = async (suggestionId: string) => {
    // Optimistically update the UI
    setSuggestions(prev => prev.map(s => 
      s.id === suggestionId ? { ...s, status: 'approved' as const } : s
    ));
    
    // Remove from selected if it was selected
    setSelectedIds(prev => {
      const newSet = new Set(prev);
      newSet.delete(suggestionId);
      return newSet;
    });
    
    try {
      await axios.post(`${apiUrl}/suggestions/${suggestionId}/approve`, {
        approved_by: 'ui_user',
        notes: 'Quick approved from review queue'
      });
      
      // If filtering by pending, remove from list; otherwise just update status
      if (statusFilter === 'pending') {
        setSuggestions(prev => prev.filter(s => s.id !== suggestionId));
        // Update total count
        setTotalCount(prev => Math.max(0, prev - 1));
      }
      
      toast.success('Tag approved and applied to video!');
    } catch (err: any) {
      // Revert optimistic update on error
      fetchSuggestions();
      toast.error(err.response?.data?.detail || 'Failed to approve');
    }
  };

  const handleQuickReject = async (suggestionId: string) => {
    // Optimistically update the UI
    setSuggestions(prev => prev.map(s => 
      s.id === suggestionId ? { ...s, status: 'rejected' as const } : s
    ));
    
    // Remove from selected if it was selected
    setSelectedIds(prev => {
      const newSet = new Set(prev);
      newSet.delete(suggestionId);
      return newSet;
    });
    
    try {
      await axios.post(`${apiUrl}/suggestions/${suggestionId}/reject`, {
        approved_by: 'ui_user',
        notes: 'Quick rejected from review queue'
      });
      
      // If filtering by pending, remove from list; otherwise just update status
      if (statusFilter === 'pending') {
        setSuggestions(prev => prev.filter(s => s.id !== suggestionId));
        // Update total count
        setTotalCount(prev => Math.max(0, prev - 1));
      }
      
      toast.success('Suggestion rejected');
    } catch (err: any) {
      // Revert optimistic update on error
      fetchSuggestions();
      toast.error(err.response?.data?.detail || 'Failed to reject');
    }
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedIds(new Set(suggestions.map(s => s.id)));
    } else {
      setSelectedIds(new Set());
    }
  };

  const handleSelectItem = (suggestionId: string, checked: boolean) => {
    const newSelected = new Set(selectedIds);
    if (checked) {
      newSelected.add(suggestionId);
    } else {
      newSelected.delete(suggestionId);
    }
    setSelectedIds(newSelected);
  };

  const handleBulkDelete = async () => {
    if (selectedIds.size === 0) {
      toast.error('No suggestions selected');
      return;
    }

    if (!window.confirm(`Are you sure you want to delete ${selectedIds.size} suggestion(s)? This action cannot be undone.`)) {
      return;
    }

    try {
      setDeleting(true);
      const response = await axios.delete(`${apiUrl}/suggestions/bulk`, {
        data: Array.from(selectedIds)
      });
      
      toast.success(response.data.message || `Deleted ${selectedIds.size} suggestion(s)`);
      setSelectedIds(new Set());
      fetchSuggestions();
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to delete suggestions';
      toast.error(errorMsg);
      console.error('Bulk delete error:', err);
    } finally {
      setDeleting(false);
    }
  };

  const handleDeleteAll = async () => {
    if (!window.confirm('Are you sure you want to delete ALL suggestions? This action cannot be undone and will delete all suggestions regardless of status.')) {
      return;
    }

    try {
      setDeletingAll(true);
      const response = await axios.delete(`${apiUrl}/suggestions/all`);
      
      toast.success(response.data.message || `Deleted all ${response.data.deleted_count || 0} suggestion(s)`);
      setSelectedIds(new Set());
      setPage(1);
      fetchSuggestions();
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to delete all suggestions';
      toast.error(errorMsg);
      console.error('Delete all error:', err);
    } finally {
      setDeletingAll(false);
    }
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.9) return '#4caf50'; // green
    if (confidence >= 0.7) return '#2196f3'; // blue
    if (confidence >= 0.5) return '#ff9800'; // orange
    return '#f44336'; // red
  };

  const getCurrentFrameIndex = (suggestionId: string): number => {
    return frameIndices.get(suggestionId) || 0;
  };

  const handleFrameHover = (suggestionId: string, frameCount: number) => {
    // Clear any existing timer for this suggestion
    const existingTimer = hoverTimersRef.current.get(suggestionId);
    if (existingTimer) {
      clearInterval(existingTimer);
      hoverTimersRef.current.delete(suggestionId);
    }

    // Immediately advance to next frame on hover
    setFrameIndices(prev => {
      const currentIndex = prev.get(suggestionId) || 0;
      const nextIndex = (currentIndex + 1) % frameCount;
      const newMap = new Map(prev);
      newMap.set(suggestionId, nextIndex);
      return newMap;
    });

    // Then continue cycling through frames every 1.5 seconds
    const timer = setInterval(() => {
      setFrameIndices(prev => {
        const currentIndex = prev.get(suggestionId) || 0;
        const nextIndex = (currentIndex + 1) % frameCount;
        const newMap = new Map(prev);
        newMap.set(suggestionId, nextIndex);
        return newMap;
      });
    }, 1500);

    hoverTimersRef.current.set(suggestionId, timer);
    setHoverTimers(new Map(hoverTimersRef.current));
  };

  const handleFrameHoverEnd = (suggestionId: string) => {
    // Clear the timer
    const timer = hoverTimersRef.current.get(suggestionId);
    if (timer) {
      clearInterval(timer);
      hoverTimersRef.current.delete(suggestionId);
      setHoverTimers(new Map(hoverTimersRef.current));
    }

    // Reset to first frame
    setFrameIndices(prev => {
      const newMap = new Map(prev);
      newMap.set(suggestionId, 0);
      return newMap;
    });
  };

  if (loading) {
    return (
      <Container sx={{ mt: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  const allSelected = suggestions.length > 0 && selectedIds.size === suggestions.length;
  const someSelected = selectedIds.size > 0 && selectedIds.size < suggestions.length;

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Review Queue
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Showing {suggestions.length} of {totalCount} suggestions (filtered by {(confidenceThreshold * 100).toFixed(0)}% confidence)
            {selectedIds.size > 0 && ` • ${selectedIds.size} selected`}
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          {selectedIds.size > 0 && (
            <Button
              variant="contained"
              color="error"
              startIcon={<Delete />}
              onClick={handleBulkDelete}
              disabled={deleting}
            >
              Delete Selected ({selectedIds.size})
            </Button>
          )}
          
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Filter by Status</InputLabel>
            <Select
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value);
                setSelectedIds(new Set()); // Clear selection when filter changes
                setPage(1); // Reset to first page
              }}
              label="Filter by Status"
              startAdornment={<FilterAlt sx={{ ml: 1, mr: -1 }} />}
            >
              <MenuItem value="all">All Suggestions</MenuItem>
              <MenuItem value="pending">Pending Review</MenuItem>
              <MenuItem value="approved">Approved</MenuItem>
              <MenuItem value="rejected">Rejected</MenuItem>
              <MenuItem value="auto_applied">Auto Applied</MenuItem>
            </Select>
          </FormControl>

          <FormControl component="fieldset" sx={{ minWidth: 250 }}>
            <FormLabel component="legend">Filter by Video</FormLabel>
            <RadioGroup
              row
              value={videoFilter}
              onChange={(e) => {
                setVideoFilter(e.target.value);
                setSelectedIds(new Set()); // Clear selection when filter changes
                setPage(1); // Reset to first page
              }}
            >
              <FormControlLabel value="all" control={<Radio />} label="All Videos" />
              {availableVideos.map((video) => (
                <FormControlLabel
                  key={video.video_id}
                  value={video.video_id}
                  control={<Radio />}
                  label={`${video.title} (ID: ${video.video_id})`}
                />
              ))}
            </RadioGroup>
          </FormControl>

          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Sort By</InputLabel>
            <Select
              value={sortBy}
              onChange={(e) => {
                setSortBy(e.target.value);
                setPage(1); // Reset to first page
              }}
              label="Sort By"
            >
              <MenuItem value="confidence">Confidence</MenuItem>
              <MenuItem value="date">Date</MenuItem>
              <MenuItem value="video">Video ID</MenuItem>
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Order</InputLabel>
            <Select
              value={sortOrder}
              onChange={(e) => {
                setSortOrder(e.target.value);
                setPage(1); // Reset to first page
              }}
              label="Order"
            >
              <MenuItem value="desc">
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <ArrowDownward fontSize="small" />
                  Descending
                </Box>
              </MenuItem>
              <MenuItem value="asc">
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <ArrowUpward fontSize="small" />
                  Ascending
                </Box>
              </MenuItem>
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Per Page</InputLabel>
            <Select
              value={itemsPerPage}
              onChange={(e) => {
                const newItemsPerPage = parseInt(e.target.value as string, 10);
                setItemsPerPage(newItemsPerPage);
                localStorage.setItem('reviewItemsPerPage', newItemsPerPage.toString());
                setPage(1); // Reset to first page
              }}
              label="Per Page"
            >
              <MenuItem value={10}>10</MenuItem>
              <MenuItem value={20}>20</MenuItem>
              <MenuItem value={50}>50</MenuItem>
              <MenuItem value={100}>100</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      {/* Confidence Threshold Filter */}
      <Box sx={{ mb: 3, p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
          <Typography variant="body2" sx={{ minWidth: 120 }}>
            Min Confidence: {(confidenceThreshold * 100).toFixed(0)}%
          </Typography>
          <Slider
            value={confidenceThreshold}
            onChange={(_, value) => {
              setConfidenceThreshold(value as number);
              localStorage.setItem('confidenceThreshold', (value as number).toString());
              setPage(1); // Reset to first page
            }}
            min={0}
            max={1}
            step={0.05}
            marks={[
              { value: 0, label: '0%' },
              { value: 0.3, label: '30%' },
              { value: 0.5, label: '50%' },
              { value: 0.7, label: '70%' },
              { value: 1, label: '100%' }
            ]}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
            sx={{ flex: 1, maxWidth: 400 }}
          />
          <TextField
            type="number"
            size="small"
            value={confidenceThreshold}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              if (!isNaN(val) && val >= 0 && val <= 1) {
                setConfidenceThreshold(val);
                localStorage.setItem('confidenceThreshold', val.toString());
                setPage(1);
              }
            }}
            inputProps={{ min: 0, max: 1, step: 0.05 }}
            sx={{ width: 80 }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
            Display filter - all suggestions are stored regardless of confidence
          </Typography>
        </Box>
      </Box>

      {suggestions.length > 0 && (
        <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <Checkbox
            checked={allSelected}
            indeterminate={someSelected}
            onChange={(e) => handleSelectAll(e.target.checked)}
          />
          <FormControlLabel
            control={<span />}
            label={
              <Typography variant="body2">
                {allSelected ? 'Deselect All' : 'Select All'}
              </Typography>
            }
            onClick={(e) => {
              e.preventDefault();
              handleSelectAll(!allSelected);
            }}
            sx={{ cursor: 'pointer', userSelect: 'none' }}
          />
        </Box>
      )}

      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

      {suggestions.length === 0 ? (
        <Alert severity="info">
          No suggestions found. Try processing a video or lowering the confidence threshold.
        </Alert>
      ) : (
        <Grid container spacing={3}>
          {suggestions.map((suggestion) => (
            <Grid item xs={12} md={6} key={suggestion.id}>
              <Card 
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column',
                  position: 'relative',
                  border: selectedIds.has(suggestion.id) ? '2px solid' : '1px solid',
                  borderColor: selectedIds.has(suggestion.id) ? 'primary.main' : 'divider'
                }}
              >
                {suggestion.evidence_frames && suggestion.evidence_frames.length > 0 && (() => {
                  const currentFrameIndex = getCurrentFrameIndex(suggestion.id);
                  const currentFrame = suggestion.evidence_frames[currentFrameIndex];
                  const imageUrl = currentFrame.thumbnail_url || 
                    (currentFrame.signals?.frame_id ? `/api/frames/${currentFrame.signals.frame_id}/image` : null);
                  
                  return imageUrl ? (
                    <Box 
                      sx={{ position: 'relative', height: 200, overflow: 'hidden', bgcolor: 'grey.900' }}
                      onMouseEnter={() => {
                        if (suggestion.evidence_frames && suggestion.evidence_frames.length > 1) {
                          handleFrameHover(suggestion.id, suggestion.evidence_frames.length);
                        }
                      }}
                      onMouseLeave={() => {
                        if (suggestion.evidence_frames && suggestion.evidence_frames.length > 1) {
                          handleFrameHoverEnd(suggestion.id);
                        }
                      }}
                    >
                      <Box sx={{ position: 'absolute', top: 8, left: 8, zIndex: 2 }}>
                        <Checkbox
                          checked={selectedIds.has(suggestion.id)}
                          onChange={(e) => handleSelectItem(suggestion.id, e.target.checked)}
                          onClick={(e) => e.stopPropagation()}
                          sx={{
                            bgcolor: 'rgba(255, 255, 255, 0.9)',
                            '&:hover': { bgcolor: 'rgba(255, 255, 255, 1)' }
                          }}
                        />
                      </Box>
                      <CardMedia
                        component="img"
                        height="200"
                        image={imageUrl}
                        alt={`Evidence frame ${currentFrame.frame_number}`}
                        sx={{ 
                          objectFit: 'contain', 
                          width: '100%',
                          height: '100%',
                          transition: 'opacity 0.3s ease-in-out'
                        }}
                        onError={(e: any) => {
                          e.target.style.display = 'none';
                        }}
                      />
                      {suggestion.evidence_frames.length > 1 && (
                        <>
                          <Chip
                            label={`${currentFrameIndex + 1}/${suggestion.evidence_frames.length}`}
                            size="small"
                            sx={{
                              position: 'absolute',
                              top: 8,
                              right: 8,
                              bgcolor: 'rgba(0,0,0,0.7)',
                              color: 'white',
                              zIndex: 2
                            }}
                          />
                          <Chip
                            label={`+${suggestion.evidence_frames.length - 1} more`}
                            size="small"
                            sx={{
                              position: 'absolute',
                              bottom: 8,
                              right: 8,
                              bgcolor: 'rgba(0,0,0,0.7)',
                              color: 'white'
                            }}
                          />
                        </>
                      )}
                      <Typography
                        variant="caption"
                        onClick={() => navigate(`/suggestions/${suggestion.id}?seekTo=${currentFrame.timestamp_seconds}`)}
                        sx={{
                          position: 'absolute',
                          bottom: 8,
                          left: 8,
                          bgcolor: 'rgba(0,0,0,0.7)',
                          color: 'white',
                          px: 1,
                          py: 0.5,
                          borderRadius: 1,
                          cursor: 'pointer',
                          '&:hover': {
                            bgcolor: 'rgba(0,0,0,0.9)',
                            textDecoration: 'underline'
                          }
                        }}
                        title={`Click to jump to ${currentFrame.timestamp_seconds.toFixed(1)}s`}
                      >
                        Frame {currentFrame.frame_number} • {currentFrame.timestamp_seconds.toFixed(1)}s
                      </Typography>
                    </Box>
                  ) : (
                    <Box sx={{ position: 'relative', height: 200, bgcolor: 'grey.900', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Box sx={{ position: 'absolute', top: 8, left: 8, zIndex: 2 }}>
                        <Checkbox
                          checked={selectedIds.has(suggestion.id)}
                          onChange={(e) => handleSelectItem(suggestion.id, e.target.checked)}
                          onClick={(e) => e.stopPropagation()}
                          sx={{
                            bgcolor: 'rgba(255, 255, 255, 0.9)',
                            '&:hover': { bgcolor: 'rgba(255, 255, 255, 1)' }
                          }}
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        No preview available
                      </Typography>
                    </Box>
                  );
                })()}
                
                <CardContent sx={{ flexGrow: 1, position: 'relative' }}>
                  {(!suggestion.evidence_frames || suggestion.evidence_frames.length === 0) && (
                    <Box sx={{ position: 'absolute', top: 8, right: 8, zIndex: 1 }}>
                      <Checkbox
                        checked={selectedIds.has(suggestion.id)}
                        onChange={(e) => handleSelectItem(suggestion.id, e.target.checked)}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </Box>
                  )}
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
                    <Box>
                      <Typography variant="h6" component="div">
                        {suggestion.tag_context.tag_name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Video: {suggestion.video_title || suggestion.video_id}
                      </Typography>
                    </Box>
                    
                    <Tooltip title={`Confidence: ${(suggestion.confidence * 100).toFixed(1)}%`}>
                      <Chip
                        label={`${(suggestion.confidence * 100).toFixed(0)}%`}
                        size="medium"
                        sx={{
                          bgcolor: getConfidenceColor(suggestion.confidence),
                          color: 'white',
                          fontWeight: 'bold'
                        }}
                      />
                    </Tooltip>
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <LinearProgress
                      variant="determinate"
                      value={suggestion.confidence * 100}
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        bgcolor: 'grey.800',
                        '& .MuiLinearProgress-bar': {
                          bgcolor: getConfidenceColor(suggestion.confidence)
                        }
                      }}
                    />
                  </Box>

                  {(suggestion.tag_context.parent_tags && suggestion.tag_context.parent_tags.length > 0) || 
                   (suggestion.tag_context.child_tags && suggestion.tag_context.child_tags.length > 0) ? (
                    <Box sx={{ mb: 2 }}>
                      {suggestion.tag_context.parent_tags && suggestion.tag_context.parent_tags.length > 0 && (
                        <Box sx={{ mb: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            Parent tags:
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
                            {suggestion.tag_context.parent_tags.map((tag, idx) => (
                              <Chip key={idx} label={tag} size="small" variant="outlined" color="primary" />
                            ))}
                          </Box>
                        </Box>
                      )}
                      {suggestion.tag_context.child_tags && suggestion.tag_context.child_tags.length > 0 && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Child tags:
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
                            {suggestion.tag_context.child_tags.map((tag, idx) => (
                              <Chip key={idx} label={tag} size="small" variant="outlined" color="secondary" />
                            ))}
                          </Box>
                        </Box>
                      )}
                    </Box>
                  ) : null}

                  <Typography variant="caption" color="text.secondary" display="block">
                    {suggestion.evidence_frames?.length || 0} evidence frames • 
                    Created {new Date(suggestion.created_at).toLocaleString()}
                  </Typography>
                </CardContent>

                <Box sx={{ p: 2, pt: 0, display: 'flex', gap: 1 }}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Visibility />}
                    onClick={() => navigate(`/suggestions/${suggestion.id}`)}
                  >
                    Review
                  </Button>
                  
                  {suggestion.status === 'pending' && (
                    <>
                      <Button
                        variant="contained"
                        color="success"
                        startIcon={<CheckCircle />}
                        onClick={() => handleQuickApprove(suggestion.id)}
                      >
                        Approve
                      </Button>
                      <Button
                        variant="contained"
                        color="error"
                        startIcon={<Cancel />}
                        onClick={() => handleQuickReject(suggestion.id)}
                      >
                        Reject
                      </Button>
                    </>
                  )}
                  
                  {suggestion.status !== 'pending' && (
                    <Chip
                      label={suggestion.status.replace('_', ' ').toUpperCase()}
                      color={
                        suggestion.status === 'approved' || suggestion.status === 'auto_applied'
                          ? 'success'
                          : 'error'
                      }
                    />
                  )}
                </Box>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Pagination */}
      {suggestions.length > 0 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4, mb: 2 }}>
          <Stack spacing={2} alignItems="center">
            <Pagination
              count={Math.ceil(totalCount / itemsPerPage)}
              page={page}
              onChange={(_, value) => {
                setPage(value);
                setSelectedIds(new Set()); // Clear selection when changing pages
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              color="primary"
              size="large"
              showFirstButton
              showLastButton
            />
            <Typography variant="body2" color="text.secondary">
              Page {page} of {Math.ceil(totalCount / itemsPerPage)} • Showing {suggestions.length} suggestions
            </Typography>
          </Stack>
        </Box>
      )}

      {/* Delete All Button */}
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4, mb: 4 }}>
        <Button
          variant="outlined"
          color="error"
          startIcon={deletingAll ? <CircularProgress size={20} /> : <DeleteSweep />}
          onClick={handleDeleteAll}
          disabled={deletingAll}
          size="large"
        >
          {deletingAll ? 'Deleting All...' : 'Delete All Suggestions'}
        </Button>
      </Box>
    </Container>
  );
};

export default ReviewQueue;
