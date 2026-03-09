import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  CircularProgress,
  Alert,
  TextField,
  InputAdornment,
  Stack,
  Checkbox,
  Pagination,
  Button,
} from '@mui/material';
import { Search as SearchIcon, PlayArrow as PlayIcon } from '@mui/icons-material';

interface Video {
  video_id: string;
  title: string;
  duration?: number;
  file_path?: string;
  created_at?: string;
  is_processed?: boolean;
  thumbnail_url?: string;
  frame_count?: number;
}

interface ScrubFrame {
  frame_id: string;
  frame_number: number;
  timestamp_seconds: number;
  thumbnail_url: string;
}

interface VideoGalleryProps {
  onVideoSelect: (videoId: string) => void;
  selectedVideoId?: string;
  selectedVideoIds?: string[];
  onMultiSelect?: (videoIds: string[]) => void;
  multiSelect?: boolean;
  limit?: number;
  showProcessed?: boolean;
  onSelectAll?: (videoIds: string[]) => void;
}

const VideoGallery: React.FC<VideoGalleryProps> = ({
  onVideoSelect,
  selectedVideoId,
  selectedVideoIds = [],
  onMultiSelect,
  multiSelect = false,
  limit = 20,
  showProcessed = true,
  onSelectAll,
}) => {
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set(selectedVideoIds));
  const [page, setPage] = useState(1);
  const [totalVideos, setTotalVideos] = useState(0);
  // Hover scrub state
  const [scrubFrames, setScrubFrames] = useState<Record<string, ScrubFrame[]>>({});
  const [hoveredVideo, setHoveredVideo] = useState<string | null>(null);
  const [scrubIndex, setScrubIndex] = useState(0);
  
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:9898';
  
  // Update selectedIds when selectedVideoIds prop changes
  React.useEffect(() => {
    setSelectedIds(new Set(selectedVideoIds));
  }, [selectedVideoIds]);

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        setLoading(true);
        const offset = (page - 1) * limit;
        const response = await fetch(`${apiUrl}/videos?limit=${limit}&offset=${offset}`);
        if (!response.ok) {
          throw new Error('Failed to fetch videos');
        }
        const data = await response.json();
        setVideos(data.videos || []);
        setTotalVideos(data.total || data.videos?.length || 0);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchVideos();
  }, [limit, apiUrl, page]);

  // Fetch scrub frames on hover (lazy-load per video)
  const fetchScrubFrames = useCallback(async (videoId: string) => {
    if (scrubFrames[videoId]) return; // Already loaded
    try {
      const response = await fetch(`${apiUrl}/videos/${encodeURIComponent(videoId)}/frames?limit=10`);
      if (!response.ok) return;
      const data = await response.json();
      setScrubFrames(prev => ({ ...prev, [videoId]: data.frames || [] }));
    } catch {
      // Silently fail — thumbnails will still show the static thumbnail
    }
  }, [apiUrl, scrubFrames]);

  const handleMouseEnter = useCallback((videoId: string) => {
    setHoveredVideo(videoId);
    setScrubIndex(0);
    fetchScrubFrames(videoId);
  }, [fetchScrubFrames]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>, videoId: string) => {
    const frames = scrubFrames[videoId];
    if (!frames || frames.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, x / rect.width));
    const idx = Math.min(Math.floor(pct * frames.length), frames.length - 1);
    setScrubIndex(idx);
  }, [scrubFrames]);

  const handleMouseLeave = useCallback(() => {
    setHoveredVideo(null);
    setScrubIndex(0);
  }, []);

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const filteredVideos = videos.filter((video) =>
    video.title.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  const handleVideoClick = (videoId: string) => {
    if (multiSelect && onMultiSelect) {
      const newSelected = new Set(selectedIds);
      if (newSelected.has(videoId)) {
        newSelected.delete(videoId);
      } else {
        newSelected.add(videoId);
      }
      setSelectedIds(newSelected);
      onMultiSelect(Array.from(newSelected));
    } else {
      onVideoSelect(videoId);
    }
  };
  
  const handleCheckboxClick = (e: React.MouseEvent, videoId: string) => {
    e.stopPropagation();
    if (multiSelect && onMultiSelect) {
      const newSelected = new Set(selectedIds);
      if (newSelected.has(videoId)) {
        newSelected.delete(videoId);
      } else {
        newSelected.add(videoId);
      }
      setSelectedIds(newSelected);
      onMultiSelect(Array.from(newSelected));
    }
  };

  // Build the thumbnail URL for a given video, accounting for hover scrub
  const getThumbnailUrl = (video: Video): string | null => {
    if (hoveredVideo === video.video_id) {
      const frames = scrubFrames[video.video_id];
      if (frames && frames.length > 0 && scrubIndex < frames.length) {
        return frames[scrubIndex].thumbnail_url;
      }
    }
    return video.thumbnail_url || null;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Error loading videos: {error}
      </Alert>
    );
  }

  const handleSelectAll = () => {
    if (multiSelect && onMultiSelect) {
      const allVideoIds = filteredVideos.map(s => s.video_id);
      setSelectedIds(new Set(allVideoIds));
      onMultiSelect(allVideoIds);
      if (onSelectAll) {
        onSelectAll(allVideoIds);
      }
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', gap: 2, mb: 3, alignItems: 'center' }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Search videos by title or tags..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
        {multiSelect && filteredVideos.length > 0 && (
          <Button
            variant="outlined"
            size="small"
            onClick={handleSelectAll}
            sx={{ whiteSpace: 'nowrap', minWidth: 150 }}
          >
            Select All (Page)
          </Button>
        )}
      </Box>

      {filteredVideos.length === 0 ? (
        <Alert severity="info">No videos found</Alert>
      ) : (
        <Grid container spacing={2}>
          {filteredVideos.map((video) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={video.video_id}>
              <Card
                sx={{
                  cursor: 'pointer',
                  position: 'relative',
                  border: (multiSelect ? selectedIds.has(video.video_id) : selectedVideoId === video.video_id) ? '3px solid #1976d2' : '1px solid #333',
                  transition: 'all 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 6,
                    '& .play-icon': {
                      opacity: 1,
                    },
                  },
                }}
                onClick={() => handleVideoClick(video.video_id)}
              >
                <Box
                  position="relative"
                  onMouseEnter={() => handleMouseEnter(video.video_id)}
                  onMouseMove={(e) => handleMouseMove(e, video.video_id)}
                  onMouseLeave={handleMouseLeave}
                >
                  {multiSelect && (
                    <Checkbox
                      checked={selectedIds.has(video.video_id)}
                      onClick={(e) => handleCheckboxClick(e, video.video_id)}
                      sx={{
                        position: 'absolute',
                        top: 8,
                        left: 8,
                        zIndex: 2,
                        bgcolor: 'rgba(255, 255, 255, 0.9)',
                        '&:hover': { bgcolor: 'rgba(255, 255, 255, 1)' }
                      }}
                    />
                  )}
                  {(() => {
                    const thumbUrl = getThumbnailUrl(video);
                    return thumbUrl ? (
                      <Box sx={{ height: 200, position: 'relative', overflow: 'hidden', bgcolor: 'grey.900' }}>
                        <img
                          src={thumbUrl}
                          alt={video.title}
                          style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            display: 'block',
                          }}
                          onError={(e: any) => {
                            e.target.style.display = 'none';
                          }}
                        />
                        {/* Scrub progress bar */}
                        {hoveredVideo === video.video_id && scrubFrames[video.video_id]?.length > 1 && (
                          <Box sx={{
                            position: 'absolute',
                            bottom: 0,
                            left: 0,
                            right: 0,
                            height: 3,
                            bgcolor: 'rgba(0,0,0,0.4)',
                          }}>
                            <Box sx={{
                              height: '100%',
                              width: `${((scrubIndex + 1) / scrubFrames[video.video_id].length) * 100}%`,
                              bgcolor: 'primary.main',
                              transition: 'width 0.05s linear',
                            }} />
                          </Box>
                        )}
                        {/* Frame count badge */}
                        {video.frame_count && video.frame_count > 0 && (
                          <Chip
                            label={`${video.frame_count} frames`}
                            size="small"
                            sx={{
                              position: 'absolute',
                              top: 8,
                              right: 8,
                              backgroundColor: 'rgba(0, 0, 0, 0.7)',
                              color: 'white',
                              fontSize: '0.7rem',
                              height: 22,
                            }}
                          />
                        )}
                      </Box>
                    ) : (
                      <Box
                        sx={{
                          height: 200,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          bgcolor: 'grey.800',
                        }}
                      >
                        <PlayIcon sx={{ fontSize: 60, color: 'grey.500' }} />
                      </Box>
                    );
                  })()}
                  {/* Play icon overlay */}
                  {!multiSelect && (
                    <Box
                      className="play-icon"
                      sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        opacity: 0,
                        transition: 'opacity 0.2s',
                        backgroundColor: 'rgba(0, 0, 0, 0.6)',
                        borderRadius: '50%',
                        width: 60,
                        height: 60,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      <PlayIcon sx={{ fontSize: 40, color: 'white' }} />
                    </Box>
                  )}
                  
                  {/* Duration badge */}
                  {video.duration && video.duration > 0 && (
                  <Chip
                    label={formatDuration(video.duration)}
                    size="small"
                    sx={{
                      position: 'absolute',
                      bottom: 8,
                      right: 8,
                      backgroundColor: 'rgba(0, 0, 0, 0.8)',
                      color: 'white',
                      fontWeight: 'bold',
                    }}
                  />
                  )}
                </Box>

                <CardContent sx={{ pb: 1 }}>
                  <Typography
                    variant="subtitle2"
                    noWrap
                    title={video.title}
                    sx={{ fontWeight: 600, mb: 0.5 }}
                  >
                    {video.title}
                  </Typography>
                  
                  <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 0.5 }}>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                    >
                      Video ID: {video.video_id}
                    </Typography>
                    {video.is_processed && showProcessed && (
                      <Chip
                        label="Processed"
                        size="small"
                        sx={{ fontSize: '0.65rem', height: 18, bgcolor: 'success.main', color: 'white' }}
                      />
                    )}
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Pagination */}
      {!loading && filteredVideos.length > 0 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4, mb: 2 }}>
          <Pagination
            count={Math.ceil(totalVideos / limit)}
            page={page}
            onChange={(_, value) => {
              setPage(value);
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            color="primary"
            size="large"
            showFirstButton
            showLastButton
          />
        </Box>
      )}
    </Box>
  );
};

export default VideoGallery;
