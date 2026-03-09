import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardMedia,
  CardContent,
  Typography,
  Grid,
  TextField,
  Button,
  Chip,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton
} from '@mui/material';
import { Search, Close, PlayArrow } from '@mui/icons-material';
import axios from 'axios';

interface Video {
  id: string;
  title: string;
  path: string;
  duration: number;
  tags: string[];
}

interface VideoBrowserProps {
  onSelectVideo: (videoId: string, videoTitle: string) => void;
  open: boolean;
  onClose: () => void;
}

const VideoBrowser: React.FC<VideoBrowserProps> = ({ onSelectVideo, open, onClose }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:9898';

  // Fetch recent videos on open
  useEffect(() => {
    if (open) {
      fetchRecentVideos();
    }
  }, [open]);

  const fetchRecentVideos = async () => {
    setLoading(true);
    setError(null);
    try {
      setVideos([]);
    } catch (err: any) {
      setError('Failed to fetch videos');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${apiUrl}/videos/${searchQuery}`);
      const video = response.data;
      
      setVideos([{
        id: video.id,
        title: video.title || `Video ${video.id}`,
        path: video.path,
        duration: video.duration,
        tags: video.tags || []
      }]);
    } catch (err: any) {
      setError(`Video not found: ${searchQuery}`);
      setVideos([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectVideo = (video: Video) => {
    setSelectedVideoId(video.id);
    onSelectVideo(video.id, video.title);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">Browse Videos</Typography>
          <IconButton onClick={onClose}>
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
          <TextField
            fullWidth
            placeholder="Enter Video ID (e.g., 2195)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          />
          <Button
            variant="contained"
            onClick={handleSearch}
            startIcon={<Search />}
            disabled={loading}
          >
            Search
          </Button>
        </Box>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {!loading && videos.length === 0 && !error && (
          <Alert severity="info">
            Enter a video ID to preview and select for processing.
          </Alert>
        )}

        <Grid container spacing={2}>
          {videos.map((video) => (
            <Grid item xs={12} key={video.id}>
              <Card 
                sx={{ 
                  cursor: 'pointer',
                  border: selectedVideoId === video.id ? '2px solid' : '1px solid #333',
                  borderColor: selectedVideoId === video.id ? 'primary.main' : '#333',
                  '&:hover': { 
                    borderColor: 'primary.light',
                    transform: 'translateY(-2px)',
                    transition: 'all 0.2s'
                  }
                }}
                onClick={() => handleSelectVideo(video)}
              >
                <Box sx={{ display: 'flex' }}>
                  <CardMedia
                    component="img"
                    sx={{ width: 200, height: 120, objectFit: 'cover' }}
                    image={video.path}
                    alt={video.title}
                  />
                  <CardContent sx={{ flex: 1 }}>
                    <Typography variant="h6" gutterBottom>
                      {video.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      ID: {video.id} • Duration: {Math.floor(video.duration / 60)}:{String(Math.floor(video.duration % 60)).padStart(2, '0')}
                    </Typography>
                    {video.tags.length > 0 && (
                      <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                        {video.tags.slice(0, 5).map((tag, idx) => (
                          <Chip key={idx} label={tag} size="small" />
                        ))}
                        {video.tags.length > 5 && (
                          <Chip label={`+${video.tags.length - 5} more`} size="small" />
                        )}
                      </Box>
                    )}
                  </CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', pr: 2 }}>
                    <Button
                      variant="contained"
                      startIcon={<PlayArrow />}
                      onClick={() => handleSelectVideo(video)}
                    >
                      Select
                    </Button>
                  </Box>
                </Box>
              </Card>
            </Grid>
          ))}
        </Grid>
      </DialogContent>
    </Dialog>
  );
};

export default VideoBrowser;
