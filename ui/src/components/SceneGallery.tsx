import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardMedia,
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

interface Scene {
  video_id: string;
  title: string;
  screenshot: string;
  duration: number;
  tags: string[];
  is_processed?: boolean;
}

interface SceneGalleryProps {
  onSceneSelect: (sceneId: string) => void;
  selectedSceneId?: string;
  selectedSceneIds?: string[];
  onMultiSelect?: (sceneIds: string[]) => void;
  multiSelect?: boolean;
  limit?: number;
  showProcessed?: boolean;
  onSelectAll?: (sceneIds: string[]) => void;
}

const SceneGallery: React.FC<SceneGalleryProps> = ({
  onSceneSelect,
  selectedSceneId,
  selectedSceneIds = [],
  onMultiSelect,
  multiSelect = false,
  limit = 20,
  showProcessed = true,
  onSelectAll,
}) => {
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set(selectedSceneIds));
  const [page, setPage] = useState(1);
  const [totalScenes, setTotalScenes] = useState(0);
  
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:9898';
  
  // Update selectedIds when selectedSceneIds prop changes
  React.useEffect(() => {
    setSelectedIds(new Set(selectedSceneIds));
  }, [selectedSceneIds]);

  useEffect(() => {
    const fetchScenes = async () => {
      try {
        setLoading(true);
        const offset = (page - 1) * limit;
        const response = await fetch(`${apiUrl}/videos?limit=${limit}&offset=${offset}`);
        if (!response.ok) {
          throw new Error('Failed to fetch videos');
        }
        const data = await response.json();
        setScenes(data.videos || []);
        setTotalScenes(data.total || data.videos?.length || 0);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchScenes();
  }, [limit, apiUrl, page]);

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const filteredScenes = scenes.filter((scene) =>
    scene.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (scene.tags || []).some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  );
  
  const handleSceneClick = (sceneId: string) => {
    if (multiSelect && onMultiSelect) {
      const newSelected = new Set(selectedIds);
      if (newSelected.has(sceneId)) {
        newSelected.delete(sceneId);
      } else {
        newSelected.add(sceneId);
      }
      setSelectedIds(newSelected);
      onMultiSelect(Array.from(newSelected));
    } else {
      onSceneSelect(sceneId);
    }
  };
  
  const handleCheckboxClick = (e: React.MouseEvent, sceneId: string) => {
    e.stopPropagation();
    if (multiSelect && onMultiSelect) {
      const newSelected = new Set(selectedIds);
      if (newSelected.has(sceneId)) {
        newSelected.delete(sceneId);
      } else {
        newSelected.add(sceneId);
      }
      setSelectedIds(newSelected);
      onMultiSelect(Array.from(newSelected));
    }
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
      const allSceneIds = filteredScenes.map(s => s.video_id);
      setSelectedIds(new Set(allSceneIds));
      onMultiSelect(allSceneIds);
      if (onSelectAll) {
        onSelectAll(allSceneIds);
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
        {multiSelect && filteredScenes.length > 0 && (
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

      {filteredScenes.length === 0 ? (
        <Alert severity="info">No videos found</Alert>
      ) : (
        <Grid container spacing={2}>
          {filteredScenes.map((scene) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={scene.video_id}>
              <Card
                sx={{
                  cursor: 'pointer',
                  position: 'relative',
                  border: (multiSelect ? selectedIds.has(scene.video_id) : selectedSceneId === scene.video_id) ? '3px solid #1976d2' : '1px solid #333',
                  transition: 'all 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 6,
                    '& .play-icon': {
                      opacity: 1,
                    },
                  },
                }}
                onClick={() => handleSceneClick(scene.video_id)}
              >
                <Box position="relative">
                  {multiSelect && (
                    <Checkbox
                      checked={selectedIds.has(scene.video_id)}
                      onClick={(e) => handleCheckboxClick(e, scene.video_id)}
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
                  <CardMedia
                    component="img"
                    height="200"
                    image={`${apiUrl}${scene.screenshot}`}
                    alt={scene.title}
                    sx={{ objectFit: 'cover' }}
                  />
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
                  {scene.duration > 0 && (
                  <Chip
                    label={formatDuration(scene.duration)}
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
                    title={scene.title}
                    sx={{ fontWeight: 600, mb: 0.5 }}
                  >
                    {scene.title}
                  </Typography>
                  
                  <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 0.5 }}>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                    >
                      Video ID: {scene.video_id}
                    </Typography>
                    {scene.is_processed && showProcessed && (
                      <Chip
                        label="Processed"
                        size="small"
                        sx={{ fontSize: '0.65rem', height: 18, bgcolor: 'success.main', color: 'white' }}
                      />
                    )}
                  </Stack>

                  {(scene.tags || []).length > 0 && (
                    <Stack direction="row" spacing={0.5} flexWrap="wrap" gap={0.5} sx={{ mt: 0.5 }}>
                      <Chip
                        label={`${scene.tags.length} tag${scene.tags.length !== 1 ? 's' : ''}`}
                        size="small"
                        sx={{ fontSize: '0.7rem', height: 20, bgcolor: 'primary.main', color: 'white' }}
                      />
                      {scene.tags.slice(0, 2).map((tag, index) => (
                        <Chip
                          key={index}
                          label={tag}
                          size="small"
                          sx={{ fontSize: '0.7rem', height: 20 }}
                        />
                      ))}
                      {scene.tags.length > 2 && (
                        <Chip
                          label={`+${scene.tags.length - 2}`}
                          size="small"
                          sx={{ fontSize: '0.7rem', height: 20 }}
                        />
                      )}
                    </Stack>
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Pagination */}
      {!loading && filteredScenes.length > 0 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4, mb: 2 }}>
          <Pagination
            count={Math.ceil(totalScenes / limit)}
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

export default SceneGallery;
