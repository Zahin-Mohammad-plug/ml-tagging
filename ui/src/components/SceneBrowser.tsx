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

interface Scene {
  id: string;
  title: string;
  path: string;
  duration: number;
  tags: string[];
}

interface SceneBrowserProps {
  onSelectScene: (sceneId: string, sceneTitle: string) => void;
  open: boolean;
  onClose: () => void;
}

const SceneBrowser: React.FC<SceneBrowserProps> = ({ onSelectScene, open, onClose }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8888';

  // Fetch recent scenes on open
  useEffect(() => {
    if (open) {
      fetchRecentScenes();
    }
  }, [open]);

  const fetchRecentScenes = async () => {
    setLoading(true);
    setError(null);
    try {
      // For now, we'll just show instructions
      setScenes([]);
    } catch (err: any) {
      setError('Failed to fetch scenes');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError(null);
    try {
      // Try to fetch the scene by ID
      const response = await axios.get(`${apiUrl}/scenes/${searchQuery}`);
      const scene = response.data;
      
      setScenes([{
        id: scene.id,
        title: scene.title || `Scene ${scene.id}`,
        path: scene.path,
        duration: scene.duration,
        tags: scene.tags || []
      }]);
    } catch (err: any) {
      setError(`Scene not found: ${searchQuery}`);
      setScenes([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectScene = (scene: Scene) => {
    setSelectedSceneId(scene.id);
    onSelectScene(scene.id, scene.title);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">Browse Scenes</Typography>
          <IconButton onClick={onClose}>
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
          <TextField
            fullWidth
            placeholder="Enter Scene ID (e.g., 2195)"
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

        {!loading && scenes.length === 0 && !error && (
          <Alert severity="info">
            Enter a scene ID to preview and select for processing.
          </Alert>
        )}

        <Grid container spacing={2}>
          {scenes.map((scene) => (
            <Grid item xs={12} key={scene.id}>
              <Card 
                sx={{ 
                  cursor: 'pointer',
                  border: selectedSceneId === scene.id ? '2px solid' : '1px solid #333',
                  borderColor: selectedSceneId === scene.id ? 'primary.main' : '#333',
                  '&:hover': { 
                    borderColor: 'primary.light',
                    transform: 'translateY(-2px)',
                    transition: 'all 0.2s'
                  }
                }}
                onClick={() => handleSelectScene(scene)}
              >
                <Box sx={{ display: 'flex' }}>
                  <CardMedia
                    component="img"
                    sx={{ width: 200, height: 120, objectFit: 'cover' }}
                    image={scene.path}
                    alt={scene.title}
                  />
                  <CardContent sx={{ flex: 1 }}>
                    <Typography variant="h6" gutterBottom>
                      {scene.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      ID: {scene.id} • Duration: {Math.floor(scene.duration / 60)}:{String(Math.floor(scene.duration % 60)).padStart(2, '0')}
                    </Typography>
                    {scene.tags.length > 0 && (
                      <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                        {scene.tags.slice(0, 5).map((tag, idx) => (
                          <Chip key={idx} label={tag} size="small" />
                        ))}
                        {scene.tags.length > 5 && (
                          <Chip label={`+${scene.tags.length - 5} more`} size="small" />
                        )}
                      </Box>
                    )}
                  </CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', pr: 2 }}>
                    <Button
                      variant="contained"
                      startIcon={<PlayArrow />}
                      onClick={() => handleSelectScene(scene)}
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

export default SceneBrowser;
