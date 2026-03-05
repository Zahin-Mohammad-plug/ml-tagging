import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  CircularProgress,
  Grid,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
} from '@mui/material';
import { Save, Settings as SettingsIcon } from '@mui/icons-material';
import axios from 'axios';

interface SettingsData {
  api_url?: string;
  auto_approve?: boolean;
  confidence_threshold?: number;
  sample_fps?: number;
  max_frames_per_scene?: number;
  min_agreement_frames?: number;
  temporal_window_seconds?: number;
  vision_weight?: number;
  asr_weight?: number;
  ocr_weight?: number;
  default_review_threshold?: number;
  default_auto_threshold?: number;
  vision_model?: string;
  asr_model?: string;
  ocr_engine?: string;
  clip_model_name?: string;
  use_calibrated_confidence?: boolean;
  cache_frames?: boolean;
  max_concurrent_jobs?: number;
  job_timeout_seconds?: number;
}

const Settings: React.FC = () => {
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8888';
  
  const [settings, setSettings] = useState<SettingsData>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get(`${apiUrl}/settings`);
      setSettings(response.data);
    } catch (err: any) {
      console.error('Failed to fetch settings:', err);
      setError('Failed to load settings. Using defaults.');
      // Set defaults on error
      setSettings({
        auto_approve: false,
        confidence_threshold: 0.7,
        sample_fps: 0.5,
        max_frames_per_scene: 100,
        min_agreement_frames: 3,
        temporal_window_seconds: 10,
        vision_weight: 0.7,
        asr_weight: 0.2,
        ocr_weight: 0.1,
        default_review_threshold: 0.3,
        default_auto_threshold: 0.8,
        use_calibrated_confidence: true,
        cache_frames: true,
        max_concurrent_jobs: 3,
        job_timeout_seconds: 1800,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      setError(null);
      // Filter out undefined values and ensure all settings are sent
      const settingsToSave: Partial<SettingsData> = {};
      Object.keys(settings).forEach((key) => {
        const value = settings[key as keyof SettingsData];
        if (value !== undefined && value !== null) {
          (settingsToSave as any)[key] = value;
        }
      });
      await axios.put(`${apiUrl}/settings`, settingsToSave);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
      // Refresh settings to ensure we have the latest from server
      await fetchSettings();
    } catch (err: any) {
      console.error('Failed to save settings:', err);
      setError(err.response?.data?.detail || 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const updateSetting = (key: keyof SettingsData, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  if (loading) {
    return (
      <Container maxWidth="md" sx={{ mt: 4, mb: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <SettingsIcon />
        Settings
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {saved && (
        <Alert severity="success" sx={{ mb: 3 }} onClose={() => setSaved(false)}>
          Settings saved successfully! Changes are now persistent.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* API Configuration */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              API Configuration
            </Typography>
            <Box sx={{ mt: 2 }}>
              <TextField
                fullWidth
                label="API URL"
                value={apiUrl}
                disabled
                helperText="API URL (read-only, set via environment variable)"
                sx={{ mb: 2 }}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Suggestion Behavior */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Suggestion Behavior
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Control how suggestions are generated and displayed
            </Typography>
            <Box sx={{ mt: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.auto_approve || false}
                    onChange={(e) => updateSetting('auto_approve', e.target.checked)}
                  />
                }
                label="Auto-approve high confidence suggestions"
              />
              <Typography variant="caption" display="block" color="text.secondary" sx={{ ml: 4, mt: 0.5, mb: 2 }}>
                Automatically apply tags with confidence above the auto threshold (see Default Thresholds section)
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* ML Processing Settings */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              ML Processing Settings
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Configure how videos are processed and analyzed
            </Typography>
            <Box sx={{ mt: 2 }}>
              <TextField
                fullWidth
                type="number"
                label="Sample FPS"
                value={settings.sample_fps || 0.5}
                onChange={(e) => updateSetting('sample_fps', parseFloat(e.target.value))}
                inputProps={{ min: 0.1, max: 10, step: 0.1 }}
                helperText="Frames per second to sample from video"
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                type="number"
                label="Max Frames Per Scene"
                value={settings.max_frames_per_scene || 100}
                onChange={(e) => updateSetting('max_frames_per_scene', parseInt(e.target.value))}
                inputProps={{ min: 1, max: 1000 }}
                helperText="Maximum number of frames to process per scene"
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                type="number"
                label="Min Agreement Frames"
                value={settings.min_agreement_frames || 3}
                onChange={(e) => updateSetting('min_agreement_frames', parseInt(e.target.value))}
                inputProps={{ min: 1, max: 50 }}
                helperText="Minimum number of frames that must agree for a tag"
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                type="number"
                label="Temporal Window (seconds)"
                value={settings.temporal_window_seconds || 10}
                onChange={(e) => updateSetting('temporal_window_seconds', parseInt(e.target.value))}
                inputProps={{ min: 1, max: 60 }}
                helperText="Time window for temporal consistency analysis"
              />
            </Box>
          </Paper>
        </Grid>

        {/* Signal Weights */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Signal Weights
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Configure how different ML signals are combined (should sum to ~1.0)
            </Typography>
            <Box sx={{ mt: 2 }}>
              <TextField
                fullWidth
                type="number"
                label="Vision Weight"
                value={settings.vision_weight || 0.7}
                onChange={(e) => updateSetting('vision_weight', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                InputProps={{
                  endAdornment: <InputAdornment position="end">{(settings.vision_weight || 0.7) * 100}%</InputAdornment>
                }}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                type="number"
                label="ASR Weight"
                value={settings.asr_weight || 0.2}
                onChange={(e) => updateSetting('asr_weight', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                InputProps={{
                  endAdornment: <InputAdornment position="end">{(settings.asr_weight || 0.2) * 100}%</InputAdornment>
                }}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                type="number"
                label="OCR Weight"
                value={settings.ocr_weight || 0.1}
                onChange={(e) => updateSetting('ocr_weight', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                InputProps={{
                  endAdornment: <InputAdornment position="end">{(settings.ocr_weight || 0.1) * 100}%</InputAdornment>
                }}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Thresholds */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Default Thresholds
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Configure confidence thresholds for review and auto-approval
            </Typography>
            <Box sx={{ mt: 2 }}>
              <TextField
                fullWidth
                type="number"
                label="Default Review Threshold"
                value={settings.default_review_threshold ?? 0.3}
                onChange={(e) => updateSetting('default_review_threshold', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.05 }}
                helperText="Default minimum confidence for Review page display filter (0.0-1.0). This is a display filter only - all suggestions are stored in the database regardless of confidence."
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                type="number"
                label="Default Auto Threshold"
                value={settings.default_auto_threshold ?? 0.8}
                onChange={(e) => updateSetting('default_auto_threshold', parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.05 }}
                helperText="Minimum confidence for auto-approving suggestions (0.0-1.0). Only applies when auto-approve is enabled above."
              />
            </Box>
          </Paper>
        </Grid>

        {/* Advanced Settings */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Advanced Settings
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Advanced configuration options for power users
            </Typography>
            <Box sx={{ mt: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.use_calibrated_confidence !== false}
                    onChange={(e) => updateSetting('use_calibrated_confidence', e.target.checked)}
                  />
                }
                label="Use Calibrated Confidence"
                sx={{ mb: 2, display: 'block' }}
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.cache_frames !== false}
                    onChange={(e) => updateSetting('cache_frames', e.target.checked)}
                  />
                }
                label="Cache Frames"
                sx={{ mb: 2, display: 'block' }}
              />
              <TextField
                fullWidth
                type="number"
                label="Max Concurrent Jobs"
                value={settings.max_concurrent_jobs || 3}
                onChange={(e) => updateSetting('max_concurrent_jobs', parseInt(e.target.value))}
                inputProps={{ min: 1, max: 10 }}
                helperText="Maximum number of jobs to process simultaneously"
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                type="number"
                label="Job Timeout (seconds)"
                value={settings.job_timeout_seconds || 1800}
                onChange={(e) => updateSetting('job_timeout_seconds', parseInt(e.target.value))}
                inputProps={{ min: 60, max: 3600 }}
                helperText="Maximum time for a job to complete"
                sx={{ mb: 2 }}
              />
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>CLIP Model</InputLabel>
                <Select
                  value={settings.clip_model_name || 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K'}
                  onChange={(e) => updateSetting('clip_model_name', e.target.value)}
                  label="CLIP Model"
                >
                  <MenuItem value="laion/CLIP-ViT-L-14-laion2B-s32B-b82K">
                    LAION CLIP ViT-L/14 (768-dim, Current)
                  </MenuItem>
                  <MenuItem value="laion/CLIP-ViT-H-14-laion2B-s32B-b79K">
                    LAION CLIP ViT-H/14 (1024-dim, Larger)
                  </MenuItem>
                  <MenuItem value="google/siglip-so400m-patch14-384">
                    Google SigLIP So400m (1152-dim, Best Performance)
                  </MenuItem>
                </Select>
                <FormHelperText>
                  Changing model requires restarting workers. New embeddings will use selected model.
                </FormHelperText>
              </FormControl>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          color="primary"
          startIcon={saving ? <CircularProgress size={20} /> : <Save />}
          onClick={handleSave}
          disabled={saving}
          size="large"
        >
          {saving ? 'Saving...' : 'Save Settings'}
        </Button>
      </Box>
    </Container>
  );
};

export default Settings;
