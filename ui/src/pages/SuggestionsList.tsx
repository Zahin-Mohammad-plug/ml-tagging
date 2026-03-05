import React, { useEffect, useState, useCallback } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Chip,
  Button,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import { CheckCircle, Cancel, Visibility } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

interface Suggestion {
  id: string;
  scene_id: string;
  tag_name: string;
  confidence: number;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  scene_title?: string;
}

const SuggestionsList: React.FC = () => {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'pending' | 'approved' | 'rejected'>('pending');
  const navigate = useNavigate();

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8888';

  const fetchSuggestions = useCallback(async () => {
    try {
      setLoading(true);
      const params = filter !== 'all' ? { status: filter } : {};
      const response = await axios.get(`${apiUrl}/suggestions`, { params });
      setSuggestions(response.data);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch suggestions:', err);
      setError('Failed to load suggestions. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  }, [apiUrl, filter]);

  useEffect(() => {
    fetchSuggestions();
  }, [fetchSuggestions]);

  const handleApprove = async (id: string) => {
    try {
      await axios.post(`${apiUrl}/suggestions/${id}/approve`, {
        approved_by: 'ui_user',
        notes: 'Approved from UI'
      });
      fetchSuggestions();
    } catch (err) {
      console.error('Failed to approve suggestion:', err);
      alert('Failed to approve suggestion');
    }
  };

  const handleReject = async (id: string) => {
    try {
      await axios.post(`${apiUrl}/suggestions/${id}/reject`, {
        approved_by: 'ui_user',
        notes: 'Rejected from UI'
      });
      fetchSuggestions();
    } catch (err) {
      console.error('Failed to reject suggestion:', err);
      alert('Failed to reject suggestion');
    }
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return 'success.main';
    if (confidence >= 0.6) return 'warning.main';
    return 'error.main';
  };

  const getStatusColor = (status: string): 'success' | 'warning' | 'error' => {
    if (status === 'approved') return 'success';
    if (status === 'pending') return 'warning';
    return 'error';
  };

  if (loading) {
    return (
      <Container sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
        <CircularProgress />
      </Container>
    );
  }

  if (error) {
    return (
      <Container sx={{ mt: 4 }}>
        <Paper sx={{ p: 3, bgcolor: 'error.dark' }}>
          <Typography variant="h6" color="error.light">
            {error}
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            Check that the ML Tagger API is running on {apiUrl}
          </Typography>
        </Paper>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Tag Suggestions
        </Typography>
        
        <FormControl sx={{ minWidth: 150 }}>
          <InputLabel>Filter</InputLabel>
          <Select
            value={filter}
            label="Filter"
            onChange={(e) => setFilter(e.target.value as any)}
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="pending">Pending</MenuItem>
            <MenuItem value="approved">Approved</MenuItem>
            <MenuItem value="rejected">Rejected</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {suggestions.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" color="text.secondary">
            No suggestions found
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Process some videos to generate tag suggestions
          </Typography>
        </Paper>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Scene</TableCell>
                <TableCell>Tag</TableCell>
                <TableCell>Confidence</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Created</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {suggestions.map((suggestion) => (
                <TableRow key={suggestion.id} hover>
                  <TableCell>
                    <Typography variant="body2">
                      {suggestion.scene_title || `Scene ${suggestion.scene_id}`}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={suggestion.tag_name}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography
                      variant="body2"
                      sx={{ color: getConfidenceColor(suggestion.confidence) }}
                    >
                      {(suggestion.confidence * 100).toFixed(1)}%
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={suggestion.status}
                      size="small"
                      color={getStatusColor(suggestion.status)}
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" color="text.secondary">
                      {new Date(suggestion.created_at).toLocaleDateString()}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                      <Button
                        size="small"
                        variant="outlined"
                        startIcon={<Visibility />}
                        onClick={() => navigate(`/suggestions/${suggestion.id}`)}
                      >
                        View
                      </Button>
                      {suggestion.status === 'pending' && (
                        <>
                          <Button
                            size="small"
                            variant="contained"
                            color="success"
                            startIcon={<CheckCircle />}
                            onClick={() => handleApprove(suggestion.id)}
                          >
                            Approve
                          </Button>
                          <Button
                            size="small"
                            variant="contained"
                            color="error"
                            startIcon={<Cancel />}
                            onClick={() => handleReject(suggestion.id)}
                          >
                            Reject
                          </Button>
                        </>
                      )}
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Container>
  );
};

export default SuggestionsList;
