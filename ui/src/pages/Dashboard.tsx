import React, { useEffect, useState, useCallback } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Card,
  CardContent,
} from '@mui/material';
import {
  CheckCircle,
  HourglassEmpty,
  Cancel,
  TrendingUp,
} from '@mui/icons-material';
import axios from 'axios';

interface Stats {
  suggestions?: {
    pending_count?: number;
    approved_count?: number;
    rejected_count?: number;
    total_suggestions?: number;
  };
  jobs?: {
    status_breakdown?: {
      queued?: number;
      sampling?: number;
      embeddings?: number;
      asr_ocr?: number;
      fusion?: number;
      completed?: number;
      failed?: number;
      cancelled?: number;
    };
  };
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<Stats>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:9898';

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${apiUrl}/stats`);
      setStats(response.data);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
      setError('Failed to load statistics. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

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

  const StatCard: React.FC<{ title: string; value: number; icon: React.ReactNode; color: string }> = 
    ({ title, value, icon, color }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box>
            <Typography variant="h4" color={color}>
              {value || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {title}
            </Typography>
          </Box>
          <Box sx={{ color, opacity: 0.7 }}>
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Overview of ML tagging suggestions and processing jobs
      </Typography>

      <Grid container spacing={3}>
        {/* Suggestions Stats */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
            Tag Suggestions
          </Typography>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Pending Review"
            value={stats.suggestions?.pending_count || 0}
            icon={<HourglassEmpty sx={{ fontSize: 40 }} />}
            color="#ff9800"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Approved"
            value={stats.suggestions?.approved_count || 0}
            icon={<CheckCircle sx={{ fontSize: 40 }} />}
            color="#4caf50"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Rejected"
            value={stats.suggestions?.rejected_count || 0}
            icon={<Cancel sx={{ fontSize: 40 }} />}
            color="#f44336"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Suggestions"
            value={stats.suggestions?.total_suggestions || 0}
            icon={<TrendingUp sx={{ fontSize: 40 }} />}
            color="#2196f3"
          />
        </Grid>

        {/* Processing Jobs Stats */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Processing Jobs
          </Typography>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Queued"
            value={stats.jobs?.status_breakdown?.queued || 0}
            icon={<HourglassEmpty sx={{ fontSize: 40 }} />}
            color="#9e9e9e"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Processing"
            value={(stats.jobs?.status_breakdown?.sampling || 0) + (stats.jobs?.status_breakdown?.embeddings || 0) + (stats.jobs?.status_breakdown?.asr_ocr || 0) + (stats.jobs?.status_breakdown?.fusion || 0)}
            icon={<CircularProgress size={40} />}
            color="#2196f3"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Completed"
            value={stats.jobs?.status_breakdown?.completed || 0}
            icon={<CheckCircle sx={{ fontSize: 40 }} />}
            color="#4caf50"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Failed"
            value={stats.jobs?.status_breakdown?.failed || 0}
            icon={<Cancel sx={{ fontSize: 40 }} />}
            color="#f44336"
          />
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              Quick Start
            </Typography>
            <Typography variant="body2" color="text.secondary">
              1. Process a video via the "Process Video" page<br />
              2. Wait for the pipeline to finish<br />
              3. Come back here to review suggestions<br />
              4. Approve or reject suggested tags
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
