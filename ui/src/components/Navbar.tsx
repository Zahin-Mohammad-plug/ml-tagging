import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { Home, PlayArrow, RateReview, ViewList, Label, Settings as SettingsIcon } from '@mui/icons-material';

const Navbar: React.FC = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 0, mr: 4 }}>
          🎯 ML Tagger
        </Typography>
        
        <Box sx={{ flexGrow: 1, display: 'flex', gap: 2 }}>
          <Button
            color="inherit"
            component={RouterLink}
            to="/"
            startIcon={<Home />}
          >
            Dashboard
          </Button>
          
          <Button
            color="inherit"
            component={RouterLink}
            to="/process"
            startIcon={<PlayArrow />}
          >
            Process
          </Button>
          
          <Button
            color="inherit"
            component={RouterLink}
            to="/review"
            startIcon={<RateReview />}
          >
            Review Queue
          </Button>
          
          <Button
            color="inherit"
            component={RouterLink}
            to="/suggestions"
            startIcon={<ViewList />}
          >
            All Suggestions
          </Button>
          
          <Button
            color="inherit"
            component={RouterLink}
            to="/tags"
            startIcon={<Label />}
          >
            Tags
          </Button>
          
          <Button
            color="inherit"
            component={RouterLink}
            to="/settings"
            startIcon={<SettingsIcon />}
          >
            Settings
          </Button>
        </Box>
        
        <Typography variant="body2" sx={{ opacity: 0.7 }}>
          v1.0
        </Typography>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
