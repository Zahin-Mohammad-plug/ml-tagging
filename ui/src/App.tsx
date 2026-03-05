import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Container } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';

import Navbar from './components/Navbar';
import SuggestionsList from './pages/SuggestionsList';
import SuggestionDetail from './pages/SuggestionDetail';
import Dashboard from './pages/Dashboard';
import Settings from './pages/Settings';
import ProcessVideo from './pages/ProcessVideo';
import ReviewQueue from './pages/ReviewQueue';
import Tags from './pages/Tags';

// Create a dark theme optimized for content review
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#ff6b6b', // Soft red for attention
      light: '#ff9999',
      dark: '#cc5555',
    },
    secondary: {
      main: '#4ecdc4', // Teal for secondary actions
      light: '#7dd3d8',
      dark: '#3ba39c',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
    body1: {
      fontSize: '0.95rem',
      lineHeight: 1.6,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: '8px',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          border: '1px solid #333',
        },
      },
    },
  },
});

// React Query client configuration
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router
          future={{
            v7_startTransition: true,
            v7_relativeSplatPath: true,
          }}
        >
          <div className="App">
            <Navbar />
            <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/process" element={<ProcessVideo />} />
                <Route path="/review" element={<ReviewQueue />} />
                <Route path="/suggestions" element={<SuggestionsList />} />
                <Route path="/suggestions/:id" element={<SuggestionDetail />} />
                <Route path="/tags" element={<Tags />} />
                <Route path="/blacklist" element={<Navigate to="/tags" replace />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Routes>
            </Container>
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#333',
                  color: '#fff',
                },
                success: {
                  iconTheme: {
                    primary: '#4ecdc4',
                    secondary: '#fff',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#ff6b6b',
                    secondary: '#fff',
                  },
                },
              }}
            />
          </div>
        </Router>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;