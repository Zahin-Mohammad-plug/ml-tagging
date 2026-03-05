import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography } from '@mui/material';

interface VideoPlayerProps {
  sceneId: string;
  sceneTitle?: string;
  videoUrl?: string;
  seekTo?: number; // Timestamp in seconds to seek to
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ 
  sceneId, 
  sceneTitle,
  videoUrl,
  seekTo
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [hasSeeked, setHasSeeked] = useState(false);
  
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8888';
  // Use provided videoUrl or construct one from the API
  const streamUrl = videoUrl || `${apiUrl}/scenes/${sceneId}/stream`;

  // Handle seeking when seekTo prop changes
  useEffect(() => {
    if (seekTo !== undefined && seekTo !== null && seekTo >= 0 && videoRef.current) {
      const video = videoRef.current;
      
      // Function to seek to the timestamp
      const seekToTime = () => {
        if (video && !isNaN(seekTo) && seekTo >= 0) {
          // Don't check duration - just seek (duration might not be loaded yet)
          try {
            video.currentTime = seekTo;
          } catch (err) {
            console.warn('Failed to seek video:', err);
          }
        }
      };
      
      // If video is already loaded enough, seek immediately
      if (video.readyState >= 2 || video.readyState >= 1) {
        seekToTime();
      }
      
      // Also wait for video to be ready
      const handleCanPlay = () => {
        seekToTime();
      };
      
      const handleLoadedMetadata = () => {
        seekToTime();
      };
      
      video.addEventListener('loadedmetadata', handleLoadedMetadata);
      video.addEventListener('canplay', handleCanPlay);
      video.addEventListener('canplaythrough', handleCanPlay);
      video.addEventListener('loadeddata', handleCanPlay);
      
      return () => {
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
        video.removeEventListener('canplay', handleCanPlay);
        video.removeEventListener('canplaythrough', handleCanPlay);
        video.removeEventListener('loadeddata', handleCanPlay);
      };
    }
  }, [seekTo]);

  return (
    <Box sx={{ width: '100%', bgcolor: 'black', borderRadius: 1, overflow: 'hidden' }}>
      <Box sx={{ position: 'relative', paddingTop: '56.25%' /* 16:9 aspect ratio */ }}>
        <video
          ref={videoRef}
          controls
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'contain'
          }}
          preload="metadata"
        >
          <source src={streamUrl} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      </Box>
      
      <Box sx={{ p: 1, bgcolor: 'grey.900', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="body2" color="white" noWrap sx={{ flex: 1 }}>
          {sceneTitle || `Scene ${sceneId}`}
        </Typography>
      </Box>
    </Box>
  );
};

export default VideoPlayer;
