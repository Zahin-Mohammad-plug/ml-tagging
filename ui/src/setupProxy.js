const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = function(app) {
  // Determine the API target URL
  // If REACT_APP_API_URL is explicitly set and not empty, use it
  // Otherwise, use the Docker service name (works in Docker) or localhost (for local dev)
  let target = process.env.REACT_APP_API_URL;
  
  // Normalize target (trim whitespace)
  if (target) {
    target = target.trim();
  }
  
  // If REACT_APP_API_URL is empty or not set, use Docker service name
  // This works because Docker Compose sets up DNS for service names
  // Also check for localhost URLs (host ports) - in Docker, we need to use service name
  // When running in Docker, localhost refers to the container itself, not the host
  // So we need to use the Docker service name to reach other containers
  if (!target || target === '' || 
      target === 'http://localhost:8080' || target === 'http://localhost:8888' ||
      target.startsWith('http://localhost:')) {
    // In Docker, use service name. Service name resolution works within Docker network
    // The API container runs on port 8080 internally
    target = 'http://tagger-api:8080';
  }
  
  console.log('Proxy target:', target);
  console.log('REACT_APP_API_URL:', process.env.REACT_APP_API_URL);
  
  app.use(
    "/api",
    createProxyMiddleware({
      target: target,
      changeOrigin: true,
      pathRewrite: {
        "^/api": "",
      },
      logLevel: "debug",
      // Increase timeout for image/file requests (frames can be large)
      timeout: 60000, // 60 seconds for file operations
      // Increase proxy timeout for slow file operations
      proxyTimeout: 60000, // 60 seconds
      // Handle timeouts gracefully
      onProxyReq: (proxyReq, req, res) => {
        // Set timeout on the proxy request
        proxyReq.setTimeout(60000);
      },
      onError: (err, req, res) => {
        console.error('Proxy error:', err.message);
        console.error('Proxy error details:', err.code, err.syscall, err.address, err.port);
        if (!res.headersSent) {
          if (err.code === 'ECONNREFUSED') {
            res.status(503).json({ 
              error: 'Service Unavailable', 
              message: `Cannot connect to API at ${target}. Is the API service running?` 
            });
          } else {
            res.status(504).json({ error: 'Gateway Timeout', message: 'The request took too long to complete' });
          }
        }
      },
    })
  );
};
