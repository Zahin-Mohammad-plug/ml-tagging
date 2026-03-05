# Multi-stage React build for UI
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM node:18-alpine as production

WORKDIR /app

# Install serve to host the built app
RUN npm install -g serve

# Copy built app from build stage
COPY --from=build /app/build ./build

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S react -u 1001

# Change ownership and switch to non-root user
RUN chown -R react:nodejs /app
USER react

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Serve the application
CMD ["serve", "-s", "build", "-l", "3000"]

# Development stage
FROM node:18-alpine as development

WORKDIR /app

# Create non-root user FIRST
RUN addgroup -g 1001 -S nodejs && \
    adduser -S react -u 1001 -G nodejs && \
    chown -R react:nodejs /app

# Switch to react user
USER react

# Copy package files with correct ownership
COPY --chown=react:nodejs package*.json ./

# Install as react user
RUN npm install

# Copy source code with correct ownership
COPY --chown=react:nodejs . .

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "start"]