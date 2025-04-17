// frontend/src/config.ts

// Use Vite's environment variable handling (import.meta.env)
// See: https://vitejs.dev/guide/env-and-mode.html

// Define the base URL from environment variables, defaulting to the ngrok tunnel for development
// In a real deployment, you would set VITE_API_BASE_URL via your deployment environment
const API_BASE_URL_HTTP = import.meta.env.VITE_API_BASE_URL_HTTP || "https://e7a9-213-192-2-118.ngrok-free.app";

// Derive WebSocket URL from the HTTP URL
const API_BASE_URL_WS = API_BASE_URL_HTTP.replace(/^http/, 'ws');

// Construct specific URLs
export const config = {
  apiBaseUrlHttp: API_BASE_URL_HTTP,
  apiBaseUrlWs: API_BASE_URL_WS,
  healthCheckUrl: `${API_BASE_URL_HTTP}/health`,
  // Add other API endpoints if needed (e.g., detect if called directly)
  // detectUrl: `${API_BASE_URL_HTTP}/detect`,
  videoFeedWsUrl: `${API_BASE_URL_WS}/ws/video_feed`,
  // Function to construct static image URLs
  getStaticImageUrl: (relativePath: string): string => {
      // Ensure relativePath doesn't start with a slash if apiBaseUrlHttp already ends with one
      const cleanRelativePath = relativePath.startsWith('/') ? relativePath.substring(1) : relativePath;
      // Construct the full URL for static images served by the backend
      return `${API_BASE_URL_HTTP}/static/images/${cleanRelativePath}`;
  }
};

console.log("Frontend Config Loaded:");
console.log("API Base URL (HTTP):", config.apiBaseUrlHttp);
console.log("API Base URL (WS):", config.apiBaseUrlWs);
console.log("Video Feed WS URL:", config.videoFeedWsUrl); 