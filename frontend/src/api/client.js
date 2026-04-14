import axios from 'axios';

// Use the Vercel Render URL in production, but keep localhost for your own computer
const apiBase = import.meta.env.VITE_API_URL 
  ? `${import.meta.env.VITE_API_URL}/api` 
  : 'http://localhost:8000/api';

const api = axios.create({
  baseURL: apiBase,
});

export default api;