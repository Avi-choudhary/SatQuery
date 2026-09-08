import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import Console from './components/features/Console';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import ChatHistory from './pages/ChatHistory';
import Layout from './components/layout/Layout';
import SettingsPage from './pages/SettingsPage';
import ProfilePage from './pages/ProfilePage';
import { TraceProvider } from './context/TraceContext';

const App = () => {
  return (
    <TraceProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Route: Landing Page */}
          <Route path="/" element={<LandingPage />} />

          {/* Auth Routes */}
          <Route path="/login" element={<Login mode="login" />} />
          <Route path="/signup" element={<Login mode="signup" />} />

          {/* Protected Routes */}
          <Route element={<Layout />}>
            <Route path="/console" element={<Console />} />
            <Route path="/map" element={<Dashboard />} />
            <Route path="/history" element={<ChatHistory />} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </TraceProvider>
  );
};

export default App;
