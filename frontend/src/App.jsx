import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import { ErrorBoundary } from './components/ErrorBoundary.jsx';
import Layout from './components/Layout.jsx';

const Home            = lazy(() => import('./pages/Home.jsx'));
const Dashboard       = lazy(() => import('./pages/Dashboard.jsx'));
const Prediction      = lazy(() => import('./pages/Prediction.jsx'));
const ModelPerformance = lazy(() => import('./pages/ModelPerformance.jsx'));
const About           = lazy(() => import('./pages/About.jsx'));
const NotFound        = lazy(() => import('./pages/NotFound.jsx'));

function PageLoader() {
  return (
    <div className="loading-center" style={{ minHeight: '60vh' }}>
      <div className="spinner" aria-label="Loading page" />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <ErrorBoundary>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Suspense fallback={<PageLoader />}><Home /></Suspense>} />
            <Route path="dashboard"   element={<Suspense fallback={<PageLoader />}><Dashboard /></Suspense>} />
            <Route path="predict"     element={<Suspense fallback={<PageLoader />}><Prediction /></Suspense>} />
            <Route path="performance" element={<Suspense fallback={<PageLoader />}><ModelPerformance /></Suspense>} />
            <Route path="about"       element={<Suspense fallback={<PageLoader />}><About /></Suspense>} />
            <Route path="404"         element={<Suspense fallback={<PageLoader />}><NotFound /></Suspense>} />
            <Route path="*"           element={<Navigate to="/404" replace />} />
          </Route>
        </Routes>
      </ErrorBoundary>
    </BrowserRouter>
  );
}
