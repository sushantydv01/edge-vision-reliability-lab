import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Shell } from '@/components/layout/Shell';
import { ErrorBoundary } from '@/components/layout/ErrorBoundary';
import { PageLoadingFallback } from '@/components/layout/PageLoadingFallback';

// Route-level code splitting via React.lazy
const OverviewPage = lazy(() =>
  import('@/pages/OverviewPage').then((m) => ({ default: m.OverviewPage }))
);
const LiveInferencePage = lazy(() =>
  import('@/pages/LiveInferencePage').then((m) => ({ default: m.LiveInferencePage }))
);
const ModelComparisonPage = lazy(() =>
  import('@/pages/ModelComparisonPage').then((m) => ({ default: m.ModelComparisonPage }))
);
const RobustnessLabPage = lazy(() =>
  import('@/pages/RobustnessLabPage').then((m) => ({ default: m.RobustnessLabPage }))
);
const EfficiencyExplorerPage = lazy(() =>
  import('@/pages/EfficiencyExplorerPage').then((m) => ({ default: m.EfficiencyExplorerPage }))
);

export function App() {
  return (
    <BrowserRouter>
      <ErrorBoundary>
        <Routes>
          <Route element={<Shell />}>
            <Route
              path="/"
              element={
                <Suspense fallback={<PageLoadingFallback />}>
                  <OverviewPage />
                </Suspense>
              }
            />
            <Route path="/overview" element={<Navigate to="/" replace />} />
            <Route
              path="/inference"
              element={
                <Suspense fallback={<PageLoadingFallback />}>
                  <LiveInferencePage />
                </Suspense>
              }
            />
            <Route
              path="/comparison"
              element={
                <Suspense fallback={<PageLoadingFallback />}>
                  <ModelComparisonPage />
                </Suspense>
              }
            />
            <Route
              path="/robustness"
              element={
                <Suspense fallback={<PageLoadingFallback />}>
                  <RobustnessLabPage />
                </Suspense>
              }
            />
            <Route
              path="/efficiency"
              element={
                <Suspense fallback={<PageLoadingFallback />}>
                  <EfficiencyExplorerPage />
                </Suspense>
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </ErrorBoundary>
    </BrowserRouter>
  );
}

export default App;
