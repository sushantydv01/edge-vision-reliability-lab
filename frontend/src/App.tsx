import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Shell } from '@/components/layout/Shell';
import { OverviewPage } from '@/pages/OverviewPage';
import { LiveInferencePage } from '@/pages/LiveInferencePage';
import { ModelComparisonPage } from '@/pages/ModelComparisonPage';
import { RobustnessLabPage } from '@/pages/RobustnessLabPage';
import { EfficiencyExplorerPage } from '@/pages/EfficiencyExplorerPage';

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/overview" element={<Navigate to="/" replace />} />
          <Route path="/inference" element={<LiveInferencePage />} />
          <Route path="/comparison" element={<ModelComparisonPage />} />
          <Route path="/robustness" element={<RobustnessLabPage />} />
          <Route path="/efficiency" element={<EfficiencyExplorerPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
