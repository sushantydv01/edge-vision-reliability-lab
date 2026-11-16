import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import {
  LineChart,
  Eye,
  AlertTriangle,
} from 'lucide-react';

const CORRUPTION_CATEGORIES = {
  Noise: ['gaussian_noise'],
  Blur: ['gaussian_blur', 'motion_blur'],
  Photometric: ['brightness', 'contrast'],
  Digital: ['jpeg_compression'],
};

export function RobustnessLabPage() {
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [selectedSeverity, setSelectedSeverity] = useState(3);
  const [selectedModel, setSelectedModel] = useState('resnet18_cifar');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-surface-border">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-bold text-text-primary tracking-tight">
              Robustness Laboratory (Custom Synthetic Suite)
            </h2>
            <Badge variant="accent">Task 5.4 Scaffold</Badge>
          </div>
          <p className="text-xs text-text-secondary mt-1">
            Out-of-distribution robustness evaluation across 6 synthetic corruption transformations and 5 severity levels.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="outline" className="font-mono">
            6 Corruptions × 5 Severities (30 Conditions)
          </Badge>
        </div>
      </div>

      {/* Exploration Controls Bar */}
      <Card className="p-4 bg-surface-secondary/50 border-surface-border">
        <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-4 gap-4 items-end">
          <Select
            label="Evaluation Architecture"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            options={[
              { value: 'resnet18_cifar', label: 'ResNet-18 (Deep Residual)' },
              { value: 'mobilenet_v3_small_cifar', label: 'MobileNetV3-Small (Inverted Bottleneck)' },
            ]}
          />

          <Select
            label="Corruption Category"
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            options={[
              { value: 'All', label: 'All Categories (6 Corruptions)' },
              { value: 'Noise', label: 'Noise (Gaussian Noise)' },
              { value: 'Blur', label: 'Blur (Gaussian Blur, Motion Blur)' },
              { value: 'Photometric', label: 'Photometric (Brightness, Contrast)' },
              { value: 'Digital', label: 'Digital (JPEG Compression)' },
            ]}
          />

          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-text-secondary">Severity Level:</span>
              <span className="font-mono text-accent font-semibold">Level {selectedSeverity} / 5</span>
            </div>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map((lvl) => (
                <button
                  key={lvl}
                  onClick={() => setSelectedSeverity(lvl)}
                  className={`flex-1 py-1.5 rounded text-xs font-mono font-medium transition-all ${
                    selectedSeverity === lvl
                      ? 'bg-accent text-white font-bold shadow-xs'
                      : 'bg-surface border border-surface-border text-text-secondary hover:text-text-primary hover:bg-surface-hover'
                  }`}
                >
                  {lvl}
                </button>
              ))}
            </div>
          </div>

          <div>
            <Button
              variant="default"
              className="w-full text-xs font-semibold"
              disabled
              title="Active exploration connected in Task 5.4"
            >
              <Eye className="w-3.5 h-3.5 mr-1 text-accent" />
              Analyze Category
            </Button>
          </div>
        </div>
      </Card>

      {/* Visual Corruptions Taxonomy Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(CORRUPTION_CATEGORIES).map(([cat, corruptions]) => (
          <Card key={cat} className="p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-surface-border pb-2">
              <span className="text-xs font-semibold text-text-primary tracking-tight">{cat}</span>
              <Badge variant="outline" className="text-[10px]">{corruptions.length} {corruptions.length === 1 ? 'type' : 'types'}</Badge>
            </div>
            <ul className="space-y-1.5">
              {corruptions.map((c) => (
                <li
                  key={c}
                  className="text-[11px] font-mono text-text-secondary flex items-center justify-between py-1 px-1.5 rounded hover:bg-surface-hover/50"
                >
                  <span>{c}</span>
                  <span className="text-[10px] text-text-muted">Sev. 1–5</span>
                </li>
              ))}
            </ul>
          </Card>
        ))}
      </div>

      {/* Degradation Curves Chart Placeholder */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm">
                Custom Suite Degradation Curves (Severity 1 to 5)
              </CardTitle>
              <CardDescription>
                Contrasting clean test accuracy against corrupted accuracy across 5 discrete severity levels
              </CardDescription>
            </div>
            <Badge variant="outline">Task 5.4 Visualization</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-64 border border-dashed border-surface-border rounded-lg flex flex-col items-center justify-center text-center p-6 bg-surface-secondary/20">
            <LineChart className="w-10 h-10 text-text-muted mb-3" />
            <span className="text-sm font-semibold text-text-primary">
              Custom Synthetic Degradation Curves Placeholder
            </span>
            <p className="text-xs text-text-secondary max-w-md mt-1 leading-relaxed">
              In Task 5.4, this interactive Recharts module will plot measured corrupted accuracy across all 6 transformations
              (Gaussian noise, Gaussian blur, motion blur, brightness, contrast, JPEG compression) directly from historical experiment logs.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Documented Methodology Limitations */}
      <div className="p-4 rounded-lg bg-surface-secondary/40 border border-surface-border space-y-2 text-xs">
        <div className="flex items-center gap-2 text-text-primary font-semibold">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
          <span>Custom Synthetic Suite Methodology & Documented Limitations</span>
        </div>
        <ul className="space-y-1 text-text-secondary text-[11px] list-disc list-inside">
          <li><strong>Horizontal motion blur only:</strong> Motion blur evaluates horizontal camera displacement using linear 1D convolution kernels.</li>
          <li><strong>Darkening-only brightness:</strong> The brightness operator attenuates pixel intensities across factors [0.30, 0.90] (illumination loss) without overexposure modeling.</li>
          <li><strong>Generalization boundary:</strong> The suite evaluates algorithmic sensitivity to controlled mathematical perturbations and does not establish real-world out-of-distribution generalization.</li>
        </ul>
      </div>
    </div>
  );
}
