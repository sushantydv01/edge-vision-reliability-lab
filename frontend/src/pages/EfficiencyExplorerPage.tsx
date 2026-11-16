import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Tabs } from '@/components/ui/Tabs';
import {
  Gauge,
  Activity,
  HardDrive,
  Cpu,
  BarChart,
  ScatterChart,
  Info,
} from 'lucide-react';

export function EfficiencyExplorerPage() {
  const [activeMetric, setActiveMetric] = useState('pareto');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-surface-border">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-bold text-text-primary tracking-tight">
              Efficiency & Pareto Explorer
            </h2>
            <Badge variant="accent">Task 5.5 Scaffold</Badge>
          </div>
          <p className="text-xs text-text-secondary mt-1">
            Empirical latency, throughput, memory scaling, and Pareto optimality analysis on Apple Silicon M3.
          </p>
        </div>

        <Tabs
          items={[
            { id: 'pareto', label: 'Pareto Frontier', icon: <ScatterChart className="w-3.5 h-3.5" /> },
            { id: 'throughput', label: 'Throughput (FPS)', icon: <Activity className="w-3.5 h-3.5" /> },
            { id: 'latency', label: 'Batch Scaling', icon: <Gauge className="w-3.5 h-3.5" /> },
          ]}
          activeId={activeMetric}
          onChange={setActiveMetric}
        />
      </div>

      {/* Metric Highlights Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-text-muted font-mono uppercase">
            <span>Total-Engine Throughput</span>
            <Activity className="w-3.5 h-3.5 text-accent" />
          </div>
          <div className="text-xl font-bold font-mono text-text-primary">
            7,754.8 <span className="text-xs font-normal text-text-secondary">img/s</span>
          </div>
          <p className="text-[11px] text-text-muted leading-tight">
            MobileNetV3-Small INT8 CPU (Batch 64, 4 threads, total latency 8.253 ms)
          </p>
        </Card>

        <Card className="p-4 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-text-muted font-mono uppercase">
            <span>Total-Engine Throughput</span>
            <Cpu className="w-3.5 h-3.5 text-accent" />
          </div>
          <div className="text-xl font-bold font-mono text-text-primary">
            5,986.3 <span className="text-xs font-normal text-text-secondary">img/s</span>
          </div>
          <p className="text-[11px] text-text-muted leading-tight">
            MobileNetV3-Small FP16 MPS (Batch 64, Metal, total latency 10.691 ms)
          </p>
        </Card>

        <Card className="p-4 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-text-muted font-mono uppercase">
            <span>Artifact Compression</span>
            <HardDrive className="w-3.5 h-3.5 text-accent" />
          </div>
          <div className="text-xl font-bold font-mono text-text-primary">
            3.96× <span className="text-xs font-normal text-text-secondary">Reduction</span>
          </div>
          <p className="text-[11px] text-text-muted leading-tight">
            ResNet-18 FP32 (44.78 MB) → INT8 (11.30 MB serialized on disk)
          </p>
        </Card>

        <Card className="p-4 space-y-1.5">
          <div className="flex items-center justify-between text-xs text-text-muted font-mono uppercase">
            <span>Forward-Path Latency</span>
            <Gauge className="w-3.5 h-3.5 text-accent" />
          </div>
          <div className="text-xl font-bold font-mono text-text-primary">
            5.550 <span className="text-xs font-normal text-text-secondary">ms</span>
          </div>
          <p className="text-[11px] text-text-muted leading-tight">
            MobileNetV3-Small INT8 CPU (Batch 64 forward-only: 11,531.5 img/s)
          </p>
        </Card>
      </div>

      {/* Pareto Frontier Scatter Chart Placeholder */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm">
                2D Multi-Objective Pareto Frontier (Accuracy vs Latency)
              </CardTitle>
              <CardDescription>
                Revealing Pareto-optimal configurations balancing corruption robustness and compute efficiency
              </CardDescription>
            </div>
            <Badge variant="outline">Task 5.5 Interactive Recharts</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-72 border border-dashed border-surface-border rounded-lg flex flex-col items-center justify-center text-center p-6 bg-surface-secondary/20">
            <ScatterChart className="w-10 h-10 text-text-muted mb-3" />
            <span className="text-sm font-semibold text-text-primary">
              Pareto Frontier Visualization Placeholder
            </span>
            <p className="text-xs text-text-secondary max-w-md mt-1 leading-relaxed">
              In Task 5.5, this interactive ScatterPlot will plot Clean Accuracy vs Median Latency,
              with Pareto-optimal convex hull overlays, tooltips with file sizes, and device filter toggles.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Batch Scaling Comparison Placeholder */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-5 space-y-3">
          <div className="flex items-center justify-between border-b border-surface-border pb-2">
            <h4 className="text-xs font-semibold text-text-primary">Batch Scaling Regimes (B=1, 8, 32, 64)</h4>
            <Badge variant="outline">Task 4.4 Data</Badge>
          </div>
          <div className="h-44 border border-dashed border-surface-border rounded-lg flex flex-col items-center justify-center text-center p-4 bg-surface-secondary/20">
            <BarChart className="w-8 h-8 text-text-muted mb-2" />
            <span className="text-xs font-medium text-text-secondary">
              Throughput Scaling Chart Placeholder
            </span>
            <span className="text-[11px] text-text-muted">
              Populated in Task 5.5 from benchmark_summary.csv
            </span>
          </div>
        </Card>

        <Card className="p-5 space-y-3">
          <div className="flex items-center justify-between border-b border-surface-border pb-2">
            <h4 className="text-xs font-semibold text-text-primary">Engine vs Forward-Path Timing</h4>
            <Badge variant="outline">Pipeline Analysis</Badge>
          </div>
          <div className="h-44 border border-dashed border-surface-border rounded-lg flex flex-col items-center justify-center text-center p-4 bg-surface-secondary/20">
            <Activity className="w-8 h-8 text-text-muted mb-2" />
            <span className="text-xs font-medium text-text-secondary">
              Preprocessing vs Forward vs Postprocessing Split
            </span>
            <span className="text-[11px] text-text-muted">
              Populated in Task 5.5 from benchmark breakdown logs
            </span>
          </div>
        </Card>
      </div>

      <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border flex items-center gap-2.5 text-xs text-text-secondary">
        <Info className="w-4 h-4 text-accent shrink-0" />
        <span>
          Throughput and latency figures reflect empirical measurements from Task 4.4 and Task 5.3C on Apple Silicon M3 (unpinned 4-thread CPU & Metal MPS).
        </span>
      </div>
    </div>
  );
}
