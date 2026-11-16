import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Activity, Gauge, Cpu, Zap } from 'lucide-react';
import { researchData, formatFps, formatMs } from '@/data';

export function EfficiencyHighlightsSection() {
  const { efficiencyHighlights } = researchData;
  const { mobilenet_int8_cpu_b64, mobilenet_fp16_mps_b64, mobilenet_cpu_b1_speedup, resnet_fp16_mps_b64 } =
    efficiencyHighlights;

  return (
    <section className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-1 border-b border-surface-border">
        <div>
          <h2 className="text-lg font-bold text-text-primary tracking-tight">
            Empirical Inference Efficiency & Scaling Highlights
          </h2>
          <p className="text-xs text-text-secondary">
            Hardware-benchmarked throughput and latency metrics on Apple Silicon M3 (Darwin arm64).
          </p>
        </div>
        <Badge variant="outline" className="font-mono text-[11px]">
          Hardware Profile: M3 8-Core (4P+4E)
        </Badge>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Peak CPU Edge Throughput */}
        <Card className="p-4 space-y-2 border-accent/30 bg-surface/80 relative overflow-hidden">
          <div className="flex items-center justify-between text-[11px] text-text-muted font-mono uppercase">
            <span>{mobilenet_int8_cpu_b64.label}</span>
            <Activity className="w-3.5 h-3.5 text-accent" />
          </div>
          <div className="text-xl font-bold font-mono text-text-primary">
            {formatFps(mobilenet_int8_cpu_b64.total_throughput_fps!)}
          </div>
          <div className="space-y-1 text-[11px] text-text-secondary border-t border-surface-border/50 pt-1.5">
            <div className="flex justify-between">
              <span>Target:</span>
              <span className="font-mono text-text-primary">MobileNet INT8 (CPU)</span>
            </div>
            <div className="flex justify-between">
              <span>Batch Size:</span>
              <span className="font-mono text-text-primary">b=64, 4 threads</span>
            </div>
            <div className="flex justify-between">
              <span>Total Engine Latency:</span>
              <span className="font-mono text-text-primary">{formatMs(mobilenet_int8_cpu_b64.median_total_ms!)}</span>
            </div>
            <div className="flex justify-between text-accent">
              <span>Forward Path Only:</span>
              <span className="font-mono font-semibold">{formatMs(mobilenet_int8_cpu_b64.median_forward_ms!)} (11.5k FPS)</span>
            </div>
          </div>
        </Card>

        {/* Card 2: Peak Apple Silicon MPS Throughput */}
        <Card className="p-4 space-y-2 border-purple-500/30 bg-surface/80 relative overflow-hidden">
          <div className="flex items-center justify-between text-[11px] text-text-muted font-mono uppercase">
            <span>{mobilenet_fp16_mps_b64.label}</span>
            <Cpu className="w-3.5 h-3.5 text-purple-400" />
          </div>
          <div className="text-xl font-bold font-mono text-text-primary">
            {formatFps(mobilenet_fp16_mps_b64.total_throughput_fps!)}
          </div>
          <div className="space-y-1 text-[11px] text-text-secondary border-t border-surface-border/50 pt-1.5">
            <div className="flex justify-between">
              <span>Target:</span>
              <span className="font-mono text-text-primary">MobileNet FP16 (MPS)</span>
            </div>
            <div className="flex justify-between">
              <span>Batch Size:</span>
              <span className="font-mono text-text-primary">b=64, Metal GPU</span>
            </div>
            <div className="flex justify-between">
              <span>Total Engine Latency:</span>
              <span className="font-mono text-text-primary">{formatMs(mobilenet_fp16_mps_b64.median_total_ms!)}</span>
            </div>
            <div className="flex justify-between text-purple-400">
              <span>Forward Path Only:</span>
              <span className="font-mono font-semibold">{formatMs(mobilenet_fp16_mps_b64.median_forward_ms!)} (8.4k FPS)</span>
            </div>
          </div>
        </Card>

        {/* Card 3: Single-Image CPU Quantization Speedup */}
        <Card className="p-4 space-y-2 border-emerald-500/30 bg-surface/80 relative overflow-hidden flex flex-col justify-between">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-[11px] text-text-muted font-mono uppercase">
              <span>{mobilenet_cpu_b1_speedup.label}</span>
              <Zap className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-xl font-bold font-mono text-emerald-400">
                {mobilenet_cpu_b1_speedup.speedup_factor}×
              </span>
              <span className="text-[10px] font-mono text-text-muted">
                (4-th default) / 9.1× (1-th)
              </span>
            </div>
            <div className="space-y-1 text-[11px] text-text-secondary border-t border-surface-border/50 pt-1.5">
              <div className="flex justify-between">
                <span>Architecture:</span>
                <span className="font-mono text-text-primary">MobileNetV3-Small</span>
              </div>
              <div className="flex justify-between">
                <span>Threading Mode:</span>
                <span className="font-mono text-text-primary">Batch 1 CPU</span>
              </div>
              <div className="flex justify-between">
                <span>Default 4-Th (FP32/INT8):</span>
                <span className="font-mono text-text-primary">25.82 ms / 0.58 ms</span>
              </div>
              <div className="flex justify-between text-emerald-400">
                <span>Isolated 1-Th (FP32/INT8):</span>
                <span className="font-mono font-semibold">3.78 ms / 0.41 ms</span>
              </div>
            </div>
          </div>
          <div className="text-[10px] text-text-muted leading-tight border-t border-surface-border/40 pt-1.5 mt-1">
            Deployment-path comparison (FP32 Eager vs INT8 TorchScript/QNNPACK): 44.9× under default 4 threads (25.82 ms vs 0.58 ms); 9.1× under isolated 1 thread (3.78 ms vs 0.41 ms) on Apple M3 CPU.
          </div>
        </Card>

        {/* Card 4: ResNet MPS Hardware Acceleration */}
        <Card className="p-4 space-y-2 border-surface-border bg-surface/80 relative overflow-hidden">
          <div className="flex items-center justify-between text-[11px] text-text-muted font-mono uppercase">
            <span>{resnet_fp16_mps_b64.label}</span>
            <Gauge className="w-3.5 h-3.5 text-accent" />
          </div>
          <div className="text-xl font-bold font-mono text-text-primary">
            {formatFps(resnet_fp16_mps_b64.total_throughput_fps!)}
          </div>
          <div className="space-y-1 text-[11px] text-text-secondary border-t border-surface-border/50 pt-1.5">
            <div className="flex justify-between">
              <span>Architecture:</span>
              <span className="font-mono text-text-primary">ResNet-18 (FP16 MPS)</span>
            </div>
            <div className="flex justify-between">
              <span>Batch Scaling:</span>
              <span className="font-mono text-text-primary">b=64, Metal GPU</span>
            </div>
            <div className="flex justify-between">
              <span>Total Latency:</span>
              <span className="font-mono text-text-primary">{formatMs(resnet_fp16_mps_b64.median_total_ms!)}</span>
            </div>
            <div className="flex justify-between text-accent">
              <span>vs CPU FP32 b=64:</span>
              <span className="font-mono font-semibold">10.2× Speedup (208 FPS)</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Critical Benchmark Boundary Notice */}
      <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border text-xs text-text-secondary leading-relaxed space-y-1.5">
        <div>
          <strong className="text-text-primary font-medium">Measurement Boundaries:</strong> All reported metrics reflect
          in-memory benchmarking with strictly timed pre-warmed engines. <strong>Forward-path latency</strong> isolates model execution,
          whereas <strong>total-engine latency</strong> includes tensor ingestion, device transfers, and top-1 output postprocessing.
          Throughput at batch size 64 represents saturated edge pipeline capacity and must not be conflated with single-image latency.
        </div>
        <div className="pt-1.5 border-t border-surface-border/50 text-[11px] space-y-1">
          <div>
            <strong className="text-text-primary font-medium">Single-Image CPU Deployment-Path Comparison:</strong> Under the default four-thread
            configuration on Apple M3 CPU, the measured FP32 Eager latency (25.821 ms) versus INT8 TorchScript/QNNPACK latency (0.575 ms) represents a 44.9× ratio.
            Under isolated single-thread execution, measured latencies were 3.78 ms for FP32 and 0.41 ms for INT8 (a 9.1× ratio), as INT8 latency varied less across
            tested thread configurations while measured FP32 latency was substantially higher under the default four-thread configuration.
          </div>
          <div className="text-text-muted">
            These measurements represent deployment-path comparisons (framework eager vs TorchScript/QNNPACK runtime), not pure precision-only speedups.
            The benchmark establishes these empirical performance differences on this Apple M3 CPU but does not independently identify their underlying causes,
            and results should not be generalized to other devices.
          </div>
        </div>
      </div>
    </section>
  );
}
