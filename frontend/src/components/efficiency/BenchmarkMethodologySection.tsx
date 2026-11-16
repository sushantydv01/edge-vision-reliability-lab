import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Cpu, Zap, Layers, AlertCircle, Database, ShieldAlert } from 'lucide-react';

export function BenchmarkMethodologySection() {
  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex items-center gap-2">
          <Badge variant="accent" className="font-mono text-[10px]">
            Section G
          </Badge>
          <CardTitle className="text-base font-bold text-text-primary">
            Benchmarking Methodology, Hardware Specifications & Protocol Limitations
          </CardTitle>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {/* 1. Hardware Specifications */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <Cpu className="w-4 h-4 text-accent shrink-0" />
              <span>Apple M3 Hardware Specification</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              Evaluated on Apple Silicon M3 (8-core architecture: 4 Performance cores + 4 Efficiency cores, 8 GB Unified Memory, macOS 27.0). Python 3.12.16, PyTorch 2.15.0, Torchvision 0.29.2. CPU benchmarks execute unpinned across available cores (default 4 intra-op threads).
            </p>
          </div>

          {/* 2. Compute Backends */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <Zap className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>Execution Runtimes (Eager vs QNNPACK)</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              FP32 and FP16 runtimes execute through standard PyTorch Eager mode with native Metal Performance Shaders (MPS) acceleration on Apple Silicon GPU. INT8 static post-training quantization executes strictly on CPU via TorchScript and the QNNPACK quantized kernel backend.
            </p>
          </div>

          {/* 3. Latency Timing Protocol */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <Layers className="w-4 h-4 text-purple-400 shrink-0" />
              <span>Total-Engine vs. Forward-Only Latency</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              <strong>Total-Engine Latency:</strong> Accounts for complete batch tensor allocation, host-to-device memory copy, forward propagation, and synchronization barriers. <strong>Forward-Only:</strong> Isolates graph execution alone. Throughput (img/s) is derived strictly from respective batch timing.
            </p>
          </div>

          {/* 4. Warmup & Repetition Protocols */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <Database className="w-4 h-4 text-amber-400 shrink-0" />
              <span>Warmup & Repetition Methodology</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              Task 4.4 primary benchmark suite conducted 10 warmup iterations followed by 100 timed iterations per batch size. Isolated CPU microbenchmarks were executed in fresh OS child processes with 20 warmup runs and 100 timed iterations across 2 independent process repetitions.
            </p>
          </div>

          {/* 5. CPU FP16 Diagnostic Emulation */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>CPU FP16 Software Emulation Note</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              In PyTorch on Apple Silicon CPU, FP16 execution lacks native SIMD vectorization and runs in software emulation (measuring 249 ms for ResNet-18). On Apple Silicon MPS, FP16 executes natively on GPU cores (4.9 ms). CPU FP16 results are flagged as diagnostic and excluded by default.
            </p>
          </div>

          {/* 6. Scientific Limitations */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <ShieldAlert className="w-4 h-4 text-accent shrink-0" />
              <span>Benchmarking Limitations & Edge Realities</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              Benchmarks utilize pre-allocated contiguous memory tensors and do not simulate camera driver DMA latency, sensor ring-buffer contention, or thermal throttling under sustained edge loads. Results characterize algorithmic throughput under optimal conditions.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
