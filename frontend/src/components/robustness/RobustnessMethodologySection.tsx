import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { ShieldAlert, Info, Database, Cpu, SlidersHorizontal, AlertCircle } from 'lucide-react';

export function RobustnessMethodologySection() {
  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex items-center gap-2">
          <Badge variant="accent" className="font-mono text-[10px]">
            Section F
          </Badge>
          <CardTitle className="text-base font-bold text-text-primary">
            Robustness Evaluation Methodology & Scientific Limitations
          </CardTitle>
        </div>
      </CardHeader>

      <CardContent className="pt-4 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {/* 1. Suite Specification */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <SlidersHorizontal className="w-4 h-4 text-accent shrink-0" />
              <span>Custom Synthetic Suite (Not CIFAR-10-C)</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              This project evaluates an independent custom synthetic corruption suite comprising 6 transformations across 5 severity levels (30 conditions). It does <strong>not</strong> implement the official 15-corruption CIFAR-10-C release (Hendrycks & Dietterich, 2019) and does not report official mean Corruption Error (mCE).
            </p>
          </div>

          {/* 2. Motion Blur Constraints */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
              <span>1D Horizontal Motion Blur</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              Motion blur is implemented strictly using linear 1D horizontal averaging kernels (lengths 3, 5, 7, 9, 11 pixels). It evaluates vulnerability to lateral transit without modeling multi-axis camera trajectories, non-linear shakes, or rolling-shutter sensor distortions.
            </p>
          </div>

          {/* 3. Photometric Shift Constraints */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <Info className="w-4 h-4 text-purple-400 shrink-0" />
              <span>Photometric Darkening Only</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              The brightness corruption evaluates strictly multiplicative attenuation factors [0.30, 0.90] (0.90, 0.75, 0.60, 0.45, 0.30). It models underexposure and illumination drop without simulating specular highlights, glare, blooming, or high-key sensor saturation.
            </p>
          </div>

          {/* 4. Dataset & Resolution Scope */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <Database className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>Dataset Scope & Resolution</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              Evaluations operate exclusively on the canonical CIFAR-10 test split (10,000 images at 32×32 pixel spatial resolution). Results demonstrate algorithmic stability under downscaled low-resolution conditions and cannot be directly generalized to high-resolution camera feeds.
            </p>
          </div>

          {/* 5. Deterministic Controls */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <ShieldAlert className="w-4 h-4 text-accent shrink-0" />
              <span>Deterministic Experimental Seed</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              All baseline models were trained and corrupted under a fixed experimental seed (42). Reported metrics represent deterministic single-seed evaluations on held-out test splits without multi-run variance confidence intervals.
            </p>
          </div>

          {/* 6. Real-World Reliability Boundaries */}
          <div className="p-3.5 rounded-xl bg-surface-secondary/40 border border-surface-border space-y-1.5 text-xs">
            <div className="flex items-center gap-2 font-bold text-text-primary">
              <Cpu className="w-4 h-4 text-rose-400 shrink-0" />
              <span>Production Safety Caveat</span>
            </div>
            <p className="text-[11px] leading-relaxed text-text-secondary">
              High accuracy retention under mathematical synthetic perturbations demonstrates topological representation stability. However, it does not certify safety-critical edge deployment reliability under complex real-world environmental shifts (fog, rain, lens condensation).
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
