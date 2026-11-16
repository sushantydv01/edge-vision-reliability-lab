import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Info, AlertCircle } from 'lucide-react';
import { researchData, formatPercent, formatMB, formatRatio, formatDrop } from '@/data';

export function PrecisionMatrixSection() {
  const { precisionVariants } = researchData;

  return (
    <section className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-1 border-b border-surface-border">
        <div>
          <h2 className="text-lg font-bold text-text-primary tracking-tight">
            Cross-Precision Compression & Reliability Matrix
          </h2>
          <p className="text-xs text-text-secondary">
            Evaluating all 6 standardized configurations across FP32, FP16, and INT8 post-training quantization.
          </p>
        </div>
        <Badge variant="outline" className="font-mono text-[11px]">
          6 Canonical Configurations
        </Badge>
      </div>

      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-surface-secondary/70 text-text-muted font-mono uppercase text-[10px] border-b border-surface-border">
                <tr>
                  <th className="py-2.5 px-3">Architecture</th>
                  <th className="py-2.5 px-3">Precision</th>
                  <th className="py-2.5 px-3">Runtime & Target</th>
                  <th className="py-2.5 px-3 text-right">Disk Size</th>
                  <th className="py-2.5 px-3 text-right">Compression</th>
                  <th className="py-2.5 px-3 text-right">Clean Acc.</th>
                  <th className="py-2.5 px-3 text-right">Corrupted Acc.</th>
                  <th className="py-2.5 px-3 text-right">Mean Drop</th>
                  <th className="py-2.5 px-3 text-right">FP32 Concordance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border font-mono text-[11px]">
                {precisionVariants.map((v) => {
                  const isBaseline = v.precision_key === 'fp32';
                  const isInt8 = v.precision_key === 'int8';
                  const isFP16 = v.precision_key === 'fp16';

                  return (
                    <tr key={v.config_id} className="hover:bg-surface-hover/30 transition-colors">
                      <td className="py-2.5 px-3 font-semibold text-text-primary font-sans">
                        {v.display_name}
                      </td>
                      <td className="py-2.5 px-3">
                        <Badge
                          variant={
                            isInt8 ? 'accent' : isFP16 ? 'purple' : 'default'
                          }
                          className="text-[10px] py-0 px-1.5"
                        >
                          {v.precision}
                        </Badge>
                      </td>
                      <td className="py-2.5 px-3 text-text-secondary text-[10px]">
                        <div>{v.runtime_backend}</div>
                        <div className="text-text-muted text-[9px]">{v.evaluation_device}</div>
                      </td>
                      <td className="py-2.5 px-3 text-right text-text-primary">
                        {formatMB(v.serialized_bytes)}
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        {isBaseline ? (
                          <span className="text-text-muted">1.00× (Baseline)</span>
                        ) : (
                          <span className="text-accent font-semibold">
                            {formatRatio(v.compression_ratio)} ({v.size_reduction_pct.toFixed(1)}% drop)
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-right font-semibold text-text-primary">
                        {formatPercent(v.clean_accuracy)}
                      </td>
                      <td className="py-2.5 px-3 text-right text-text-primary">
                        {formatPercent(v.mean_corrupted_accuracy)}
                      </td>
                      <td className="py-2.5 px-3 text-right text-rose-400">
                        {formatDrop(v.mean_accuracy_drop_pp)}
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        {isBaseline ? (
                          <span className="text-text-muted">100.0% (Ref)</span>
                        ) : (
                          <span className={v.concordance_pct >= 99 ? 'text-emerald-400 font-semibold' : 'text-amber-400 font-semibold'}>
                            {v.concordance_pct.toFixed(2)}%
                            <span className="text-[9px] text-text-muted ml-1">
                              ({v.changed_predictions} diff)
                            </span>
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Methodological Context & Precision Trade-off Explanation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border space-y-1.5">
          <div className="flex items-center gap-1.5 text-text-primary font-semibold">
            <Info className="w-3.5 h-3.5 text-accent shrink-0" />
            <span>Serialized Disk Size vs Theoretical Parameter Payload</span>
          </div>
          <p className="text-[11px] text-text-secondary leading-relaxed">
            Reported storage reflects actual serialized TorchScript archives and state dict files on disk.
            For example, ResNet-18 INT8 occupies 11.30 MB on disk (3.96× compression) compared to theoretical
            parameter bytes (11.17 MB). MobileNetV3-Small INT8 occupies 1.94 MB on disk (3.22× compression) compared to theoretical
            parameter bytes (1.53 MB), reflecting TorchScript archive serialization and non-quantized operator representation.
          </p>
        </div>

        <div className="p-3.5 rounded-lg bg-surface-secondary/40 border border-surface-border space-y-1.5">
          <div className="flex items-center gap-1.5 text-text-primary font-semibold">
            <AlertCircle className="w-3.5 h-3.5 text-purple-400 shrink-0" />
            <span>Hardware Target & Runtime Separation</span>
          </div>
          <p className="text-[11px] text-text-secondary leading-relaxed">
            <strong>INT8 quantization</strong> executes exclusively on Host CPU via PyTorch's QNNPACK engine
            (MPS does not support quantized INT8 operators). In contrast, <strong>FP16 half-precision</strong> was
            benchmarked on Apple Silicon MPS via Metal acceleration. Cross-precision comparisons illustrate
            practical edge deployment configurations rather than single-runtime scaling.
          </p>
        </div>
      </div>
    </section>
  );
}
