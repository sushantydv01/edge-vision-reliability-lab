export interface ProjectMetadata {
  title: string;
  objective: string;
  dataset: string;
  hardware: string;
  os: string;
  software: string;
  architectures_count: number;
  precisions_count: number;
  total_configurations: number;
  corruptions_count: number;
  severity_levels_count: number;
  conditions_per_model: number;
  samples_per_condition: number;
  samples_per_model: number;
  forward_passes_per_tier: number;
  total_project_predictions: number;
  seed: number;
  evaluation_label?: string;
  evaluation_description?: string;
}

export interface BaselineModel {
  model_id: string;
  name: string;
  description: string;
  parameters: number;
  state_dict_elements: number;
  fp32_serialized_bytes: number;
  fp32_serialized_mb: number;
  clean_accuracy: number;
  cross_entropy_loss: number;
  mean_corrupted_accuracy: number;
  mean_accuracy_drop_pp: number;
  mean_relative_accuracy_drop: number;
}

export interface BaselineComparison {
  resnet18: BaselineModel;
  mobilenet_v3_small: BaselineModel;
  comparison_ratios: {
    parameter_ratio: number;
    size_ratio: number;
    clean_accuracy_diff_pp: number;
    corrupted_accuracy_diff_pp: number;
  };
}

export interface PrecisionVariant {
  config_id: string;
  model_id: string;
  display_name: string;
  precision: 'FP32' | 'FP16' | 'INT8';
  precision_key: 'fp32' | 'fp16' | 'int8';
  parameters: number;
  theoretical_param_bytes: number;
  serialized_bytes: number;
  serialized_mb: number;
  clean_accuracy: number;
  cross_entropy_loss: number;
  concordance_pct: number;
  changed_predictions: number;
  mean_corrupted_accuracy: number;
  mean_accuracy_drop_pp: number;
  compression_ratio: number;
  size_reduction_pct: number;
  runtime_backend: string;
  evaluation_device: string;
}

export interface SeverityDataPoint {
  severity: number;
  label: string;
  resnet_fp32: number;
  resnet_fp16: number;
  resnet_int8: number;
  mobilenet_fp32: number;
  mobilenet_fp16: number;
  mobilenet_int8: number;
}

export interface CorruptionDataPoint {
  key: string;
  name: string;
  category: string;
  resnet_fp32: number;
  resnet_fp16: number;
  resnet_int8: number;
  mobilenet_fp32: number;
  mobilenet_fp16: number;
  mobilenet_int8: number;
  resnet_drop_pp: number;
  mobilenet_drop_pp: number;
  resnet_advantage_pp: number;
  is_crossover: boolean;
}

export interface EfficiencyHighlightItem {
  label: string;
  model: string;
  precision?: string;
  device: string;
  backend?: string;
  batch_size: number;
  total_throughput_fps?: number;
  forward_throughput_fps?: number;
  median_total_ms?: number;
  median_forward_ms?: number;
  fp32_median_total_ms?: number;
  int8_median_total_ms?: number;
  speedup_factor?: number;
  fp32_single_thread_ms?: number;
  int8_single_thread_ms?: number;
  single_thread_speedup_factor?: number;
  comparison_type?: string;
  scope_note?: string;
}

export interface KeyFinding {
  id: number;
  title: string;
  badge: string;
  summary: string;
  evidence: string;
}

export interface MethodologyLimitation {
  title: string;
  detail: string;
}

export interface BenchmarkRecord {
  model: string;
  precision: 'fp32' | 'fp16' | 'int8';
  device: 'cpu' | 'mps';
  backend: 'pytorch_eager' | 'torchscript';
  batch_size: number;
  throughput_fps: number;
  median_total_ms: number;
  mean_total_ms: number;
  p95_total_ms: number;
  std_total_ms: number;
  per_image_ms: number;
  median_forward_ms: number;
  forward_only_throughput_fps: number;
  load_time_ms: number;
  is_diagnostic: boolean;
  note: string;
}

export interface ParetoPoint {
  config_id: string;
  model: string;
  display_name: string;
  precision: 'FP32' | 'FP16' | 'INT8';
  precision_key: 'fp32' | 'fp16' | 'int8';
  size_mb: number;
  clean_accuracy: number;
  mean_corrupted_accuracy: number;
  mean_accuracy_drop_pp: number;
  latency_th4_b1_ms: number | null;
  latency_th2_b1_ms: number | null;
  latency_th1_b1_ms: number | null;
  throughput_th4_b64_img_s: number | null;
  throughput_th2_b64_img_s: number | null;
  throughput_th1_b64_img_s: number | null;
  is_pareto_efficient?: boolean;
  dominated_by?: string[];
}

export interface ParetoFrontierSet {
  frontier: ParetoPoint[];
  dominated: ParetoPoint[];
  all_items: ParetoPoint[];
}

export interface ParetoFrontiers {
  all_configs: ParetoPoint[];
  pareto_size_vs_corrupted: ParetoFrontierSet;
  pareto_size_vs_clean: ParetoFrontierSet;
  pareto_cpu_latency_th4_vs_corrupted?: ParetoFrontierSet;
  pareto_cpu_latency_th4_vs_clean?: ParetoFrontierSet;
  cross_architecture?: Record<string, ParetoFrontierSet>;
  within_architecture?: Record<string, Record<string, ParetoFrontierSet>>;
}

export interface ResearchDataset {
  metadata: ProjectMetadata;
  baselineComparison: BaselineComparison;
  precisionVariants: PrecisionVariant[];
  severityCurves: SeverityDataPoint[];
  perCorruptionData: CorruptionDataPoint[];
  efficiencyHighlights: Record<string, EfficiencyHighlightItem>;
  keyFindings: KeyFinding[];
  methodologyLimitations: MethodologyLimitation[];
  benchmarkRecords: BenchmarkRecord[];
  paretoFrontiers: ParetoFrontiers;
}
