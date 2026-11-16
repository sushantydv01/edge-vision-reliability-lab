import { useState, useRef, DragEvent, ChangeEvent } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import {
  UploadCloud,
  Zap,
  AlertTriangle,
  X,
  Sparkles,
  Info,
  RefreshCw,
} from 'lucide-react';
import { SYSTEM_LIMITS } from '@/config/api';

export interface ImageFileState {
  file: File;
  previewUrl: string;
  width: number;
  height: number;
  sizeBytes: number;
  source: 'upload' | 'sample';
}

interface ImageUploadWorkspaceProps {
  activeImage: ImageFileState | null;
  onImageSelected: (imageState: ImageFileState | null) => void;
  onRunInference: (options: { resize: boolean; top_k: number }) => Promise<void>;
  isInferenceRunning: boolean;
  isModelLoaded: boolean;
  isBackendConnected: boolean;
  inferenceError: string | null;
  isOperationBusy?: boolean;
}

const CIFAR_SAMPLES = [
  { id: 'cat', label: 'Cat', filename: 'sample_cat.png' },
  { id: 'airplane', label: 'Airplane', filename: 'sample_airplane.png' },
  { id: 'automobile', label: 'Automobile', filename: 'sample_automobile.png' },
  { id: 'ship', label: 'Ship', filename: 'sample_ship.png' },
  { id: 'frog', label: 'Frog', filename: 'sample_frog.png' },
];

export function ImageUploadWorkspace({
  activeImage,
  onImageSelected,
  onRunInference,
  isInferenceRunning,
  isModelLoaded,
  isBackendConnected,
  inferenceError,
  isOperationBusy = false,
}: ImageUploadWorkspaceProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [resizeEnabled, setResizeEnabled] = useState(true);
  const [topK, setTopK] = useState(5);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Validate and inspect an uploaded file
  const processSelectedFile = (file: File, source: 'upload' | 'sample' = 'upload') => {
    setValidationError(null);

    // 1. Check file size <= 10 MB
    if (file.size > SYSTEM_LIMITS.maxImageSizeBytes) {
      setValidationError(
        `File '${file.name}' exceeds the maximum allowed size of 10 MB (${(file.size / (1024 * 1024)).toFixed(2)} MB).`
      );
      return;
    }

    // 2. Check extension
    const ext = `.${file.name.split('.').pop()?.toLowerCase()}`;
    const allowed = SYSTEM_LIMITS.supportedExtensions as readonly string[];
    if (!allowed.includes(ext)) {
      setValidationError(
        `File extension '${ext}' is unsupported. Allowed formats: ${allowed.join(', ')}.`
      );
      return;
    }

    // 3. Inspect image dimensions using in-memory Image
    const previewUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const width = img.naturalWidth;
      const height = img.naturalHeight;

      if (width <= 0 || height <= 0) {
        URL.revokeObjectURL(previewUrl);
        setValidationError(`Image '${file.name}' has invalid dimensions (${width}×${height}).`);
        return;
      }

      if (width > SYSTEM_LIMITS.maxImageDimension || height > SYSTEM_LIMITS.maxImageDimension) {
        URL.revokeObjectURL(previewUrl);
        setValidationError(
          `Image dimensions (${width}×${height}) exceed the maximum permitted ceiling of ${SYSTEM_LIMITS.maxImageDimension}px.`
        );
        return;
      }

      // Valid image selected
      onImageSelected({
        file,
        previewUrl,
        width,
        height,
        sizeBytes: file.size,
        source,
      });
    };

    img.onerror = () => {
      URL.revokeObjectURL(previewUrl);
      setValidationError(`Failed to decode image from '${file.name}'. Ensure it is a valid, uncorrupted image.`);
    };

    img.src = previewUrl;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processSelectedFile(file, 'upload');
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processSelectedFile(file, 'upload');
    }
  };

  const handleSelectSample = async (sampleFilename: string, sampleLabel: string) => {
    try {
      setValidationError(null);
      const res = await fetch(`/samples/${sampleFilename}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const file = new File([blob], sampleFilename, { type: 'image/png' });
      processSelectedFile(file, 'sample');
    } catch {
      setValidationError(`Failed to load CIFAR-10 sample '${sampleLabel}'.`);
    }
  };

  const handleRemoveImage = () => {
    if (activeImage?.previewUrl) {
      URL.revokeObjectURL(activeImage.previewUrl);
    }
    onImageSelected(null);
    setValidationError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleTriggerPrediction = () => {
    if (!activeImage || !isModelLoaded || isInferenceRunning || isOperationBusy) return;
    onRunInference({ resize: resizeEnabled, top_k: topK });
  };

  return (
    <Card className="border-surface-border bg-surface/90 flex flex-col justify-between">
      <div>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Badge variant="accent" className="font-mono text-[10px]">
                Phase 2
              </Badge>
              <CardTitle className="text-sm font-bold text-text-primary">
                Input Image Workspace
              </CardTitle>
            </div>
            <span className="text-[11px] font-mono text-text-muted">
              Target: 32×32 RGB
            </span>
          </div>
          <CardDescription className="text-xs">
            Upload an arbitrary image (up to 2048px, 10 MB) or select a pre-packaged CIFAR-10 test sample.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Hidden native input */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept={SYSTEM_LIMITS.supportedExtensions.join(',')}
            className="hidden"
            aria-label="Upload input image"
          />

          {/* Drag & Drop or Active Preview */}
          {!activeImage ? (
            <div
              role="button"
              tabIndex={0}
              aria-label="Upload input image by browsing or drag and drop"
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  fileInputRef.current?.click();
                }
              }}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`p-6 border-2 border-dashed rounded-xl flex flex-col items-center justify-center text-center cursor-pointer transition-all outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface ${
                isDragOver
                  ? 'border-accent bg-accent/10 scale-[0.99]'
                  : 'border-surface-border hover:border-accent/50 hover:bg-surface-secondary/40'
              }`}
            >
              <div className="p-3 bg-surface-secondary rounded-full border border-surface-border mb-3 text-accent">
                <UploadCloud className="w-6 h-6" />
              </div>
              <p className="text-xs font-semibold text-text-primary">
                Click to browse or drag and drop an image
              </p>
              <p className="text-[11px] text-text-secondary mt-1 max-w-xs">
                Supports PNG, JPEG, WebP, BMP, and TIFF up to 10 MB. Pre-raster safety bounds input to 2048px.
              </p>
            </div>
          ) : (
            <div className="p-3 rounded-xl border border-surface-border bg-surface-secondary/40 space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-20 h-20 rounded-lg overflow-hidden border border-surface-border bg-black/40 flex items-center justify-center shrink-0 relative group">
                  <img
                    src={activeImage.previewUrl}
                    alt="Uploaded preview"
                    className="w-full h-full object-contain [image-rendering:pixelated]"
                  />
                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <span className="text-[10px] text-white font-mono">
                      {activeImage.width}×{activeImage.height}
                    </span>
                  </div>
                </div>

                <div className="flex-1 min-w-0 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-xs text-text-primary truncate block" title={activeImage.file.name}>
                      {activeImage.file.name}
                    </span>
                    <button
                      onClick={handleRemoveImage}
                      disabled={isInferenceRunning}
                      className="text-text-muted hover:text-rose-400 p-1 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                      title="Remove image"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <div className="flex flex-wrap gap-1.5 pt-0.5">
                    <Badge variant="outline" className="font-mono text-[10px] py-0 px-1.5">
                      {activeImage.width}×{activeImage.height} px
                    </Badge>
                    <Badge variant="outline" className="font-mono text-[10px] py-0 px-1.5">
                      {(activeImage.sizeBytes / 1024).toFixed(1)} KB
                    </Badge>
                    {activeImage.source === 'sample' && (
                      <Badge variant="purple" className="text-[10px] py-0 px-1.5">
                        CIFAR-10 Test Sample
                      </Badge>
                    )}
                  </div>

                  <p className="text-[10px] text-text-muted pt-1">
                    {activeImage.width === 32 && activeImage.height === 32
                      ? 'Image matches native CIFAR-10 resolution (32×32).'
                      : 'Image will be bilinearly interpolated to 32×32 during preprocessing.'}
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-end gap-2 pt-1 border-t border-surface-border/60">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isInferenceRunning || isOperationBusy}
                  className="text-xs h-7 px-2.5"
                >
                  <RefreshCw className="w-3 h-3 mr-1" />
                  Replace Image
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleRemoveImage}
                  disabled={isInferenceRunning || isOperationBusy}
                  className="text-xs h-7 px-2.5 text-rose-400 hover:text-rose-300 hover:bg-rose-500/10"
                >
                  Remove
                </Button>
              </div>
            </div>
          )}

          {/* Quick CIFAR-10 Sample Selectors */}
          <div className="space-y-1.5">
            <span className="text-[11px] font-medium text-text-secondary flex items-center gap-1.5">
              <Sparkles className="w-3 h-3 text-accent" />
              Quick CIFAR-10 Benchmark Samples:
            </span>
            <div className="flex flex-wrap gap-1.5">
              {CIFAR_SAMPLES.map((sample) => (
                <button
                  key={sample.id}
                  onClick={() => handleSelectSample(sample.filename, sample.label)}
                  disabled={isInferenceRunning || isOperationBusy}
                  className={`px-2.5 py-1 rounded text-xs font-mono transition-all border ${
                    activeImage?.file.name === sample.filename
                      ? 'bg-accent/20 border-accent text-accent font-semibold'
                      : 'bg-surface-secondary/70 border-surface-border text-text-secondary hover:text-text-primary hover:border-surface-border-subtle'
                  }`}
                >
                  {sample.label}
                </button>
              ))}
            </div>
          </div>

          {/* Inference Preprocessing Controls */}
          <div className="p-3 rounded-lg bg-surface-secondary/40 border border-surface-border space-y-2.5 text-xs">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <span className="font-semibold text-text-primary block">Automatic 32×32 Resizing</span>
                <span className="text-[11px] text-text-muted block">
                  Bilinearly scales non-32×32 inputs to canonical resolution.
                </span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={resizeEnabled}
                  onChange={(e) => setResizeEnabled(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-8 h-4 bg-surface-border peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-text-secondary after:border-surface-border after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-accent peer-checked:after:bg-white" />
              </label>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-surface-border/60">
              <span className="font-semibold text-text-primary">Top-K Predictions</span>
              <div className="flex items-center gap-1">
                {[3, 5, 10].map((k) => (
                  <button
                    key={k}
                    onClick={() => setTopK(k)}
                    className={`px-2 py-0.5 rounded font-mono text-[11px] border transition-colors ${
                      topK === k
                        ? 'bg-accent text-white border-accent font-bold'
                        : 'bg-surface border-surface-border text-text-secondary hover:text-text-primary'
                    }`}
                  >
                    {k}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Validation Error Feedback */}
          {validationError && (
            <div role="alert" className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{validationError}</span>
            </div>
          )}

          {/* Inference Backend Error Feedback */}
          {inferenceError && (
            <div role="alert" className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <strong className="block font-medium">Inference Execution Failed</strong>
                <span className="text-[11px] text-rose-300 leading-snug">{inferenceError}</span>
              </div>
            </div>
          )}
        </CardContent>
      </div>

      {/* Action Footer */}
      <div className="p-4 bg-surface-secondary/40 border-t border-surface-border space-y-2">
        <Button
          variant="accent"
          size="lg"
          onClick={handleTriggerPrediction}
          disabled={!activeImage || !isModelLoaded || isInferenceRunning || isOperationBusy || !isBackendConnected}
          className="w-full text-xs font-bold h-11"
          title={
            !isBackendConnected
              ? 'Backend is offline'
              : isOperationBusy
              ? 'Another operation (model load or comparative evaluation) is currently in progress'
              : !isModelLoaded
              ? 'Please load a model configuration in Phase 1 first'
              : !activeImage
              ? 'Please upload or select an input image first'
              : 'Execute classification inference through the loaded model'
          }
        >
          {isInferenceRunning ? (
            <>
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              Evaluating Predictions on Device...
            </>
          ) : (
            <>
              <Zap className="w-4 h-4 mr-2" />
              Run Live Prediction (POST /predict)
            </>
          )}
        </Button>

        {!isModelLoaded && isBackendConnected && (
          <p className="text-[11px] text-amber-400 text-center flex items-center justify-center gap-1">
            <Info className="w-3 h-3" />
            Inference requires a resident model. Click "Load Model" above.
          </p>
        )}
      </div>
    </Card>
  );
}
