import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { History, Trash2, ChevronRight } from 'lucide-react';
import type { SinglePredictionResponse } from '@/api';

export interface HistoryRecord {
  id: string;
  filename: string;
  previewUrl: string;
  modelName: string;
  precision: string;
  device: string;
  predictedClass: string;
  confidence: number;
  totalMs: number;
  timestamp: string;
  fullResult: SinglePredictionResponse;
}

interface SessionInferenceHistoryProps {
  history: HistoryRecord[];
  onSelectRecord: (record: HistoryRecord) => void;
  onClearHistory: () => void;
  activeResultId?: string;
}

export function SessionInferenceHistory({
  history,
  onSelectRecord,
  onClearHistory,
  activeResultId,
}: SessionInferenceHistoryProps) {
  if (history.length === 0) {
    return (
      <Card className="border-surface-border bg-surface/90">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <History className="w-4 h-4 text-text-muted" />
              <CardTitle className="text-sm font-bold text-text-primary">
                Session Prediction Log
              </CardTitle>
            </div>
            <Badge variant="outline" className="font-mono text-[10px]">
              0 Records
            </Badge>
          </div>
          <CardDescription className="text-xs">
            In-memory evaluation history for this workstation session.
          </CardDescription>
        </CardHeader>
        <CardContent className="py-6 text-center text-xs text-text-muted">
          No predictions recorded in this session yet. Run an inference to populate the log.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-surface-border bg-surface/90">
      <CardHeader className="pb-3 border-b border-surface-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <History className="w-4 h-4 text-accent" />
            <CardTitle className="text-sm font-bold text-text-primary">
              Session Prediction Log
            </CardTitle>
            <Badge variant="accent" className="font-mono text-[10px]">
              {history.length} {history.length === 1 ? 'Record' : 'Records'}
            </Badge>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClearHistory}
            className="text-xs h-7 px-2 text-text-muted hover:text-rose-400"
            title="Clear all session history"
          >
            <Trash2 className="w-3.5 h-3.5 mr-1" />
            Clear Log
          </Button>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-surface-secondary/70 text-text-muted font-mono uppercase text-[10px] border-b border-surface-border">
              <tr>
                <th className="py-2 px-3">Time</th>
                <th className="py-2 px-3">Image</th>
                <th className="py-2 px-3">Model & Precision</th>
                <th className="py-2 px-3">Predicted Class</th>
                <th className="py-2 px-3 text-right">Confidence</th>
                <th className="py-2 px-3 text-right">Latency</th>
                <th className="py-2 px-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border font-mono text-[11px]">
              {history.map((rec) => {
                const isActive = activeResultId === rec.id;
                return (
                  <tr
                    key={rec.id}
                    onClick={() => onSelectRecord(rec)}
                    className={`cursor-pointer transition-colors ${
                      isActive ? 'bg-accent/10 border-l-2 border-accent' : 'hover:bg-surface-hover/30'
                    }`}
                  >
                    <td className="py-2 px-3 text-text-muted text-[10px] whitespace-nowrap">
                      {rec.timestamp}
                    </td>
                    <td className="py-2 px-3">
                      <div className="flex items-center gap-2">
                        <img
                          src={rec.previewUrl}
                          alt="Thumb"
                          className="w-5 h-5 rounded object-cover border border-surface-border [image-rendering:pixelated]"
                        />
                        <span className="font-sans font-medium text-text-primary truncate max-w-[120px]" title={rec.filename}>
                          {rec.filename}
                        </span>
                      </div>
                    </td>
                    <td className="py-2 px-3">
                      <div className="flex items-center gap-1">
                        <span className="font-sans text-text-secondary">
                          {rec.modelName === 'resnet18_cifar' ? 'ResNet' : 'MobileNet'}
                        </span>
                        <Badge variant="outline" className="text-[9px] py-0 px-1 uppercase">
                          {rec.precision}
                        </Badge>
                      </div>
                    </td>
                    <td className="py-2 px-3">
                      <span className="font-bold text-text-primary capitalize font-sans">
                        {rec.predictedClass}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-right">
                      <span className={rec.confidence >= 0.8 ? 'text-emerald-400 font-bold' : 'text-accent'}>
                        {(rec.confidence * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2 px-3 text-right text-text-secondary">
                      {rec.totalMs.toFixed(2)} ms
                    </td>
                    <td className="py-2 px-3 text-right">
                      <span className="text-accent text-[10px] hover:underline inline-flex items-center">
                        View
                        <ChevronRight className="w-3 h-3 ml-0.5" />
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
