import { useEffect, useState } from 'react';
import { API_BASE_URL, getApiUrl } from '@/config/api';
import { Server } from 'lucide-react';

type ApiStatus = 'unverified' | 'connected' | 'disconnected';

export function StatusIndicator() {
  const [status, setStatus] = useState<ApiStatus>('unverified');

  useEffect(() => {
    let mounted = true;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    fetch(getApiUrl('health'), { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (mounted) {
          if (data?.status === 'healthy') {
            setStatus('connected');
          } else {
            setStatus('disconnected');
          }
        }
      })
      .catch(() => {
        if (mounted) {
          setStatus('disconnected');
        }
      })
      .finally(() => {
        clearTimeout(timeoutId);
      });

    return () => {
      mounted = false;
      controller.abort();
      clearTimeout(timeoutId);
    };
  }, []);

  return (
    <div
      className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-surface-secondary border border-surface-border text-xs"
      title={`Target Service: ${API_BASE_URL} (${status})`}
    >
      {status === 'connected' ? (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
        </span>
      ) : status === 'disconnected' ? (
        <span className="relative flex h-2 w-2">
          <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
        </span>
      ) : (
        <span className="relative flex h-2 w-2">
          <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-400"></span>
        </span>
      )}
      <Server className="w-3.5 h-3.5 text-text-muted" />
      <span className="text-[11px] font-mono text-text-secondary hidden sm:inline">
        API: <span className="text-text-primary">{API_BASE_URL.replace(/^https?:\/\//, '')}</span>
      </span>
      {status === 'connected' ? (
        <span className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wider bg-emerald-500/10 px-1.5 py-0.2 rounded border border-emerald-500/20">
          Connected
        </span>
      ) : status === 'disconnected' ? (
        <span className="text-[10px] font-semibold text-rose-400 uppercase tracking-wider bg-rose-500/10 px-1.5 py-0.2 rounded border border-rose-500/20">
          Disconnected
        </span>
      ) : (
        <span className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider bg-amber-500/10 px-1.5 py-0.2 rounded border border-amber-500/20">
          Unverified
        </span>
      )}
    </div>
  );
}
