/**
 * Formatting utilities for experimental dashboard metrics.
 */

export function formatPercent(value: number, decimals = 2): string {
  if (value === null || value === undefined || isNaN(value)) return 'N/A';
  return `${value.toFixed(decimals)}%`;
}

export function formatDrop(value: number, decimals = 2): string {
  if (value === null || value === undefined || isNaN(value)) return 'N/A';
  return `-${Math.abs(value).toFixed(decimals)} pp`;
}

export function formatNumber(value: number): string {
  if (value === null || value === undefined || isNaN(value)) return 'N/A';
  return new Intl.NumberFormat('en-US').format(value);
}

export function formatMB(bytes: number, decimals = 1): string {
  if (bytes === null || bytes === undefined || isNaN(bytes)) return 'N/A';
  const mb = bytes / (1000 * 1000);
  return `${mb.toFixed(decimals)} MB`;
}

export function formatRatio(ratio: number, decimals = 2): string {
  if (ratio === null || ratio === undefined || isNaN(ratio)) return 'N/A';
  return `${ratio.toFixed(decimals)}×`;
}

export function formatMs(ms: number, decimals = 2): string {
  if (ms === null || ms === undefined || isNaN(ms)) return 'N/A';
  return `${ms.toFixed(decimals)} ms`;
}

export function formatFps(fps: number, decimals = 1): string {
  if (fps === null || fps === undefined || isNaN(fps)) return 'N/A';
  return `${new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(fps)} img/s`;
}
