import { getUserErrorMessage } from '../api/checkedFetch';

export interface BatchResult {
  name: string;
  status: string;
  exit_code: number | null;
  error: string | null;
  run_id?: string;
}

export interface BatchStatus {
  batch_id: string;
  status: 'pending' | 'running' | 'paused' | 'stopping' | 'completed' | 'failed' | 'cancelled';
  current_index: number;
  current_exp_name: string | null;
  total: number;
  error: string | null;
  pause_requested?: boolean;
  completed_count: number;
  success_count: number;
  failed_count: number;
  cancelled_count: number;
  unknown_count?: number;
  results: BatchResult[];
  revision?: number;
}

export interface BatchMetricProgress {
  runId: string;
  round: number | null;
  total: number | null;
  percent: number | null;
  status: string;
}

export const isBatchActive = (status: string): boolean =>
  ['pending', 'running', 'paused', 'stopping'].includes(status);

// New backends provide a monotonic revision. Keep conservative guards for older
// servers too: neither terminal states nor completed work may move backwards.
export const shouldAcceptBatchStatus = (previous: BatchStatus | null, next: BatchStatus): boolean => {
  if (!previous) return true;
  if (previous.batch_id !== next.batch_id) return false;
  if (!isBatchActive(previous.status) && isBatchActive(next.status)) return false;
  if (previous.revision != null && next.revision != null) return next.revision >= previous.revision;
  return (next.completed_count ?? 0) >= (previous.completed_count ?? 0)
    && (next.current_index ?? -1) >= (previous.current_index ?? -1);
};

export const getBatchResultDisplay = (result?: BatchResult, percent: number | null = null) => {
  switch (result?.status) {
    case 'running': return { label: `正在进行，进度 ${percent ?? 0}%`, color: 'processing', detail: '' };
    case 'completed': return { label: '已完成', color: 'success', detail: '' };
    case 'failed': {
      const reason = result.error ? getUserErrorMessage(result.error, '请查看运行日志。') : '请查看运行日志。';
      const code = result.exit_code != null && !reason.includes(`退出码 ${result.exit_code}`) ? `退出码 ${result.exit_code}` : '';
      return { label: '失败', color: 'error', detail: [code, reason].filter(Boolean).join(' · ') };
    }
    case 'cancelled': return { label: '已取消', color: 'default', detail: result.error ? getUserErrorMessage(result.error) : '' };
    case 'unknown': return { label: '未记录', color: 'default', detail: '旧版本未记录执行明细，无法确认该实验的执行状态。' };
    default: return { label: '等待中', color: 'default', detail: '' };
  }
};