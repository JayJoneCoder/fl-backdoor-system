export const takeLogBatch = (buffer: { current: string[] }): string[] => {
  const batch = buffer.current;
  buffer.current = [];
  return batch;
};

// The state updater only reads this detached batch, so re-evaluation is safe.
export const mergeLogLines = (
  previous: string[],
  batch: string[],
  limit: number = 1000,
): string[] => [...previous, ...batch].slice(-limit);

const parseInteger = (value: unknown, minimum: number): number | null => {
  if (typeof value !== 'number' && typeof value !== 'string') return null;
  if (typeof value === 'string' && value.trim() === '') return null;
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= minimum ? number : null;
};

export const parseRound = (value: unknown): number | null => parseInteger(value, 0);
export const parseTotalRounds = (value: unknown): number | null => parseInteger(value, 1);

export interface MetricPoint {
  round: number;
  accuracy: number;
  asr: number;
  loss: number;
}

export const parseMetricPoint = (value: unknown): MetricPoint | null => {
  if (!value || typeof value !== 'object') return null;
  const row = value as Record<string, unknown>;
  const round = parseRound(row.round);
  if (round === null) return null;
  const metrics: number[] = [];
  for (const key of ['accuracy', 'asr', 'loss']) {
    const raw = row[key];
    if (typeof raw !== 'number' && typeof raw !== 'string') return null;
    if (typeof raw === 'string' && raw.trim() === '') return null;
    const number = Number(raw);
    if (!Number.isFinite(number)) return null;
    metrics.push(number);
  }
  return { round, accuracy: metrics[0], asr: metrics[1], loss: metrics[2] };
};

// The history, rather than ECharts' current options, is the source of truth.
// Replayed/out-of-order rows are sorted and update an existing round, not duplicated.
export const mergeMetricPoint = (history: MetricPoint[], point: MetricPoint): MetricPoint[] =>
  [...history.filter(previous => previous.round !== point.round), point]
    .sort((left, right) => left.round - right.round);

export const getProgressPercent = (
  round: number | null,
  total: number | null,
): number | null => {
  if (round === null || total === null || total <= 0) return null;
  return Math.round(Math.min(1, Math.max(0, round / total)) * 100);
};

export const normalizeExperimentStatus = (value: unknown): 'idle' | 'running' | 'finished' => {
  switch (String(value ?? '').toLowerCase()) {
    case 'starting':
    case 'running':
      return 'running';
    case 'completed':
    case 'finished':
      return 'finished';
    default:
      return 'idle';
  }
};