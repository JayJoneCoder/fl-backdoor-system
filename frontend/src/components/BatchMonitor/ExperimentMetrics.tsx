import React, { useEffect, useRef, useState } from 'react';
import { Alert, Button, Card, Col, Row, Typography } from 'antd';
import * as echarts from 'echarts';
import { getWebSocketUrl } from '../../api/endpoints';
import { getUserErrorMessage } from '../../api/checkedFetch';
import { getProgressPercent, mergeMetricPoint, parseMetricPoint, parseTotalRounds } from '../../utils/experimentStream';
import type { MetricPoint } from '../../utils/experimentStream';
import type { BatchMetricProgress } from '../../utils/batchStream';

interface Props {
  name: string;
  runId: string;
  onProgress?: (progress: BatchMetricProgress) => void;
}

// Uses the same replayable CSV/status stream and metric normalization as the
// single-experiment page. A run-scoped key prevents data leaking across runs.
const ExperimentMetrics: React.FC<Props> = ({ name, runId, onProgress }) => {
  const accuracyRef = useRef<HTMLDivElement>(null);
  const lossRef = useRef<HTMLDivElement>(null);
  const callbackRef = useRef(onProgress);
  const [connectionError, setConnectionError] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [hasMetrics, setHasMetrics] = useState(false);
  const [streamStatus, setStreamStatus] = useState('running');
  const reconnectRef = useRef<(() => void) | null>(null);
  useEffect(() => { callbackRef.current = onProgress; }, [onProgress]);

  useEffect(() => {
    if (!accuracyRef.current || !lossRef.current) return;
    const accuracyChart = echarts.init(accuracyRef.current);
    const lossChart = echarts.init(lossRef.current);
    const options = (names: string[], color: string[]) => ({
      color,
      tooltip: { trigger: 'axis' as const },
      legend: { data: names, bottom: 4, left: 'center' },
      grid: { left: 50, right: 24, top: 24, bottom: 86 },
      xAxis: { type: 'category' as const, name: '轮次', nameLocation: 'middle' as const, nameGap: 30, data: [] as number[] },
      yAxis: { type: 'value' as const },
      series: names.map(seriesName => ({ name: seriesName, type: 'line' as const, showSymbol: false, data: [] as number[] })),
    });
    accuracyChart.setOption(options(['准确率 (ACC)', '攻击成功率 (ASR)'], ['#2563b8', '#c17a37']));
    lossChart.setOption(options(['损失 (Loss)'], ['#398673']));
    let history: MetricPoint[] = [];
    let round: number | null = null;
    let total: number | null = null;
    let state = 'running';
    let active = true;
    let socket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let frame: number | null = null;
    const isTerminal = () => ['finished', 'completed', 'failed', 'cancelled'].includes(state);
    const publish = () => callbackRef.current?.({ runId, round, total, percent: getProgressPercent(round, total), status: state });
    const paint = () => {
      if (frame !== null) return;
      frame = requestAnimationFrame(() => {
        frame = null;
        const rounds = history.map(point => point.round);
        accuracyChart.setOption({ xAxis: { data: rounds }, series: [
          { name: '准确率 (ACC)', data: history.map(point => point.accuracy) },
          { name: '攻击成功率 (ASR)', data: history.map(point => point.asr) },
        ] });
        lossChart.setOption({ xAxis: { data: rounds }, series: [{ name: '损失 (Loss)', data: history.map(point => point.loss) }] });
      });
    };
    const connect = () => {
      if (!active) return;
      const connection = new WebSocket(getWebSocketUrl(`/ws/logs/${encodeURIComponent(name)}?run_id=${encodeURIComponent(runId)}`));
      socket = connection;
      connection.onopen = () => { if (active && socket === connection) setConnectionError(false); };
      connection.onmessage = event => {
        if (!active || socket !== connection) return;
        try {
          const packet = JSON.parse(event.data);
          if (packet.type === 'status') {
            if (!isTerminal()) state = packet.data.status;
            setStreamStatus(state);
            setConnectionError(false);
            total = parseTotalRounds(packet.data.total_rounds) ?? total;
            round ??= 0;
            setStreamError(packet.data.error ? getUserErrorMessage(packet.data.error) : null);
            publish();
          } else if (packet.type === 'csv_update') {
            const point = parseMetricPoint(packet.data);
            if (!point) return;
            history = mergeMetricPoint(history, point);
            round = history.at(-1)!.round;
            setHasMetrics(true);
            publish();
            paint();
          }
        } catch {
          // This connection also carries plain log lines; batch logs are shown separately.
        }
      };
      connection.onerror = () => { if (active && socket === connection && !isTerminal()) setConnectionError(true); };
      connection.onclose = () => {
        if (!active || socket !== connection) return;
        // Do not close on terminal status: final CSV rows can still follow it.
        // Once the server closes a terminal stream, no reconnect is necessary.
        if (isTerminal()) { setConnectionError(false); return; }
        setConnectionError(true);
        retry = setTimeout(connect, 2000);
      };
    };
    reconnectRef.current = () => { clearTimeout(retry); socket?.close(); connect(); };
    connect();
    const resize = () => { accuracyChart.resize(); lossChart.resize(); };
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize);
    observer?.observe(accuracyRef.current);
    observer?.observe(lossRef.current);
    window.addEventListener('resize', resize);
    return () => {
      active = false;
      reconnectRef.current = null;
      clearTimeout(retry);
      socket?.close();
      if (frame !== null) cancelAnimationFrame(frame);
      observer?.disconnect();
      window.removeEventListener('resize', resize);
      accuracyChart.dispose();
      lossChart.dispose();
    };
  }, [name, runId]);

  return (
    <div style={{ width: '100%' }}>
      <Typography.Text type="secondary">指标实验：{name}（切换到下一项时自动清空，结束后保留最后一项曲线）</Typography.Text>
      {connectionError && <Alert type="warning" showIcon title="指标连接中断，正在自动重连；已收到的数据会保留。"
        action={<Button size="small" onClick={() => reconnectRef.current?.()}>重连指标</Button>} />}
      {streamError && <Alert type="error" showIcon title={streamError} />}
      {!hasMetrics && <Typography.Paragraph type="secondary">{['finished', 'completed', 'failed', 'cancelled'].includes(streamStatus)
        ? '该实验已结束，未生成可用指标；请查看实验状态及运行日志。' : '等待首轮指标，数据到达后自动绘图。'}</Typography.Paragraph>}
      <Row gutter={[16, 16]}>
        <Col xs={24} xl={12}><Card size="small" title="ACC / ASR"><div ref={accuracyRef} role="img" aria-label="ACC 与 ASR 实时曲线" style={{ width: '100%', height: 260 }} /></Card></Col>
        <Col xs={24} xl={12}><Card size="small" title="Loss"><div ref={lossRef} role="img" aria-label="Loss 实时曲线" style={{ width: '100%', height: 260 }} /></Card></Col>
      </Row>
    </div>
  );
};

export default ExperimentMetrics;