import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button, Space, Input, message, Card, Tag, Alert, Progress } from 'antd';
import { PlayCircleOutlined, StopOutlined, SyncOutlined } from '@ant-design/icons';
import * as echarts from 'echarts';
import { startExperiment, stopExperiment, getExperimentStatus, getConfig } from '../../api/client';
import { getWebSocketUrl } from '../../api/endpoints';
import LogViewer from '../../components/LogViewer';
import { useBufferedLogs } from '../../hooks/useBufferedLogs';
import { getProgressPercent, isExperimentBusy, mergeMetricPoint, normalizeExperimentStatus, parseMetricPoint, parseTotalRounds } from '../../utils/experimentStream';
import type { ExperimentStatus, MetricPoint } from '../../utils/experimentStream';

interface ExperimentControlProps {
  onBeforeStart?: () => Promise<{ action: 'cancel' | 'save' | 'backup' }>;
}

const ExperimentControl: React.FC<ExperimentControlProps> = ({ onBeforeStart }) => {
  const [expName, setExpName] = useState('test_exp');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<ExperimentStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  const { logs, queueLog, flushLogs, resetLogs } = useBufferedLogs();
  const wsRef = useRef<WebSocket | null>(null);
  const connectedNameRef = useRef<string | null>(null);
  const startPendingRef = useRef(false);

  const [currentRound, setCurrentRound] = useState<number | null>(null);
  const [totalRound, setTotalRound] = useState<number | null>(null);

  const chartRef = useRef<HTMLDivElement>(null);          // 主图 ACC + ASR
  const chartInstance = useRef<echarts.ECharts | null>(null);

  const lossChartRef = useRef<HTMLDivElement>(null);      // 副图 Loss
  const lossChartInstance = useRef<echarts.ECharts | null>(null);

  // Keep replayed data even before initialization or while painting is paused.
  const metricHistoryRef = useRef<MetricPoint[]>([]);
  const renderFrameRef = useRef<number | null>(null);

  const renderCharts = useCallback(() => {
    const history = metricHistoryRef.current;
    const rounds = history.map(point => point.round);
    chartInstance.current?.setOption({
      xAxis: { data: rounds },
      series: [
        { name: 'ACC', data: history.map(point => point.accuracy) },
        { name: 'ASR', data: history.map(point => point.asr) },
      ],
    });
    lossChartInstance.current?.setOption({
      xAxis: { data: rounds },
      series: [{ name: 'Loss', data: history.map(point => point.loss) }],
    });
  }, []);

  const scheduleChartRender = useCallback(() => {
    if (renderFrameRef.current !== null) return;
    renderFrameRef.current = requestAnimationFrame(() => {
      renderFrameRef.current = null;
      renderCharts();
    });
  }, [renderCharts]);

  const resetCharts = useCallback(() => {
    if (renderFrameRef.current !== null) cancelAnimationFrame(renderFrameRef.current);
    renderFrameRef.current = null;
    metricHistoryRef.current = [];
    renderCharts();
  }, [renderCharts]);

  // ---------- 初始化两张图表 ----------
  useEffect(() => {
    const rid = requestAnimationFrame(() => {
      // 主图
      if (chartRef.current && !chartInstance.current) {
        chartInstance.current = echarts.init(chartRef.current);
        chartInstance.current.setOption({
          xAxis: { type: 'category', data: [] },
          yAxis: { type: 'value' },
          tooltip: { trigger: 'axis' },
          legend: { data: ['ACC', 'ASR'] },            // ★ 固定图例
          series: [
            { name: 'ACC', type: 'line', data: [] },
            { name: 'ASR', type: 'line', data: [] },
          ],
        });
      }
      // 副图（Loss）
      if (lossChartRef.current && !lossChartInstance.current) {
        lossChartInstance.current = echarts.init(lossChartRef.current);
        lossChartInstance.current.setOption({
          xAxis: { type: 'category', data: [] },
          yAxis: { type: 'value' },
          tooltip: { trigger: 'axis' },
          legend: { data: ['Loss'] },
          series: [
            { name: 'Loss', type: 'line', data: [] },
          ],
        });
      }
      // History may have arrived while initialization was still waiting for a frame.
      renderCharts();
    });

    const handleResize = () => {
      requestAnimationFrame(() => {
        chartInstance.current?.resize();
        lossChartInstance.current?.resize();
      });
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(rid);
      if (renderFrameRef.current !== null) cancelAnimationFrame(renderFrameRef.current);
      renderFrameRef.current = null;
      chartInstance.current?.dispose();
      chartInstance.current = null;
      lossChartInstance.current?.dispose();
      lossChartInstance.current = null;
      window.removeEventListener('resize', handleResize);
    };
  }, [renderCharts]);

  // ---------- WebSocket 连接 ----------
  const connectWS = useCallback((name: string) => {
    if (wsRef.current) wsRef.current.close();

    // Each connection replays the complete run. Start a fresh display to avoid
    // duplicated logs and mixing one experiment's history with another's.
    resetLogs();
    resetCharts();
    setCurrentRound(null);
    setTotalRound(null);

    const socket = new WebSocket(getWebSocketUrl(`/ws/logs/${encodeURIComponent(name)}`));
    wsRef.current = socket;
    connectedNameRef.current = name;

    socket.onmessage = (event) => {
      if (wsRef.current !== socket) return;
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === 'status') {
          const nextStatus = normalizeExperimentStatus(msg.data.status);
          setStatus(nextStatus);
          setError(msg.data.error ?? null);
          const total = parseTotalRounds(msg.data.total_rounds);
          if (total !== null) setTotalRound(total);
          if (nextStatus === 'running') setCurrentRound(previous => previous ?? 0);
          else flushLogs();
          return;
        }

        if (msg.type === 'csv_update' && msg.data) {
          const point = parseMetricPoint(msg.data);
          if (point) {
            metricHistoryRef.current = mergeMetricPoint(metricHistoryRef.current, point);
            setCurrentRound(metricHistoryRef.current.at(-1)!.round);
            scheduleChartRender();
          }
          return;
        }
      } catch {
        // Preserve ANSI color escapes, but remove terminal-only backspace/VT/FF.
        // eslint-disable-next-line no-control-regex
        const cleanLine = event.data.replace(/[\x08\x0B\x0C]/g, '');
        queueLog(cleanLine);
      }
    };

    socket.onclose = () => {
      if (wsRef.current === socket) {
        flushLogs();
        wsRef.current = null;
      }
    };
    socket.onerror = () => {
      if (wsRef.current === socket) queueLog('[System] 日志连接异常，正在等待重连。');
    };
  }, [scheduleChartRender, resetCharts, resetLogs, queueLog, flushLogs]);

  // ---------- 启动 / 停止 ----------
  const handleStart = async () => {
    if (isExperimentBusy(status)) return;
    if (!expName.trim()) { message.error('请输入实验名称'); return; }
    if (onBeforeStart) {
      const result = await onBeforeStart();
      if (result.action === 'cancel') return;
    }
    setLoading(true);
    startPendingRef.current = true;
    try {
      const config = await getConfig();
      const total = parseTotalRounds(config.data['num-server-rounds']);
      wsRef.current?.close();
      wsRef.current = null;
      resetLogs();
      setCurrentRound(0);
      setTotalRound(total);
      resetCharts();

      const name = expName.trim();
      // Connect after the old log has been truncated. History replay fills the gap.
      await startExperiment(name);
      setStatus('running');
      connectWS(name);
      setCurrentRound(0);
      setTotalRound(total);
      message.success(`实验 "${name}" 已启动`);
    } catch {
      message.error('启动失败');
    } finally {
      startPendingRef.current = false;
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    setStatus('stopping');
    try {
      const response = await stopExperiment();
      const nextStatus = normalizeExperimentStatus(response.data.status);
      setStatus(nextStatus);
      if (!isExperimentBusy(nextStatus)) message.success('实验已停止并完成清理');
    } catch (e: unknown) {
      message.error(e instanceof Error ? e.message : '停止失败');
      // The server may still be cleaning. Polling, not a failed request, releases
      // the controls; never advertise idle while owned processes remain.
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let disposed = false;
    let checking = false;
    const checkStatus = async () => {
      if (checking || startPendingRef.current) return;
      checking = true;
      const observedSocket = wsRef.current;
      try {
        const res = await getExperimentStatus();
        if (disposed || startPendingRef.current || wsRef.current !== observedSocket) return;
        const nextStatus = normalizeExperimentStatus(res.data.status);
        setStatus(nextStatus);
        setError(res.data.error ?? null);
        if (nextStatus !== 'idle' && res.data.exp_name) {
          setExpName(res.data.exp_name);
          if (!wsRef.current || connectedNameRef.current !== res.data.exp_name) {
            connectWS(res.data.exp_name);
          }
        }
        const total = parseTotalRounds(res.data.total_rounds);
        if (total !== null) setTotalRound(total);
      } catch {
        // A transient backend failure is retried on the next polling interval.
      } finally {
        checking = false;
      }
    };
    void checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => {
      disposed = true;
      clearInterval(interval);
      wsRef.current?.close();
      wsRef.current = null;
      connectedNameRef.current = null;
    };
  }, [connectWS]);

  const progress = getProgressPercent(currentRound, totalRound);
  const busy = isExperimentBusy(status);
  const statusLabel: Record<ExperimentStatus, string> = {
    idle: '空闲', running: '运行中', stopping: '停止／清理中', cleaning: '清理中',
    cleanup_failed: '清理失败（请重试）', finished: '已完成', failed: '失败', cancelled: '已取消',
  };

  return (
    <Card title="实验控制" style={{ marginTop: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space>
          <Input value={expName} onChange={e => setExpName(e.target.value)}
                 style={{ width: 200 }} disabled={busy} />
          <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleStart}
                  loading={loading && !busy} disabled={busy}>
            启动实验
          </Button>
          <Button danger icon={<StopOutlined />} onClick={handleStop}
                  loading={loading && busy} disabled={!['running', 'cleanup_failed'].includes(status)}>
            {status === 'cleanup_failed' ? '重试清理' : '停止实验'}
          </Button>
          <Tag color={busy ? 'processing' : status === 'finished' ? 'success' : status === 'failed' ? 'error' : 'default'}>
            {busy && <SyncOutlined spin />} {statusLabel[status]}
          </Tag>
        </Space>

        {status === 'running' && <Alert message="实验运行中" type="info" showIcon />}
        {error && <Alert message={error} type="error" showIcon />}

        {progress !== null && (
          <div style={{ marginBottom: 10 }}>
            <Progress percent={progress} status={status === 'running' ? 'active' : 'normal'} />
            <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>ROUND {currentRound} / {totalRound}</div>
          </div>
        )}

        {/* 主图：ACC + ASR */}
        <div ref={chartRef} style={{ width: '100%', height: 250, marginBottom: 10 }} />

        {/* 副图：Loss */}
        <div ref={lossChartRef} style={{ width: '100%', height: 250 }} />

        <LogViewer logs={logs} height={300} />
      </Space>
    </Card>
  );
};

export default ExperimentControl;