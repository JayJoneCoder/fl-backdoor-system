import React, { useState, useEffect, useRef } from 'react';
import { Card, Progress, Tag, Alert, Space, Typography, Row, Col, Button } from 'antd';
import { SyncOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import LogViewer from '../LogViewer';
import { getWebSocketUrl } from '../../api/endpoints';
import { useBufferedLogs } from '../../hooks/useBufferedLogs';
import { getUserErrorMessage } from '../../api/checkedFetch';
import ExperimentMetrics from './ExperimentMetrics';
import { isBatchActive, shouldAcceptBatchStatus } from '../../utils/batchStream';
import type { BatchStatus, BatchMetricProgress } from '../../utils/batchStream';
export type { BatchStatus } from '../../utils/batchStream';

const { Text } = Typography;
interface Props {
  batchId: string;
  onComplete?: (status: BatchStatus, notify: boolean) => void;
  onStatus?: (status: BatchStatus) => boolean | void;
  onProgress?: (progress: BatchMetricProgress) => void;
}

const BatchMonitor: React.FC<Props> = ({ batchId, onComplete, onStatus, onProgress }) => {
  const [status, setStatus] = useState<BatchStatus | null>(null);
  const [metricProgress, setMetricProgress] = useState<BatchMetricProgress | null>(null);
  const [connectionError, setConnectionError] = useState(false);
  const [logsCollapsed, setLogsCollapsed] = useState(false);
  const { logs, queueLog, flushLogs, resetLogs } = useBufferedLogs();

  const onCompleteRef = useRef(onComplete);
  const onStatusRef = useRef(onStatus);
  const reconnectRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    onCompleteRef.current = onComplete;
    onStatusRef.current = onStatus;
  }, [onComplete, onStatus]);

  useEffect(() => {
    let active = true;
    let completed = false;
    let observedActive = false;
    let socket: WebSocket | null = null;
    let lastStatus: BatchStatus | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      if (!active) return;
      resetLogs();
      const connection = new WebSocket(getWebSocketUrl(`/ws/batch/${encodeURIComponent(batchId)}`));
      socket = connection;
      connection.onopen = () => { if (active && socket === connection) setConnectionError(false); };

      connection.onmessage = (event) => {
        if (!active || socket !== connection) return;
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'status') {
            if (data.data.batch_id !== batchId || !shouldAcceptBatchStatus(lastStatus, data.data)) return;
            // The parent may already have a newer HTTP control response.
            if (onStatusRef.current?.(data.data) === false) return;
            lastStatus = data.data;
            setConnectionError(false);
            setStatus(data.data);
            if (isBatchActive(data.data.status)) observedActive = true;
            if (!completed && ['completed', 'failed', 'cancelled'].includes(data.data.status)) {
              completed = true;
              flushLogs();
              // Restore terminal snapshots silently, but always release the parent's busy state.
              onCompleteRef.current?.(data.data, observedActive);
            }
            return;
          }
        } catch {
          // Preserve ANSI color escapes, but remove terminal-only backspace/VT/FF.
          // eslint-disable-next-line no-control-regex
          const clean = event.data.replace(/[\x08\x0B\x0C]/g, '');
          queueLog(clean);
        }
      };

      connection.onerror = () => { if (active && socket === connection && !completed) setConnectionError(true); };
      connection.onclose = () => {
        if (!active || socket !== connection) return;
        flushLogs();
        if (!completed) {
          setConnectionError(true);
          retry = setTimeout(connect, 2000);
        }
      };
    };
    reconnectRef.current = () => { clearTimeout(retry); socket?.close(); connect(); };
    connect();
    return () => {
      active = false;
      reconnectRef.current = null;
      clearTimeout(retry);
      socket?.close();
    };
  }, [batchId, queueLog, flushLogs, resetLogs]);

  const done = status?.completed_count ?? 0;
  const progress = status && status.total > 0 ? Math.round(done / status.total * 100) : 0;
  const current = status?.current_exp_name ? status.results?.[status.current_index] : undefined;
  const latestRun = status?.results?.filter(item => item.run_id).at(-1);
  const target = current ?? latestRun;
  const targetProgress = metricProgress?.runId === target?.run_id ? metricProgress : null;
  const currentPercent = target?.status === 'completed' ? 100 : (targetProgress?.percent ?? 0);

  const renderTag = () => {
    if (!status) return null;
    if (status.status === 'running') return <Tag icon={<SyncOutlined spin />} color="processing">运行中</Tag>;
    if (status.status === 'paused') return <Tag color="warning">队列已暂停</Tag>;
    if (status.status === 'stopping') return <Tag icon={<SyncOutlined spin />} color="warning">停止／清理中</Tag>;
    if (status.status === 'cancelled') return <Tag color="default">已取消</Tag>;
    if (status.status === 'completed') return <Tag icon={<CheckCircleOutlined />} color="success">已完成</Tag>;
    if (status.status === 'failed') return <Tag icon={<CloseCircleOutlined />} color="error">失败</Tag>;
    return <Tag>等待中</Tag>;
  };

  return (
    <Card title="批量实验监控" style={{ marginTop: 24 }}>
      <Space orientation="vertical" size="middle" style={{ width: '100%' }}>
        {connectionError && <Alert title="监控连接中断，正在自动重连。" type="warning" showIcon
          action={<Button size="small" onClick={() => reconnectRef.current?.()}>重连监控</Button>} />}
        {status && (
          <>
            <Space wrap>
              <Text strong>状态：</Text>
              {renderTag()}
              <Text>已结束：{done}/{status.total}</Text>
              <Text>成功 {status.success_count ?? 0} / 失败 {status.failed_count ?? 0} / 取消 {status.cancelled_count ?? 0}</Text>
              {!!status.unknown_count && <Text type="secondary">状态未记录 {status.unknown_count}</Text>}
              {status.current_exp_name && <Text>当前实验：{status.current_exp_name}</Text>}
            </Space>

            <Row gutter={[24, 16]}>
              <Col xs={24} md={12}>
                <Text strong>全批次实验进度</Text>
                <Progress percent={progress} status={status.status === 'failed' ? 'exception' : status.status === 'completed' ? 'success' : 'normal'} />
                <Text type="secondary">按已结束项统计（含完成、失败和取消）：{done} / {status.total}</Text>
              </Col>
              <Col xs={24} md={12}>
                <Text strong>{current ? '当前实验进度' : '最近实验进度'}{target ? `：${target.name}` : ''}</Text>
                <Progress percent={currentPercent} status={target?.status === 'failed' ? 'exception' : target?.status === 'completed' ? 'success' : current ? 'active' : 'normal'} />
                <Text type="secondary">{targetProgress?.total ? `ROUND ${targetProgress.round ?? 0} / ${targetProgress.total}` : '等待实验启动及轮次信息'}</Text>
              </Col>
            </Row>

            {status.error && <Alert title={getUserErrorMessage(status.error)} type="error" showIcon />}
            {status.pause_requested && <Alert title={status.status === 'paused' ? '队列已暂停，点击“继续队列”后启动后续实验。' : '已请求暂停队列：当前实验继续运行，结束后暂停。'} type="warning" showIcon />}
          </>
        )}

        {target?.run_id && <ExperimentMetrics key={target.run_id} name={target.name} runId={target.run_id}
          onProgress={value => { setMetricProgress(value); onProgress?.(value); }} />}
        <Card size="small" title="批次日志" styles={{ body: logsCollapsed ? { display: 'none' } : {} }}
          extra={<Button type="text" size="small" aria-label={logsCollapsed ? '展开批次日志' : '收起批次日志'}
            aria-expanded={!logsCollapsed} onClick={() => setLogsCollapsed(!logsCollapsed)}>{logsCollapsed ? '展开' : '收起'}</Button>}>
          <LogViewer logs={logs} height={300} />
        </Card>
      </Space>
    </Card>
  );
};

export default BatchMonitor;