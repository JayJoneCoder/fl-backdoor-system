import React, { useState, useEffect, useRef } from 'react';
import { Card, Progress, Tag, Alert, Space, Typography } from 'antd';
import { SyncOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import LogViewer from '../LogViewer';
import { getWebSocketUrl } from '../../api/endpoints';
import { useBufferedLogs } from '../../hooks/useBufferedLogs';
import { getUserErrorMessage } from '../../api/checkedFetch';

const { Text } = Typography;
const RESULT_LABELS: Record<string, string> = {
  pending: '等待中', running: '运行中', completed: '成功', failed: '失败', cancelled: '已取消', unknown: '未记录',
};

export interface BatchStatus {
  batch_id: string;
  status: 'pending' | 'running' | 'stopping' | 'completed' | 'failed' | 'cancelled';
  current_index: number;
  current_exp_name: string | null;
  total: number;
  error: string | null;
  completed_count: number;
  success_count: number;
  failed_count: number;
  cancelled_count: number;
  unknown_count?: number;
  results: { name: string; status: string; exit_code: number | null; error: string | null }[];
}

const BatchMonitor: React.FC<{ batchId: string; onComplete?: (status: BatchStatus, notify: boolean) => void }> = ({ batchId, onComplete }) => {
  const [status, setStatus] = useState<BatchStatus | null>(null);
  const { logs, queueLog, flushLogs, resetLogs } = useBufferedLogs();

  const onCompleteRef = useRef(onComplete);

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  useEffect(() => {
    resetLogs();
    let active = true;
    let completed = false;
    let observedActive = false;
    const socket = new WebSocket(getWebSocketUrl(`/ws/batch/${encodeURIComponent(batchId)}`));

    socket.onmessage = (event) => {
      if (!active) return;
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'status') {
          setStatus(data.data);
          if (['pending', 'running', 'stopping'].includes(data.data.status)) observedActive = true;
          if (!completed && ['completed', 'failed', 'cancelled'].includes(data.data.status)) {
            completed = true;
            flushLogs();
            // Always synchronize the parent's busy state, but don't announce a
            // historical terminal snapshot as a newly completed experiment.
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

    socket.onclose = () => { if (active) flushLogs(); };
    return () => {
      active = false;
      socket.close();
    };
  }, [batchId, queueLog, flushLogs, resetLogs]);

  const done = status?.completed_count ?? 0;
  const progress = status && status.total > 0 ? Math.round(done / status.total * 100) : 0;

  const renderTag = () => {
    if (!status) return null;
    if (status.status === 'running') return <Tag icon={<SyncOutlined spin />} color="processing">运行中</Tag>;
    if (status.status === 'stopping') return <Tag icon={<SyncOutlined spin />} color="warning">停止／清理中</Tag>;
    if (status.status === 'cancelled') return <Tag color="default">已取消</Tag>;
    if (status.status === 'completed') return <Tag icon={<CheckCircleOutlined />} color="success">已完成</Tag>;
    if (status.status === 'failed') return <Tag icon={<CloseCircleOutlined />} color="error">失败</Tag>;
    return <Tag>等待中</Tag>;
  };

  return (
    <Card title="批量实验监控" style={{ marginTop: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        {status && (
          <>
            <Space>
              <Text strong>状态：</Text>
              {renderTag()}
              <Text>已结束：{done}/{status.total}</Text>
              <Text>成功 {status.success_count ?? 0} / 失败 {status.failed_count ?? 0} / 取消 {status.cancelled_count ?? 0}</Text>
              {!!status.unknown_count && <Text type="secondary">状态未记录 {status.unknown_count}</Text>}
              {status.current_exp_name && <Text>当前实验：{status.current_exp_name}</Text>}
            </Space>

            <Progress percent={progress} status={status.status === 'failed' ? 'exception' : status.status === 'completed' ? 'success' : 'normal'} />

            {status.error && <Alert message={getUserErrorMessage(status.error)} type="error" />}
            {status.results?.map((item, index) => (
              <Text key={`${index}-${item.name}`}>{item.name}：{RESULT_LABELS[item.status] ?? '未知状态'}
                {item.exit_code !== null && `（退出码 ${item.exit_code}）`}{item.error && ` — ${getUserErrorMessage(item.error)}`}</Text>
            ))}
          </>
        )}

        <LogViewer logs={logs} height={300} />
      </Space>
    </Card>
  );
};

export default BatchMonitor;