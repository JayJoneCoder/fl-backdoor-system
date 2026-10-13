import React, { useState, useEffect, useRef } from 'react';
import { Card, Progress, Tag, Alert, Space, Typography } from 'antd';
import { SyncOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import LogViewer from '../LogViewer';
import { getWebSocketUrl } from '../../api/endpoints';
import { useBufferedLogs } from '../../hooks/useBufferedLogs';

const { Text } = Typography;

interface BatchStatus {
  batch_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current_index: number;
  current_exp_name: string | null;
  total: number;
  error: string | null;
}

const BatchMonitor: React.FC<{ batchId: string; onComplete?: () => void }> = ({ batchId, onComplete }) => {
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
    const socket = new WebSocket(getWebSocketUrl(`/ws/batch/${encodeURIComponent(batchId)}`));

    socket.onmessage = (event) => {
      if (!active) return;
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'status') {
          setStatus(data.data);
          if (!completed && (data.data.status === 'completed' || data.data.status === 'failed')) {
            completed = true;
            flushLogs();
            onCompleteRef.current?.();
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

  const progress = status ? Math.round((status.current_index / status.total) * 100) : 0;

  const renderTag = () => {
    if (!status) return null;
    if (status.status === 'running') return <Tag icon={<SyncOutlined spin />} color="processing">运行中</Tag>;
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
              <Text>进度：{status.current_index}/{status.total}</Text>
              {status.current_exp_name && <Text>当前实验：{status.current_exp_name}</Text>}
            </Space>

            <Progress percent={progress} status={status.status === 'failed' ? 'exception' : 'active'} />

            {status.error && <Alert message={status.error} type="error" />}
          </>
        )}

        <LogViewer logs={logs} height={300} />
      </Space>
    </Card>
  );
};

export default BatchMonitor;