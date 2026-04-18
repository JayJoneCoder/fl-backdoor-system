import React, { useState, useEffect, useRef } from 'react';
import { Card, Progress, Tag, Alert, Space, Typography } from 'antd';
import { SyncOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import LogViewer from '../LogViewer';

const { Text } = Typography;

const MAX_LOG_LINES = 1000;
const RENDER_INTERVAL = 200;

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
  const [logs, setLogs] = useState<string[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const bufferRef = useRef<string[]>([]);
  const lastRenderRef = useRef(0);

  useEffect(() => {
    const socket = new WebSocket(`ws://127.0.0.1:8000/ws/batch/${batchId}`);
    wsRef.current = socket;

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'status') {
          setStatus(data.data);
          if (data.data.status === 'completed' || data.data.status === 'failed') {
            onComplete?.();
          }
          return;
        }
      } catch {
        const clean = event.data.replace(/[\x08\x0B\x0C]/g, '');
        bufferRef.current.push(clean);
      }

      const now = Date.now();
      if (now - lastRenderRef.current > RENDER_INTERVAL) {
        lastRenderRef.current = now;

        setLogs(prev => {
          const merged = [...prev, ...bufferRef.current];
          bufferRef.current = [];
          return merged.length > MAX_LOG_LINES ? merged.slice(-MAX_LOG_LINES) : merged;
        });
      }
    };

    return () => socket.close();
  }, [batchId, onComplete]);

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