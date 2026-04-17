// frontend/src/components/BatchMonitor/index.tsx
import React, { useState, useEffect, useRef } from 'react';
import { Card, Progress, Tag, Alert, Space, Typography } from 'antd';
import { SyncOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import AnsiToHtml from 'ansi-to-html';

const { Text } = Typography;
const ansiToHtml = new AnsiToHtml({ fg: '#d4d4d4', bg: '#1e1e1e', escapeXML: false });

interface BatchStatus {
  batch_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current_index: number;
  current_exp_name: string | null;
  total: number;
  start_time: string | null;
  end_time: string | null;
  error: string | null;
}

interface BatchMonitorProps {
  batchId: string;
  onComplete?: () => void;
}

const BatchMonitor: React.FC<BatchMonitorProps> = ({ batchId, onComplete }) => {
  const [status, setStatus] = useState<BatchStatus | null>(null);
  const [logs, setLogs] = useState('');
  const [ws, setWs] = useState<WebSocket | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!batchId) return;
    const socketRef = { current: null as WebSocket | null };
    
    const timer = setTimeout(() => {
      const socket = new WebSocket(`ws://127.0.0.1:8000/ws/batch/${batchId}`);
      socketRef.current = socket;
      setWs(socket);

      socket.onopen = () => {
        console.log('[BatchMonitor] WebSocket connected');
      };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received:', event.data)
        if (data.type === 'status') {
          setStatus(data.data);
          if (data.data.status === 'completed' || data.data.status === 'failed') {
            onComplete?.();
          }
        }
      } catch {
        // 普通日志行
        const cleanLine = event.data.replace(/[\x08\x0B\x0C]/g, '');
        setLogs(prev => prev + (prev ? '\n' : '') + cleanLine);
        if (logContainerRef.current) {
          setTimeout(() => {
            if (logContainerRef.current) {
              logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
            }
          }, 10);
        }
      }
    };

    socket.onerror = (event) => {
      console.error('[BatchMonitor] WebSocket error', event);
      // 额外打印 readyState
      console.log('[BatchMonitor] ReadyState:', socket.readyState);
    };

    socket.onclose = () => {
      console.log('[BatchMonitor] WebSocket closed');
    };
    }, 50);

    return () => {
    clearTimeout(timer);
    if (socketRef.current) {
      socketRef.current.close();
    }
  };
}, [batchId, onComplete]);

  const progress = status ? Math.round((status.current_index / status.total) * 100) : 0;

  const renderStatusTag = () => {
    if (!status) return null;
    switch (status.status) {
      case 'running':
        return <Tag icon={<SyncOutlined spin />} color="processing">运行中</Tag>;
      case 'completed':
        return <Tag icon={<CheckCircleOutlined />} color="success">已完成</Tag>;
      case 'failed':
        return <Tag icon={<CloseCircleOutlined />} color="error">失败</Tag>;
      default:
        return <Tag>等待中</Tag>;
    }
  };

  const htmlContent = ansiToHtml.toHtml(logs);

  return (
    <Card title="批量实验监控" style={{ marginTop: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        {status && (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Space>
                <Text strong>状态：</Text>
                {renderStatusTag()}
                <Text>进度：{status.current_index} / {status.total}</Text>
                {status.current_exp_name && <Text>当前实验：{status.current_exp_name}</Text>}
              </Space>
              <Text type="secondary">批次ID: {status.batch_id}</Text>
            </div>
            <Progress percent={progress} status={status.status === 'failed' ? 'exception' : 'active'} />
            {status.error && <Alert message={`错误: ${status.error}`} type="error" showIcon />}
          </>
        )}
        <div
          ref={logContainerRef}
          style={{
            background: '#1e1e1e',
            color: '#d4d4d4',
            padding: 12,
            borderRadius: 4,
            height: 300,
            overflowY: 'auto',
            fontFamily: 'monospace',
            fontSize: 12,
            marginTop: 16,
            textAlign: 'left',
            whiteSpace: 'pre-wrap',
          }}
        >
          {logs.length === 0 ? (
            <span style={{ color: '#888' }}>等待日志输出...</span>
          ) : (
            <div dangerouslySetInnerHTML={{ __html: htmlContent }} />
          )}
        </div>
      </Space>
    </Card>
  );
};

export default BatchMonitor;