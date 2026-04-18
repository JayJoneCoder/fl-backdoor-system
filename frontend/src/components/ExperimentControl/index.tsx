import React, { useState, useEffect, useRef } from 'react';
import { Button, Space, Input, message, Card, Tag, Alert } from 'antd';
import { PlayCircleOutlined, StopOutlined, SyncOutlined } from '@ant-design/icons';
import { startExperiment, stopExperiment, getExperimentStatus } from '../../api/client';
import LogViewer from '../../components/LogViewer';

const MAX_LOG_LINES = 1000;
const RENDER_INTERVAL = 200;

interface ExperimentControlProps {
  onBeforeStart?: () => Promise<{ action: 'cancel' | 'save' | 'backup' }>;
}

const ExperimentControl: React.FC<ExperimentControlProps> = ({ onBeforeStart }) => {
  const [expName, setExpName] = useState('test_exp');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<'idle' | 'running' | 'finished'>('idle');

  const [logs, setLogs] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const bufferRef = useRef<string[]>([]);
  const lastRenderRef = useRef(0);

  const connectWS = (name: string) => {
    if (wsRef.current) wsRef.current.close();

    const socket = new WebSocket(`ws://127.0.0.1:8000/ws/logs/${name}`);
    wsRef.current = socket;

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'status') {
          const raw = String(data.data.status ?? '').toLowerCase();
          if (raw === 'running' || raw === 'starting') setStatus('running');
          else if (raw === 'completed' || raw === 'finished') setStatus('finished');
          else if (raw === 'idle') setStatus('idle');
          return;
        }

        if (data.type === 'csv_update') {
          bufferRef.current.push(`[CSV] ${data.file}: ${JSON.stringify(data.data)}`);
        }
      } catch {
        const cleanLine = event.data.replace(/[\x08\x0B\x0C]/g, '');
        bufferRef.current.push(cleanLine);
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

    socket.onclose = () => {
      wsRef.current = null;
    };
  };

  const handleStart = async () => {
    if (!expName.trim()) {
      message.error('请输入实验名称');
      return;
    }

    if (onBeforeStart) {
      const result = await onBeforeStart();
      if (result.action === 'cancel') return;
    }

    setLoading(true);
    try {
      setLogs([]); // 清空日志
      connectWS(expName);

      await startExperiment(expName);
      message.success(`实验 "${expName}" 已启动`);
      setStatus('running');
    } catch {
      message.error('启动失败');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      const stopPromise = stopExperiment();
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('停止请求超时')), 15000)
      );
      await Promise.race([stopPromise, timeoutPromise]);

      message.success('实验已停止');
      setStatus('idle');
    } catch (e: any) {
      message.error(e?.message || '停止失败');
      setStatus('idle');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await getExperimentStatus();
        setStatus(res.data.status);

        if (res.data.status === 'running' && res.data.exp_name) {
          setExpName(res.data.exp_name);
          if (!wsRef.current) connectWS(res.data.exp_name);
        }
      } catch {}
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Card title="实验控制" style={{ marginTop: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space>
          <Input
            value={expName}
            onChange={(e) => setExpName(e.target.value)}
            style={{ width: 200 }}
            disabled={status === 'running'}
          />

          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={handleStart}
            loading={loading && status === 'idle'}
            disabled={status === 'running'}
          >
            启动实验
          </Button>

          <Button
            danger
            icon={<StopOutlined />}
            onClick={handleStop}
            loading={loading && status === 'running'}
            disabled={status !== 'running'}
          >
            停止实验
          </Button>

          <Tag color={status === 'running' ? 'processing' : status === 'finished' ? 'success' : 'default'}>
            {status === 'running' ? <><SyncOutlined spin /> 运行中</> : status === 'finished' ? '已完成' : '空闲'}
          </Tag>
        </Space>

        {status === 'running' && (
          <Alert message="实验运行中，实时日志如下" type="info" showIcon />
        )}

        <LogViewer logs={logs} height={300} />
      </Space>
    </Card>
  );
};

export default ExperimentControl;