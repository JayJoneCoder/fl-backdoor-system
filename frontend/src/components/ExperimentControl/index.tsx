import React, { useState, useEffect, useRef } from 'react';
import { Button, Space, Input, message, Card, Tag, Alert } from 'antd';
import { PlayCircleOutlined, StopOutlined, SyncOutlined } from '@ant-design/icons';
import { startExperiment, stopExperiment, getExperimentStatus } from '../../api/client';
import AnsiToHtml from 'ansi-to-html';

// 配置 ansi-to-html：默认白色文字，黑色背景
const ansiToHtml = new AnsiToHtml({
  fg: '#d4d4d4',
  bg: '#1e1e1e',
  escapeXML: false,
});

const ExperimentControl: React.FC = () => {
  const [expName, setExpName] = useState('test_exp');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<'idle' | 'running' | 'finished'>('idle');
  const [ansiLog, setAnsiLog] = useState('');
  const [ws, setWs] = useState<WebSocket | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  const connectWebSocket = (name: string): Promise<WebSocket> => {
    return new Promise((resolve, reject) => {
      if (ws) ws.close();
      const socket = new WebSocket(`ws://localhost:8000/ws/logs/${name}`);

      socket.onopen = () => {
        setWs(socket);
        resolve(socket);
      };

      socket.onerror = (err) => {
        reject(err);
      };

      socket.onmessage = (event) => {
        let rawLine = '';
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'status') {
            setStatus(data.data.status);
            return;
          } else if (data.type === 'csv_update') {
            rawLine = `[CSV] ${data.file}: ${JSON.stringify(data.data)}`;
          }
        } catch {
          rawLine = event.data;
        }

        // ✅ 只过滤退格等控制字符，保留 \x1B (ANSI 转义序列)
        const cleanLine = rawLine.replace(/[\x08\x0B\x0C]/g, '');
        setAnsiLog(prev => prev + (prev ? '\n' : '') + cleanLine);

        if (logContainerRef.current) {
          setTimeout(() => {
            if (logContainerRef.current) {
              logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
            }
          }, 10);
        }
      };

      socket.onclose = () => setWs(null);
    });
  };

  const handleStart = async () => {
    if (!expName.trim()) {
      message.error('请输入实验名称');
      return;
    }
    setLoading(true);
    try {
      setAnsiLog('');
      await connectWebSocket(expName);
      await startExperiment(expName);
      message.success(`实验 "${expName}" 已启动`);
      setStatus('running');
    } catch (error) {
      message.error('启动失败');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await stopExperiment();
      message.success('实验已停止');
      setStatus('idle');
      ws?.close();
    } catch (error) {
      message.error('停止失败');
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
          if (!ws || ws.readyState !== WebSocket.OPEN) {
            await connectWebSocket(res.data.exp_name);
          }
        }
      } catch (error) {
        // ignore
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, [ws]);

  const htmlContent = ansiToHtml.toHtml(ansiLog);

  return (
    <Card title="实验控制" style={{ marginTop: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space>
          <Input
            placeholder="实验名称"
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
          <Alert message="实验运行中，实时日志如下" type="info" showIcon style={{ marginTop: 16 }} />
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
          {ansiLog.length === 0 ? (
            <span style={{ color: '#888' }}>等待日志输出...</span>
          ) : (
            <div dangerouslySetInnerHTML={{ __html: htmlContent }} />
          )}
        </div>
      </Space>
    </Card>
  );
};

export default ExperimentControl;