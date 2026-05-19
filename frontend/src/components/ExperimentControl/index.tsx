import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button, Space, Input, message, Card, Tag, Alert, Progress } from 'antd';
import { PlayCircleOutlined, StopOutlined, SyncOutlined } from '@ant-design/icons';
import * as echarts from 'echarts';
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

  const [currentRound, setCurrentRound] = useState<number | null>(null);
  const [totalRound, setTotalRound] = useState<number | null>(null);

  const chartRef = useRef<HTMLDivElement>(null);          // 主图 ACC + ASR
  const chartInstance = useRef<echarts.ECharts | null>(null);

  const lossChartRef = useRef<HTMLDivElement>(null);      // 副图 Loss
  const lossChartInstance = useRef<echarts.ECharts | null>(null);

  const lastRoundRef = useRef<number>(-1);

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
      chartInstance.current?.dispose();
      chartInstance.current = null;
      lossChartInstance.current?.dispose();
      lossChartInstance.current = null;
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // ---------- 更新图表数据 ----------
  const updateChart = useCallback((round: number, accuracy: number, asr: number, loss: number) => {
    requestAnimationFrame(() => {
      // ★ 防重复
      if (round <= lastRoundRef.current) return;
      lastRoundRef.current = round;

      // ---- 主图 ACC + ASR ----
      if (chartInstance.current) {
        chartInstance.current.setOption({
          xAxis: { data: [...(chartInstance.current.getOption().xAxis as any)?.[0]?.data ?? [], round] },
          series: [
            { data: [...((chartInstance.current.getOption().series as any)?.[0]?.data ?? []), accuracy] },
            { data: [...((chartInstance.current.getOption().series as any)?.[1]?.data ?? []), asr] },
          ],
        });
      }

      // ---- 副图 Loss ----
      if (lossChartInstance.current) {
        lossChartInstance.current.setOption({
          xAxis: { data: [...(lossChartInstance.current.getOption().xAxis as any)?.[0]?.data ?? [], round] },
          series: [
            { data: [...((lossChartInstance.current.getOption().series as any)?.[0]?.data ?? []), loss] },
          ],
        });
      }
    });
  }, []);

  // ---------- WebSocket 连接 ----------
  const connectWS = useCallback((name: string) => {
    if (wsRef.current) wsRef.current.close();

    const socket = new WebSocket(`ws://127.0.0.1:8000/ws/logs/${name}`);
    wsRef.current = socket;

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === 'status') {
          const raw = String(msg.data.status ?? '').toLowerCase();
          if (raw === 'running' || raw === 'starting') setStatus('running');
          else if (raw === 'completed' || raw === 'finished') setStatus('finished');
          else if (raw === 'idle') setStatus('idle');
          return;
        }

        if (msg.type === 'csv_update' && msg.data) {
          const d = msg.data;
          if (d.round !== null && d.round !== undefined) {
            updateChart(Number(d.round), Number(d.accuracy), Number(d.asr), Number(d.loss));
          }
          return;
        }
      } catch {
        const cleanLine = event.data.replace(/[\x08\x0B\x0C]/g, '');
        const roundMatch = cleanLine.match(/ROUND\s+(\d+)\/(\d+)/i);
        if (roundMatch) {
          setCurrentRound(parseInt(roundMatch[1], 10));
          setTotalRound(parseInt(roundMatch[2], 10));
        }
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

    socket.onclose = () => { wsRef.current = null; };
  }, [updateChart]);

  // ---------- 启动 / 停止 ----------
  const handleStart = async () => {
    if (!expName.trim()) { message.error('请输入实验名称'); return; }
    if (onBeforeStart) {
      const result = await onBeforeStart();
      if (result.action === 'cancel') return;
    }
    setLoading(true);
    try {
      setLogs([]);
      setCurrentRound(null);
      setTotalRound(null);
      lastRoundRef.current = -1;

      // ★ 同时清空两张图
      requestAnimationFrame(() => {
        chartInstance.current?.setOption({
          xAxis: { data: [] },
          series: [
            { name: 'ACC', type: 'line', data: [] },
            { name: 'ASR', type: 'line', data: [] },
          ],
        });
        lossChartInstance.current?.setOption({
          xAxis: { data: [] },
          series: [
            { name: 'Loss', type: 'line', data: [] },
          ],
        });
      });

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
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, [connectWS]);

  return (
    <Card title="实验控制" style={{ marginTop: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space>
          <Input value={expName} onChange={e => setExpName(e.target.value)}
                 style={{ width: 200 }} disabled={status === 'running'} />
          <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleStart}
                  loading={loading && status === 'idle'} disabled={status === 'running'}>
            启动实验
          </Button>
          <Button danger icon={<StopOutlined />} onClick={handleStop}
                  loading={loading && status === 'running'} disabled={status !== 'running'}>
            停止实验
          </Button>
          <Tag color={status === 'running' ? 'processing' : status === 'finished' ? 'success' : 'default'}>
            {status === 'running' ? <><SyncOutlined spin /> 运行中</> : status === 'finished' ? '已完成' : '空闲'}
          </Tag>
        </Space>

        {status === 'running' && <Alert message="实验运行中" type="info" showIcon />}

        {currentRound !== null && totalRound !== null && (
          <div style={{ marginBottom: 10 }}>
            <Progress percent={Math.round((currentRound / totalRound) * 100)} status="active" />
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