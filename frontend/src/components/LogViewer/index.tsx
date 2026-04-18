import React, { useRef, useEffect, useState } from 'react';
import * as echarts from 'echarts';
import { Progress } from 'antd';
import AnsiToHtml from 'ansi-to-html';

const ansiToHtml = new AnsiToHtml({
  fg: '#d4d4d4',
  bg: '#1e1e1e',
  escapeXML: false,
});

interface LogViewerProps {
  logs: string[];
  height?: number;
}

const LogViewer: React.FC<LogViewerProps> = ({ logs, height = 300 }) => {
  const logRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  // ✅ 必须在组件内部！
  const [currentRound, setCurrentRound] = useState<number | null>(null);
  const [totalRound, setTotalRound] = useState<number | null>(null);

  // =========================
  // 初始化图表
  // =========================
  useEffect(() => {
    if (chartRef.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      chartInstance.current?.dispose();
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // =========================
  // 更新图表 + round
  // =========================
  useEffect(() => {
    if (!chartInstance.current) return;

    const rounds: number[] = [];
    const acc: number[] = [];
    const asr: number[] = [];
    const loss: number[] = [];

    logs.forEach(line => {
      // ✅ 兼容 batch 和单实验（去掉 [] 限制）
      const roundMatch = line.match(/ROUND\s+(\d+)\/(\d+)/i);
      if (roundMatch) {
        setCurrentRound(parseInt(roundMatch[1], 10));
        setTotalRound(parseInt(roundMatch[2], 10));
      }

      if (!line.includes('[CSV]')) return;

      try {
        const jsonPart = line.substring(line.indexOf('{'));
        const data = JSON.parse(jsonPart);

        rounds.push(data.round ?? 0);
        acc.push(data.accuracy ?? 0);
        asr.push(data.asr ?? 0);
        loss.push(data.loss ?? 0);
      } catch {}
    });

    chartInstance.current.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['ACC', 'ASR', 'Loss'] },
      xAxis: { type: 'category', data: rounds },
      yAxis: { type: 'value' },
      series: [
        { name: 'ACC', type: 'line', data: acc },
        { name: 'ASR', type: 'line', data: asr },
        { name: 'Loss', type: 'line', data: loss },
      ],
    });
  }, [logs]);

  // =========================
  // 自动滚动日志
  // =========================
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const htmlContent = ansiToHtml.toHtml(logs.join('\n'));

  return (
    <div>
      {/* ✅ ROUND 进度条 */}
      {currentRound !== null && totalRound !== null && (
        <div style={{ marginBottom: 10 }}>
          <Progress
            percent={Math.round((currentRound / totalRound) * 100)}
            status="active"
          />
          <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>
            ROUND {currentRound} / {totalRound}
          </div>
        </div>
      )}

      {/* ✅ 图表 */}
      <div
        ref={chartRef}
        style={{
          width: '100%',
          height: 250,
          marginBottom: 10,
        }}
      />

      {/* ✅ 日志 */}
      <div
        ref={logRef}
        style={{
          background: '#1e1e1e',
          color: '#d4d4d4',
          padding: 12,
          borderRadius: 4,
          height,
          overflowY: 'auto',
          fontFamily: 'monospace',
          fontSize: 12,
          whiteSpace: 'pre-wrap',
          textAlign: 'left',
        }}
      >
        {logs.length === 0 ? (
          <span style={{ color: '#888' }}>等待日志输出...</span>
        ) : (
          <div dangerouslySetInnerHTML={{ __html: htmlContent }} />
        )}
      </div>
    </div>
  );
};

export default LogViewer;