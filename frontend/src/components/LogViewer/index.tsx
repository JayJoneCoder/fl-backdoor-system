import React, { useRef, useEffect } from 'react';
import AnsiToHtml from 'ansi-to-html';

const ansiToHtml = new AnsiToHtml({
  fg: '#c8d3e0',
  bg: '#152132',
  escapeXML: false,
});

interface LogViewerProps {
  logs: string[];
  height?: number;
}

const LogViewer: React.FC<LogViewerProps> = ({ logs, height = 300 }) => {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const htmlContent = ansiToHtml.toHtml(logs.join('\n'));

  return (
    <div
      ref={logRef}
      className="experiment-log"
      role="region"
      aria-label="运行日志"
      tabIndex={0}
      style={{ height }}
    >
      {logs.length === 0 ? (
        <span style={{ color: '#8b9db4' }}>等待日志输出...</span>
      ) : (
        <div dangerouslySetInnerHTML={{ __html: htmlContent }} />
      )}
    </div>
  );
};

export default LogViewer;