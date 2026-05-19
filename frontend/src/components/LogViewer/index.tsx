import React, { useRef, useEffect } from 'react';
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

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const htmlContent = ansiToHtml.toHtml(logs.join('\n'));

  return (
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
  );
};

export default LogViewer;