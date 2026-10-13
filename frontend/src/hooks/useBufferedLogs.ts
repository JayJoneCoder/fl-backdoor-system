import { useCallback, useEffect, useRef, useState } from 'react';
import { mergeLogLines, takeLogBatch } from '../utils/experimentStream';

export const useBufferedLogs = () => {
  const [logs, setLogs] = useState<string[]>([]);
  const bufferRef = useRef<string[]>([]);

  const queueLog = useCallback((line: string) => {
    bufferRef.current.push(line);
  }, []);

  const flushLogs = useCallback(() => {
    const batch = takeLogBatch(bufferRef);
    if (batch.length > 0) setLogs(previous => mergeLogLines(previous, batch));
  }, []);

  const resetLogs = useCallback(() => {
    bufferRef.current = [];
    setLogs([]);
  }, []);

  useEffect(() => {
    const timer = setInterval(flushLogs, 200);
    return () => clearInterval(timer);
  }, [flushLogs]);

  return { logs, queueLog, flushLogs, resetLogs };
};