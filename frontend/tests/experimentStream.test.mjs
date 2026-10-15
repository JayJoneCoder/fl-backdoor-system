import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const source = readFileSync(new URL('../src/utils/experimentStream.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 },
});
const stream = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const { takeLogBatch, mergeLogLines, parseRound, parseTotalRounds, parseMetricPoint, mergeMetricPoint, getProgressPercent, normalizeExperimentStatus } = stream;

test('日志批次从缓冲中分离，状态更新重复调用不会吞掉日志', () => {
  const buffer = { current: ['Starting local SuperLink', '[ROUND 1/10]'] };
  const batch = takeLogBatch(buffer);
  assert.deepEqual(buffer.current, []);
  const previous = ['history'];
  const updater = state => mergeLogLines(state, batch);
  assert.deepEqual(updater(previous), updater(previous));
  assert.deepEqual(updater(previous), ['history', ...batch]);
  assert.deepEqual(previous, ['history']);
  buffer.current.push('next round');
  assert.equal(batch.length, 2);
});

test('日志保留最新 1000 行，空批次不丢失末尾日志', () => {
  const previous = Array.from({ length: 1000 }, (_, index) => String(index));
  const result = mergeLogLines(previous, ['last line']);
  assert.equal(result.length, 1000);
  assert.equal(result[0], '1');
  assert.equal(result.at(-1), 'last line');
  assert.deepEqual(mergeLogLines(result, []), result);
});

test('CSV 初始轮次 0 有效，总轮次必须为正整数', () => {
  assert.equal(parseRound(0), 0);
  assert.equal(parseRound('10'), 10);
  assert.equal(parseTotalRounds('10'), 10);
  assert.equal(parseTotalRounds(0), null);
  for (const value of [null, undefined, '', ' ', 'invalid', -1, 1.5, Infinity, NaN, true, {}]) {
    assert.equal(parseRound(value), null);
    assert.equal(parseTotalRounds(value), null);
  }
});

test('进度只需 CSV 轮次和总轮数，不依赖 ROUND 文本日志', () => {
  assert.equal(getProgressPercent(parseRound(0), parseTotalRounds(10)), 0);
  assert.equal(getProgressPercent(parseRound(3), parseTotalRounds(10)), 30);
  assert.equal(getProgressPercent(parseRound(10), parseTotalRounds(10)), 100);
  assert.equal(getProgressPercent(12, 10), 100);
  assert.equal(getProgressPercent(-1, 10), 0);
  assert.equal(getProgressPercent(null, 10), null);
  assert.equal(getProgressPercent(1, null), null);
  assert.equal(getProgressPercent(1, 0), null);
});

test('starting 与完成状态映射到组件支持的状态', () => {
  for (const value of ['running', 'starting', 'RUNNING']) assert.equal(normalizeExperimentStatus(value), 'running');
  for (const value of ['finished', 'completed']) assert.equal(normalizeExperimentStatus(value), 'finished');
  for (const value of ['idle', undefined, null]) assert.equal(normalizeExperimentStatus(value), 'idle');
});

test('清理中、失败和取消不能映射为空闲或成功', () => {
  for (const value of ['stopping', 'cleaning', 'cleanup_failed', 'failed', 'cancelled']) {
    assert.equal(normalizeExperimentStatus(value), value);
  }
  for (const value of ['running', 'stopping', 'cleaning', 'cleanup_failed']) assert.equal(stream.isExperimentBusy(value), true);
  for (const value of ['idle', 'finished', 'failed', 'cancelled']) assert.equal(stream.isExperimentBusy(value), false);
});

test('完整历史与实时数据按轮次合并，重复回放不会产生重复点', () => {
  const point = round => ({ round, accuracy: round / 10, asr: 0.2, loss: 1.5 });
  const previous = [point(0), point(1), point(2)];
  let history = mergeMetricPoint(previous, point(4));
  for (const row of [point(0), point(3), point(1), point(2), point(4)]) {
    history = mergeMetricPoint(history, row);
  }
  assert.deepEqual(history.map(row => row.round), [0, 1, 2, 3, 4]);
  assert.deepEqual(previous.map(row => row.round), [0, 1, 2]);
  history = mergeMetricPoint(history, { ...point(2), accuracy: 0.9 });
  assert.equal(history.length, 5);
  assert.equal(history[2].accuracy, 0.9);
});

test('恢复数据允许有限数字字符串，拒绝空值和不完整指标', () => {
  assert.deepEqual(parseMetricPoint({ round: '0', accuracy: '0.5', asr: 0.2, loss: '1.2' }),
    { round: 0, accuracy: 0.5, asr: 0.2, loss: 1.2 });
  for (const value of [null, {}, { round: 1 }, { round: 1, accuracy: null, asr: 0.2, loss: 1 },
    { round: 1, accuracy: '', asr: 0.2, loss: 1 },
    { round: 1, accuracy: 0.5, asr: Infinity, loss: 1 },
    { round: 1.5, accuracy: 0.5, asr: 0.2, loss: 1 }]) {
    assert.equal(parseMetricPoint(value), null);
  }
});

// Minimal hook harness: no additional DOM/test dependencies. Re-evaluate every
// state updater against the same previous state, as a StrictMode regression.
const createHookHarness = () => {
  let state = [];
  let tick;
  let timerCleared = false;
  const effects = [];
  const exports = {};
  const react = {
    useState: () => [state, update => {
      if (typeof update === 'function') {
        const first = update(state);
        const second = update(state);
        assert.deepEqual(first, second);
        state = second;
      } else state = update;
    }],
    useRef: value => ({ current: value }),
    useCallback: callback => callback,
    useEffect: callback => effects.push(callback),
  };
  const hookSource = readFileSync(new URL('../src/hooks/useBufferedLogs.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(hookSource, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023 },
  }).outputText;
  runInNewContext(compiled, {
    exports,
    require: name => {
      if (name === 'react') return react;
      if (name === '../utils/experimentStream') return stream;
      throw new Error(`Unexpected import: ${name}`);
    },
    setInterval: (callback, interval) => {
      assert.equal(interval, 200);
      tick = callback;
      return 1;
    },
    clearInterval: id => { assert.equal(id, 1); timerCleared = true; },
  });
  const hook = exports.useBufferedLogs();
  const cleanup = effects[0]();
  return { hook, tick, cleanup, getState: () => [...state], timerCleared: () => timerCleared };
};

test('没有后续日志时，定时器也会刷新最后一条日志', () => {
  const harness = createHookHarness();
  harness.hook.queueLog('last log before exit');
  assert.deepEqual(harness.getState(), []);
  harness.tick();
  assert.deepEqual(harness.getState(), ['last log before exit']);
  harness.tick();
  assert.deepEqual(harness.getState(), ['last log before exit']);
  harness.cleanup();
  assert.equal(harness.timerCleared(), true);
});

test('新实验清除待刷新旧日志，断线时可显式刷新剩余日志', () => {
  const harness = createHookHarness();
  harness.hook.queueLog('old buffered line');
  harness.hook.resetLogs();
  harness.tick();
  assert.deepEqual(harness.getState(), []);
  harness.hook.queueLog('new run');
  harness.hook.flushLogs();
  assert.deepEqual(harness.getState(), ['new run']);
  harness.cleanup();
});

test('重连日志回放替换旧显示和待刷缓冲，不重复累加', () => {
  const harness = createHookHarness();
  harness.hook.queueLog('history line 1');
  harness.hook.flushLogs();
  harness.hook.queueLog('old pending line');
  harness.hook.resetLogs();
  harness.hook.queueLog('history line 1');
  harness.hook.queueLog('history line 2');
  harness.tick();
  assert.deepEqual(harness.getState(), ['history line 1', 'history line 2']);
  harness.cleanup();
});