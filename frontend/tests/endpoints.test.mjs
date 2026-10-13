import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

// 使用项目已有的 TypeScript 编译器，不依赖额外测试框架或 Node 的 TS 加载器。
const source = readFileSync(new URL('../src/api/endpoints.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2023,
  },
});
const { API_BASE, getBackendUrl, getWebSocketUrl } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`
);

test('默认 API 地址使用同源路径', () => {
  assert.equal(API_BASE, '');
  assert.equal(getBackendUrl('/api/config'), '/api/config');
  assert.equal(getBackendUrl('results/test_exp/run.log'), '/results/test_exp/run.log');
});

test('后端地址与请求路径之间只保留一个斜杠', () => {
  assert.equal(
    getBackendUrl('//api/config', 'http://127.0.0.1:8000///'),
    'http://127.0.0.1:8000/api/config',
  );
  assert.equal(getBackendUrl('/api/config', '/backend/'), '/backend/api/config');
});

test('HTTP 页面上的 WebSocket 保持同一主机和端口', () => {
  assert.equal(
    getWebSocketUrl('/ws/logs/test_exp', '', 'http://127.0.0.1:5173/'),
    'ws://127.0.0.1:5173/ws/logs/test_exp',
  );
});

test('HTTPS 页面使用 WSS，避免混合内容', () => {
  assert.equal(
    getWebSocketUrl('/ws/batch/batch-1', '', 'https://fl.example.test/config'),
    'wss://fl.example.test/ws/batch/batch-1',
  );
});

test('独立部署的 WebSocket 使用指定后端地址及其协议', () => {
  assert.equal(
    getWebSocketUrl('/ws/logs/test_exp', 'https://api.example.test/', 'https://fl.example.test/'),
    'wss://api.example.test/ws/logs/test_exp',
  );
  assert.equal(
    getWebSocketUrl('/ws/batch/batch-1', 'http://127.0.0.1:8000', 'http://localhost:5173/'),
    'ws://127.0.0.1:8000/ws/batch/batch-1',
  );
});

test('相对后端前缀与 IPv6 页面地址均可用于 WebSocket', () => {
  assert.equal(
    getWebSocketUrl('/ws/logs/test_exp', '/backend/', 'https://fl.example.test/config'),
    'wss://fl.example.test/backend/ws/logs/test_exp',
  );
  assert.equal(
    getWebSocketUrl('/ws/logs/test_exp', '', 'http://[::1]:5173/'),
    'ws://[::1]:5173/ws/logs/test_exp',
  );
});

test('中文及特殊字符名称在结果与 WebSocket 路径中保持编码', () => {
  const name = encodeURIComponent('实验 #1');
  assert.equal(getBackendUrl(`/results/${name}/run.log`), `/results/${name}/run.log`);
  assert.equal(
    getWebSocketUrl(`/ws/logs/${name}`, '', 'http://127.0.0.1:5173/'),
    `ws://127.0.0.1:5173/ws/logs/${name}`,
  );
});

test('环境变量中的后端地址会去掉首尾空白', async () => {
  const configuredSource = `import.meta.env = { VITE_API_BASE_URL: '  https://api.example.test/  ' };\n${outputText}`;
  const configured = await import(
    `data:text/javascript;base64,${Buffer.from(configuredSource).toString('base64')}`
  );
  assert.equal(configured.API_BASE, 'https://api.example.test/');
  assert.equal(configured.getBackendUrl('/api/config'), 'https://api.example.test/api/config');
});