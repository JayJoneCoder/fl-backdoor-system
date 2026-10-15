import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/api/checkedFetch.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 },
}).outputText;
const { checkedFetch, HttpRequestError, getConfigErrorMessage } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`
);

test('成功响应保持原样，并传递请求参数', async t => {
  const response = new Response(JSON.stringify({ status: 'success' }), { status: 200 });
  let received;
  t.mock.method(globalThis, 'fetch', async (...args) => { received = args; return response; });
  const options = { method: 'POST', body: 'content' };
  assert.equal(await checkedFetch('/api/config/raw', options), response);
  assert.deepEqual(received, ['/api/config/raw', options]);
  assert.deepEqual(await response.json(), { status: 'success' });
});

test('HTTP 400 不会当成成功，返回后端校验信息', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ detail: 'batch-size: invalid' }), { status: 400 }));
  await assert.rejects(checkedFetch('/api/config/raw'), error => {
    assert.ok(error instanceof HttpRequestError);
    assert.equal(error.status, 400);
    assert.equal(error.message, 'batch-size: invalid');
    return true;
  });
});

test('HTTP 500 非 JSON 内容仍然失败', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('Internal server error', { status: 500 }));
  await assert.rejects(checkedFetch('/api/config/raw'), /HTTP 500/);
});

test('仅显式允许的 409 留给重名冲突处理', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('{}', { status: 409 }));
  assert.equal((await checkedFetch('/api/config/backups', {}, [409])).status, 409);
  await assert.rejects(checkedFetch('/api/config/backups'), error => error.status === 409);
});

test('网络异常不会被吞掉', async t => {
  const error = new Error('network unavailable');
  t.mock.method(globalThis, 'fetch', async () => { throw error; });
  await assert.rejects(checkedFetch('/api/config'), received => received === error);
});

test('配置错误提取优先显示后端 detail，支持 axios 与 fetch 异常', () => {
  assert.equal(getConfigErrorMessage({ response: { data: { detail: 'invalid config' } } }), 'invalid config');
  assert.equal(getConfigErrorMessage(new HttpRequestError(400, 'bad TOML')), 'bad TOML');
  assert.equal(getConfigErrorMessage(null), '配置操作失败');
});