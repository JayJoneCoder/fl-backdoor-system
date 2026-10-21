import { useEffect, useState } from 'react';
import { Alert, Button, Card, Col, InputNumber, Row, Space, Spin, Tooltip, Typography } from 'antd';
import { DownOutlined, QuestionCircleOutlined, UpOutlined } from '@ant-design/icons';
import { getBackendUrl } from '../../api/endpoints';
import { checkedFetch, getUserErrorMessage, HttpRequestError } from '../../api/checkedFetch';

interface PreviewResult {
  dataset: string;
  index: number;
  total: number;
  label_text: string;
  target_label_text: string;
  clean_image: string;
  poisoned_image: string;
  changed_pixels: number;
  local_image?: string;
  global_image?: string;
  local_changed_pixels?: number;
  global_changed_pixels?: number;
  local_trigger_index?: number;
  local_trigger_count?: number;
  local_sub_blocks?: number;
  notice: string;
}

interface Props {
  config: Record<string, unknown>;
  ready: boolean;
}

export default function AttackPreview({ config, ready }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [selection, setSelection] = useState({ dataset: '', index: 0 });
  const [clientId, setClientId] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const [result, setResult] = useState<PreviewResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const dataset = String(config.dataset ?? 'cifar10');
  const attack = String(config.attack ?? 'none');
  const distributed = attack === 'dba' || attack === 'fcba';
  const index = selection.dataset === dataset ? selection.index : 0;
  const clients = Math.max(1, Number(config['num-clients']) || 1);
  const selectedClient = Math.min(clientId, clients - 1);
  // Do not request another image for unrelated edits (run name, defenses, etc.).
  const draft = Object.fromEntries(Object.entries(config).filter(([key]) =>
    ['dataset', 'attack', 'seed', 'num-clients', 'trigger-size', 'target-label', 'wanet-noise', 'wanet-grid-size'].includes(key)
    || /^(frequency-|dba-|fcba-)/.test(key)));
  const payload = JSON.stringify({ config: draft, index, scope: distributed ? 'local' : 'global', client_id: selectedClient });

  useEffect(() => {
    if (!ready || collapsed) return;
    const controller = new AbortController();
    let active = true;
    setLoading(true);
    setError('');
    setResult(null);
    const timer = window.setTimeout(async () => {
      try {
        const response = await checkedFetch(getBackendUrl('/api/attack/preview'), {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: payload, signal: controller.signal,
        });
        const data: PreviewResult = await response.json();
        if (distributed && (!data.local_image || !data.global_image)) {
          throw new Error('当前后端尚未支持分布式攻击三图对比，请在没有实验运行时重启后端，再点击“刷新预览”。');
        }
        if (active) setResult(data);
      } catch (cause) {
        if (active) {
          // A missing route means an old backend (or wrong API target), not a
          // missing dataset. Preserve the latter's explicit 404 explanation.
          const missingEndpoint = cause instanceof HttpRequestError && cause.status === 404
            && /^(Not Found|请求失败（HTTP 404）)$/i.test(cause.message.trim());
          setError(missingEndpoint
            ? '当前后端未提供后门样本预览接口，请在没有实验运行时重启后端，再点击“刷新预览”；若仍失败，请检查后端地址配置。'
            : getUserErrorMessage(cause, '加载图片预览失败，请检查数据集和对抗性安全测试参数后重试。'));
        }
      } finally {
        if (active) setLoading(false);
      }
    }, 350);
    return () => { active = false; window.clearTimeout(timer); controller.abort(); };
  }, [payload, distributed, ready, collapsed, refresh]);

  const choose = (next: number) => setSelection({ dataset, index: Math.max(0, next) });
  const pictureStyle = { width: 224, height: 224, maxWidth: '100%', objectFit: 'contain' as const,
    imageRendering: 'pixelated' as const, borderRadius: 6, border: '1px solid #d9d9d9', background: '#fafafa' };

  return (
    <Card title="后门样本预览" style={{ marginBottom: 16 }}
      styles={{ body: collapsed ? { display: 'none' } : {} }}
      extra={<Space>
        <Tooltip title="使用页面当前参数，对所选训练图片强制添加触发器；不保存配置，不代表实际投毒抽样结果。">
          <QuestionCircleOutlined />
        </Tooltip>
        <Button type="text" size="small" htmlType="button" aria-expanded={!collapsed}
          aria-label={`${collapsed ? '展开' : '收起'}后门样本预览`}
          icon={collapsed ? <DownOutlined /> : <UpOutlined />} onClick={() => setCollapsed(value => !value)}>
          {collapsed ? '展开' : '收起'}
        </Button>
      </Space>}>
      <Space wrap style={{ marginBottom: 16 }}>
        <Typography.Text>训练集图片编号</Typography.Text>
        <InputNumber aria-label="训练集图片编号" min={1} max={result?.total} precision={0}
          value={index + 1} onChange={value => { if (value !== null) choose(value - 1); }} />
        <Button htmlType="button" disabled={index === 0} onClick={() => choose(index - 1)}>上一张</Button>
        <Button htmlType="button" disabled={!result || index + 1 >= result.total} onClick={() => choose(index + 1)}>下一张</Button>
        <Button htmlType="button" onClick={() => setRefresh(value => value + 1)} loading={loading}>刷新预览</Button>
        {result && <Typography.Text type="secondary">共 {result.total.toLocaleString()} 张</Typography.Text>}
        {distributed && <>
          <Typography.Text>预览客户端 ID</Typography.Text>
          <InputNumber aria-label="预览客户端 ID" min={0} max={clients - 1} precision={0} value={selectedClient}
            onChange={value => { if (value !== null) setClientId(value); }} />
        </>}
      </Space>
      <Alert type="info" showIcon style={{ marginBottom: 16 }}
        title="仅展示植入效果，不代表该图片在实际训练中被选中投毒。"
        description={distributed
          ? '同时对比原图、客户端训练投毒效果和完整触发器（ASR 评估用）。中间图复用训练时的局部触发器，FCBA 可能包含多个子块；假设所选客户端为恶意客户端且该图已被选中投毒，忽略投毒比例和目标类别排除规则，不代表实际投毒抽样结果。不会修改数据集或启动实验。'
          : '默认展示所选数据集训练集的第一张图片，可按编号选择。使用当前未保存的对抗性安全测试参数，忽略投毒抽样比例和目标类别排除规则；不会修改数据集或启动实验。'} />
      {error && <Alert type="warning" showIcon title={error} style={{ marginBottom: 16 }} />}
      <Spin spinning={loading}>
        <Row gutter={[24, 16]} style={{ minHeight: 250 }}>
          <Col xs={24} sm={distributed ? 24 : 12} md={distributed ? 8 : 12} style={{ textAlign: 'center' }}>
            <Typography.Title level={5}>原始图片</Typography.Title>
            {result && <><img src={result.clean_image} alt="未植入后门的原始图片" style={pictureStyle} />
              <div>原始标签：{result.label_text}</div></>}
          </Col>
          <Col xs={24} sm={distributed ? 24 : 12} md={distributed ? 8 : 12} style={{ textAlign: 'center' }}>
            <Typography.Title level={5}>{distributed ? '客户端训练投毒效果（局部触发器）'
              : attack === 'none' || attack === 'identity' ? '未启用攻击' : '植入后门效果'}</Typography.Title>
            {result && <><img src={distributed ? result.local_image : result.poisoned_image}
              alt={distributed ? '客户端训练时局部触发器的强制植入示意' : '强制植入后门后的效果示意'} style={pictureStyle} />
              <div>{attack === 'none' || attack === 'identity' ? '图像保持不变' : `投毒目标标签：${result.target_label_text}`}</div>
              {distributed && <div>客户端 {selectedClient} · 局部触发器 {(result.local_trigger_index ?? 0) + 1}/{result.local_trigger_count}
                {result.local_sub_blocks !== undefined && ` · ${result.local_sub_blocks} 个子块`}</div>}
              <Typography.Text type="secondary">变化像素：{distributed ? result.local_changed_pixels : result.changed_pixels}</Typography.Text></>}
          </Col>
          {distributed && <Col xs={24} md={8} style={{ textAlign: 'center' }}>
            <Typography.Title level={5}>完整触发器效果（ASR 评估用）</Typography.Title>
            {result && <><img src={result.global_image} alt="完整触发器的强制植入示意（ASR 评估用）" style={pictureStyle} />
              <div>投毒目标标签：{result.target_label_text}</div>
              <div><Typography.Text type="secondary">不是单客户端通常使用的局部触发器</Typography.Text></div>
              <Typography.Text type="secondary">变化像素：{result.global_changed_pixels}</Typography.Text></>}
          </Col>}
        </Row>
      </Spin>
    </Card>
  );
}