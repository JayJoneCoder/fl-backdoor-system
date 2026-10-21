import React, { useState, useEffect, useRef } from 'react';
import { Card, Upload, Button, Space, message, Table, Alert, Input, Row, Col, Tag, Typography, Popconfirm, Select } from 'antd';
import type { TableRef } from 'antd/es/table';
import { UploadOutlined, PlayCircleOutlined, EditOutlined, EyeOutlined, UpOutlined, DownOutlined, StopOutlined, PauseCircleOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import { parseBatchConfig, runBatchExperiments, getBatchStatus, cancelBatch, pauseBatch, resumeBatch, stopCurrentBatchExperiment } from '../api/client';
import BatchMonitor from '../components/BatchMonitor';
import type { BatchStatus } from '../components/BatchMonitor';
import { getUserErrorMessage } from '../api/checkedFetch';
import { getBatchResultDisplay, isBatchActive, shouldAcceptBatchStatus } from '../utils/batchStream';
import type { BatchMetricProgress } from '../utils/batchStream';

const { TextArea } = Input;
type BatchExperiment = { name: string } & Record<string, unknown>;

const BatchPage: React.FC = () => {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [experiments, setExperiments] = useState<BatchExperiment[]>([]);
  const [loading, setLoading] = useState(false);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [jsonContent, setJsonContent] = useState<string>('');
  const [parsedJson, setParsedJson] = useState('');
  const [editMode, setEditMode] = useState(false);
  const [listCollapsed, setListCollapsed] = useState(false);
  const [batchStatus, setBatchStatus] = useState<BatchStatus | null>(null);
  const [metricProgress, setMetricProgress] = useState<BatchMetricProgress | null>(null);
  const [controlLoading, setControlLoading] = useState<string | null>(null);
  const startedHereRef = useRef<string | null>(null);
  const didStartRef = useRef(false);
  const confirmedRunRef = useRef<string | null>(null);
  const mountedRef = useRef(true);
  const generationRef = useRef(0);
  const statusRef = useRef<BatchStatus | null>(null);
  const batchIdRef = useRef<string | null>(null);
  const statusTickRef = useRef(0);
  const [restoreError, setRestoreError] = useState<string | null>(null);
  const [restoreAttempt, setRestoreAttempt] = useState(0);
  const [statusFilter, setStatusFilter] = useState('all');
  const tableRef = useRef<TableRef>(null);
  const jsonDirty = jsonContent !== parsedJson;

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  const receiveStatus = (value: BatchStatus) => {
    if (!mountedRef.current || value.batch_id !== batchIdRef.current || !shouldAcceptBatchStatus(statusRef.current, value)) return false;
    statusRef.current = value;
    statusTickRef.current += 1;
    setBatchStatus(value);
    setBatchRunning(isBatchActive(value.status));
    return true;
  };

  useEffect(() => {
    let active = true;
    const generation = generationRef.current;
    const checkActiveBatch = async () => {
      try {
        const res = await getBatchStatus();
        if (!active || generation !== generationRef.current || didStartRef.current) return;
        setRestoreError(null);
        if (isBatchActive(res.data.status) && res.data.batch_id) {
          // 有正在运行的批量任务，恢复状态
          setBatchId(res.data.batch_id);
          batchIdRef.current = res.data.batch_id;
          receiveStatus(res.data);
          // 从后端返回的数据中恢复实验列表（如果有）
          if (res.data.experiments && res.data.experiments.length > 0) {
            setExperiments(res.data.experiments);
            const restoredJson = JSON.stringify({ experiments: res.data.experiments }, null, 2);
            setJsonContent(restoredJson);
            setParsedJson(restoredJson);
            setEditMode(false);
          }
        }
      } catch (error) {
        if (active && generation === generationRef.current && !didStartRef.current) setRestoreError(getUserErrorMessage(error));
      }
    };
    checkActiveBatch();
    return () => { active = false; };
    // Each attempt is isolated from local uploads/starts and earlier attempts.
  }, [restoreAttempt]);

  const replaceExperiments = (items: BatchExperiment[]) => {
    setExperiments(items);
    setBatchId(null);
    batchIdRef.current = null;
    statusRef.current = null;
    setBatchStatus(null);
    setMetricProgress(null);
    setControlLoading(null);
    setRestoreError(null);
    setListCollapsed(false);
    setStatusFilter('all');
  };

  const handleUpload = async () => {
    if (batchRunning || loading || fileList.length === 0) return;
    const file = fileList[0].originFileObj;
    if (!file) return;
    setLoading(true);
    const generation = ++generationRef.current;
    try {
      // 读取文件内容用于编辑
      const text = await file.text();
      if (!mountedRef.current || generation !== generationRef.current) return;
      setJsonContent(text);
      setEditMode(true);
      const res = await parseBatchConfig(file);
      if (!mountedRef.current || generation !== generationRef.current) return;
      replaceExperiments(res.data.experiments);
      setParsedJson(text);
      message.success(`成功解析 ${res.data.count} 个实验`);
      setEditMode(false);
    } catch (error) {
      if (mountedRef.current && generation === generationRef.current) message.error(getUserErrorMessage(error, '解析失败，请检查 JSON 格式。'));
    } finally {
      if (mountedRef.current && generation === generationRef.current) setLoading(false);
    }
  };

  const handleParseContent = async () => {
    if (batchRunning || loading || !jsonContent.trim()) return;
    setLoading(true);
    const generation = ++generationRef.current;
    try {
      // 将文本转为 Blob 再上传解析
      const blob = new Blob([jsonContent], { type: 'application/json' });
      const file = new File([blob], 'batch.json', { type: 'application/json' });
      const res = await parseBatchConfig(file);
      if (!mountedRef.current || generation !== generationRef.current) return;
      replaceExperiments(res.data.experiments);
      setParsedJson(jsonContent);
      message.success(`成功解析 ${res.data.count} 个实验`);
      setEditMode(false);
    } catch (error) {
      if (mountedRef.current && generation === generationRef.current) message.error(getUserErrorMessage(error, '解析失败，请检查 JSON 格式。'));
    } finally {
      if (mountedRef.current && generation === generationRef.current) setLoading(false);
    }
  };

  const handleRun = async () => {
    if (batchRunning || loading || editMode || jsonDirty || experiments.length === 0) return;
    setLoading(true);
    const generation = ++generationRef.current;
    try {
      const res = await runBatchExperiments(experiments);
      if (!mountedRef.current || generation !== generationRef.current) return;
      const data = res.data;
      didStartRef.current = true;
      startedHereRef.current = data.batch_id;
      setBatchId(data.batch_id);
      batchIdRef.current = data.batch_id;
      statusRef.current = null;
      setBatchRunning(true);
      setBatchStatus(null);
      setMetricProgress(null);
      setControlLoading(null);
      setRestoreError(null);
      message.success(`批量实验已启动，批次ID: ${data.batch_id}`);
    } catch (error: unknown) {
      if (!mountedRef.current || generation !== generationRef.current) return;
      console.error('[Batch] Error:', error);
      message.error('启动失败：' + getUserErrorMessage(error));
    } finally {
      if (mountedRef.current && generation === generationRef.current) setLoading(false);
    }
  };

  const handleBatchComplete = (status: BatchStatus, notify: boolean) => {
    if (!receiveStatus(status)) return;
    const startedHere = startedHereRef.current === status.batch_id;
    if (startedHere) startedHereRef.current = null;
    // A batch started here can finish before its first WebSocket snapshot.
    // Restoring a terminal snapshot must only synchronize state, not toast.
    if (!notify && !startedHere) return;
    if (status.status === 'completed') message.success('批量实验全部成功');
    else if (status.status === 'cancelled') message.info(`批量任务已结束：完成 ${status.success_count ?? 0} 项，取消 ${status.cancelled_count ?? 0} 项`);
    else message.error(`${status.failed_count > 0 ? '批量实验有失败项' : '批量任务未能完成'}：${getUserErrorMessage(status.error, '请查看明细和日志。')}`);
  };

  const handleControl = async (action: 'cancel' | 'pause' | 'resume' | 'stop-current') => {
    if (!batchId || controlLoading || !isBatchActive(statusRef.current?.status ?? 'idle')) return;
    const requestedBatch = batchId;
    const tick = statusTickRef.current;
    const isRelevant = () => mountedRef.current && batchIdRef.current === requestedBatch;
    const currentRun = confirmedRunRef.current;
    if (action === 'stop-current' && !currentRun) return;
    setControlLoading(action);
    try {
      const response = action === 'cancel' ? await cancelBatch(batchId)
        : action === 'pause' ? await pauseBatch(batchId)
        : action === 'resume' ? await resumeBatch(batchId)
        : await stopCurrentBatchExperiment(batchId, currentRun!);
      if (!isRelevant()) return;
      // With legacy backends, a newer websocket snapshot wins over this HTTP
      // response. Versioned responses can be safely ordered across transports.
      if ((response.data.revision != null || tick === statusTickRef.current) && receiveStatus(response.data)) {
        if (!isBatchActive(response.data.status)) return;
      } else return;
      if (action === 'cancel') message.info('已请求结束整批；清理完成后释放任务。');
      else if (action === 'stop-current') message.info('已请求结束当前实验，后续队列保留。');
      else if (action === 'pause') message.info('已请求暂停队列，当前实验不受影响。');
      else message.success('队列已继续');
    } catch (error) {
      if (isRelevant()) message.error(`操作失败：${getUserErrorMessage(error)} 如清理失败，可通过“结束整批／重试清理”重试。`);
    } finally { if (isRelevant()) setControlLoading(null); }
  };

  const originalIndex = (record: BatchExperiment, fallback: number) => {
    const index = experiments.indexOf(record);
    return index >= 0 ? index : fallback;
  };
  const visibleExperiments = experiments.filter((_item, index) => statusFilter === 'all' || (batchStatus?.results?.[index]?.status ?? 'pending') === statusFilter);
  const columns = [
    { title: '序号', key: 'index', width: 70, render: (_: unknown, record: BatchExperiment, index: number) => originalIndex(record, index) + 1 },
    { title: '实验名称', dataIndex: 'name', key: 'name', width: 220, render: (name: string) => <Typography.Text strong style={{ wordBreak: 'break-all' }}>{name}</Typography.Text> },
    {
      title: '配置',
      dataIndex: 'config',
      key: 'config',
      render: (_: unknown, record: BatchExperiment) => {
        const config = Object.fromEntries(Object.entries(record).filter(([key]) => key !== 'name'));
        return (
          <pre style={{ maxHeight: 88, overflow: 'auto', fontSize: 12, margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', textAlign: 'left' }}>
            {JSON.stringify(config, null, 2)}
          </pre>
        );
      },
    },
    {
      title: '实验状态', key: 'status', width: 290,
      render: (_: unknown, record: BatchExperiment, index: number) => {
        const result = batchStatus?.results?.[originalIndex(record, index)];
        const percent = result?.run_id && metricProgress?.runId === result.run_id ? metricProgress.percent : null;
        const display = getBatchResultDisplay(result, percent);
        return <Space orientation="vertical" size={4}>
          <Tag color={display.color}>{display.label}</Tag>
          {display.detail && <Typography.Text type={result?.status === 'failed' ? 'danger' : 'secondary'}
            style={{ fontSize: 12, wordBreak: 'break-word' }} title={result?.error ?? undefined}>{display.detail}</Typography.Text>}
        </Space>;
      },
    },
  ];

  return (
    <div className="workspace-page">
      <div className="page-heading">
        <h2 className="page-title">批量实验</h2>
        <p className="page-description">导入实验配置，按队列执行任务并持续跟踪每项实验的进度。</p>
      </div>
      {restoreError && <Alert type="warning" showIcon title={`无法确认活动批次：${restoreError}`}
        action={<Button size="small" onClick={() => setRestoreAttempt(restoreAttempt + 1)}>重试检查</Button>} style={{ marginBottom: 16 }} />}
      <Card title="批量配置">
        <Space orientation="vertical" style={{ width: '100%' }} size="large">
          <Alert
            title="上传 JSON 配置文件"
            description='文件格式示例：{ "experiments": [ { "name": "exp1", "attack": "badnets" } ], "common": { "dataset": "mnist" } }'
            type="info"
            showIcon
          />
          <Row gutter={[16, 16]} align="middle">
            <Col xs={24} lg={16}>
              <Space wrap>
                <Upload
                  fileList={fileList}
                  beforeUpload={() => false}
                  onChange={({ fileList }) => setFileList(fileList)}
                  maxCount={1}
                  accept=".json,application/json"
                  disabled={batchRunning || loading}
                >
                  <Button icon={<UploadOutlined />}>选择 JSON 文件</Button>
                </Upload>
                <Button type="primary" onClick={handleUpload} loading={loading} disabled={batchRunning || fileList.length === 0}>
                  解析文件
                </Button>
              </Space>
            </Col>
            <Col xs={24} lg={8} className="batch-upload-actions">
              <Button
                icon={editMode ? <EyeOutlined /> : <EditOutlined />}
                onClick={async () => { if (editMode && jsonDirty) await handleParseContent(); else setEditMode(!editMode); }}
                disabled={!jsonContent || batchRunning || loading}
              >
                {editMode ? '预览表格' : '编辑 JSON'}
              </Button>
            </Col>
          </Row>

          {editMode && (
            <Card size="small" title="编辑 JSON 配置">
              <TextArea
                value={jsonContent}
                disabled={loading || batchRunning}
                onChange={(e) => setJsonContent(e.target.value)}
                rows={12}
                style={{ fontFamily: 'monospace' }}
              />
              <Space wrap style={{ marginTop: 12 }}>
                <Button type="primary" onClick={handleParseContent} loading={loading} disabled={batchRunning}>
                  重新解析
                </Button>
                <Button disabled={loading} onClick={() => { setJsonContent(parsedJson); setEditMode(false); }}>取消编辑</Button>
              </Space>
              {jsonDirty && <Alert type="warning" showIcon title="JSON 已修改，重新解析成功前不能运行；预览表格会先解析当前内容。" style={{ marginTop: 12 }} />}
            </Card>
          )}

          {experiments.length > 0 && !batchRunning && (
            <div style={{ marginTop: 16 }}>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleRun}
                loading={loading}
                disabled={editMode || jsonDirty}
                size="large"
              >
                运行批量实验
              </Button>
            </div>
          )}
          {batchRunning && (
            <Space orientation="vertical" style={{ width: '100%' }}>
              <Space wrap>
                <Button icon={batchStatus?.pause_requested ? <PlayCircleOutlined /> : <PauseCircleOutlined />}
                  disabled={!!controlLoading || !batchStatus || batchStatus.status === 'stopping'}
                  loading={controlLoading === 'pause' || controlLoading === 'resume'}
                  onClick={() => handleControl(batchStatus?.pause_requested ? 'resume' : 'pause')}>
                  {batchStatus?.pause_requested ? '继续队列' : '暂停队列'}
                </Button>
                <Popconfirm title="仅结束当前实验？" description="当前实验将标记为已取消，清理完成后继续后续队列；已暂停的队列仍保持暂停。"
                  onOpenChange={open => { if (open) confirmedRunRef.current = batchStatus?.results?.[batchStatus.current_index]?.run_id ?? null; }}
                  okText="结束当前" cancelText="返回" onConfirm={() => handleControl('stop-current')}>
                  <Button danger icon={<StopOutlined />} loading={controlLoading === 'stop-current'}
                    disabled={!!controlLoading || batchStatus?.status !== 'running' || !batchStatus?.current_exp_name || !batchStatus?.results?.[batchStatus.current_index]?.run_id}>仅结束当前实验</Button>
                </Popconfirm>
                <Popconfirm title="结束整批实验？" description="当前实验及所有等待项都会取消，不再启动后续实验。"
                  okText="结束整批" cancelText="返回" onConfirm={() => handleControl('cancel')}>
                  <Button danger loading={controlLoading === 'cancel'} disabled={!!controlLoading}>结束整批／重试清理</Button>
                </Popconfirm>
              </Space>
              <Typography.Text type="secondary">暂停仅作用于队列：当前实验会继续运行至结束，不会挂起训练进程。</Typography.Text>
            </Space>
          )}
        </Space>
      </Card>

      {experiments.length > 0 && (
        <Card title={`实验列表 (${experiments.length})`} style={{ marginTop: 16 }}
          styles={{ body: listCollapsed ? { display: 'none' } : {} }}
          extra={<Button type="text" size="small" aria-expanded={!listCollapsed}
            aria-label={`${listCollapsed ? '展开' : '收起'}实验列表`} icon={listCollapsed ? <DownOutlined /> : <UpOutlined />}
            onClick={() => setListCollapsed(!listCollapsed)}>{listCollapsed ? '展开' : '收起'}</Button>}>
          <Space wrap style={{ marginBottom: 12 }}>
            <Select aria-label="按实验状态筛选" value={statusFilter} onChange={setStatusFilter} style={{ width: 160 }} options={[
              { value: 'all', label: '全部状态' }, { value: 'pending', label: '等待中' }, { value: 'running', label: '正在进行' },
              { value: 'completed', label: '已完成' }, { value: 'failed', label: '失败' }, { value: 'cancelled', label: '已取消' }, { value: 'unknown', label: '未记录' },
            ]} />
            <Typography.Text type="secondary">显示 {visibleExperiments.length} / {experiments.length} 项</Typography.Text>
            <Button disabled={!batchStatus?.current_exp_name} onClick={() => {
              setStatusFilter('all'); setListCollapsed(false);
              requestAnimationFrame(() => tableRef.current?.scrollTo({ key: batchStatus?.current_exp_name ?? undefined }));
            }}>定位当前实验</Button>
          </Space>
          <Table ref={tableRef} dataSource={visibleExperiments} columns={columns} rowKey="name" pagination={false}
            scroll={{ x: 880, y: 480 }} size="small"
            onRow={(record, index) => ({ style: originalIndex(record, index ?? 0) === batchStatus?.current_index && !!batchStatus?.current_exp_name ? { background: '#e6f4ff' } : {} })} />
        </Card>
      )}
      {batchId && (
        <BatchMonitor key={batchId} batchId={batchId} onComplete={handleBatchComplete}
          onStatus={receiveStatus}
          onProgress={setMetricProgress} />
      )}
    </div>
  );
};

export default BatchPage;