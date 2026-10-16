import React, { useState, useEffect, useRef } from 'react';
import { Card, Upload, Button, Space, message, Table, Alert, Input, Row, Col } from 'antd';
import { UploadOutlined, PlayCircleOutlined, EditOutlined, EyeOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import { parseBatchConfig, runBatchExperiments, getBatchStatus, cancelBatch } from '../api/client';
import BatchMonitor from '../components/BatchMonitor';
import type { BatchStatus } from '../components/BatchMonitor';
import { getUserErrorMessage } from '../api/checkedFetch';

const { TextArea } = Input;
type BatchExperiment = { name: string } & Record<string, unknown>;

const BatchPage: React.FC = () => {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [experiments, setExperiments] = useState<BatchExperiment[]>([]);
  const [loading, setLoading] = useState(false);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [jsonContent, setJsonContent] = useState<string>('');
  const [editMode, setEditMode] = useState(false);
  const startedHereRef = useRef<string | null>(null);
  const didStartRef = useRef(false);

  useEffect(() => {
    let active = true;
    const checkActiveBatch = async () => {
      try {
        const res = await getBatchStatus();
        if (active && !didStartRef.current && ['pending', 'running', 'stopping'].includes(res.data.status) && res.data.batch_id) {
          // 有正在运行的批量任务，恢复状态
          setBatchId(res.data.batch_id);
          setBatchRunning(true);
          // 从后端返回的数据中恢复实验列表（如果有）
          if (res.data.experiments && res.data.experiments.length > 0) {
            setExperiments(res.data.experiments);
          }
        }
      } catch {
        // 忽略错误
      }
    };
    checkActiveBatch();
    return () => { active = false; };
  }, []);

  const handleUpload = async () => {
    if (fileList.length === 0) return;
    const file = fileList[0].originFileObj;
    if (!file) return;
    setLoading(true);
    try {
      // 读取文件内容用于编辑
      const text = await file.text();
      setJsonContent(text);
      const res = await parseBatchConfig(file);
      setExperiments(res.data.experiments);
      message.success(`成功解析 ${res.data.count} 个实验`);
      setEditMode(false);
    } catch (error) {
      message.error(getUserErrorMessage(error, '解析失败，请检查 JSON 格式。'));
    } finally {
      setLoading(false);
    }
  };

  const handleParseContent = async () => {
    if (!jsonContent.trim()) return;
    setLoading(true);
    try {
      // 将文本转为 Blob 再上传解析
      const blob = new Blob([jsonContent], { type: 'application/json' });
      const file = new File([blob], 'batch.json', { type: 'application/json' });
      const res = await parseBatchConfig(file);
      setExperiments(res.data.experiments);
      message.success(`成功解析 ${res.data.count} 个实验`);
      setEditMode(false);
    } catch (error) {
      message.error(getUserErrorMessage(error, '解析失败，请检查 JSON 格式。'));
    } finally {
      setLoading(false);
    }
  };

  const handleRun = async () => {
    if (experiments.length === 0) return;
    setLoading(true);
    try {
      const res = await runBatchExperiments(experiments);
      const data = res.data;
      didStartRef.current = true;
      startedHereRef.current = data.batch_id;
      setBatchId(data.batch_id);
      setBatchRunning(true);
      message.success(`批量实验已启动，批次ID: ${data.batch_id}`);
    } catch (error: unknown) {
      console.error('[Batch] Error:', error);
      message.error('启动失败：' + getUserErrorMessage(error));
    } finally {
      setLoading(false);
    }
  };

  const handleBatchComplete = (status: BatchStatus, notify: boolean) => {
    setBatchRunning(false);
    const startedHere = startedHereRef.current === status.batch_id;
    if (startedHere) startedHereRef.current = null;
    // A batch started here can finish before its first WebSocket snapshot.
    // Restoring a terminal snapshot must only synchronize state, not toast.
    if (!notify && !startedHere) return;
    if (status.status === 'completed') message.success('批量实验全部成功');
    else if (status.status === 'cancelled') message.info('批量实验已取消');
    else message.error(`${status.failed_count > 0 ? '批量实验有失败项' : '批量任务未能完成'}：${getUserErrorMessage(status.error, '请查看明细和日志。')}`);
  };

  const handleCancel = async () => {
    if (!batchId) return;
    setLoading(true);
    try {
      const response = await cancelBatch(batchId);
      if (['completed', 'failed', 'cancelled'].includes(response.data.status)) setBatchRunning(false);
      else message.info('正在停止并清理，完成前不会启动下一项');
    } catch (error) {
      message.error(`取消／清理失败：${getUserErrorMessage(error)} 新任务仍被阻止，可重试清理。`);
    } finally { setLoading(false); }
  };

  const columns = [
    { title: '实验名称', dataIndex: 'name', key: 'name' },
    {
      title: '配置',
      dataIndex: 'config',
      key: 'config',
      render: (_: unknown, record: BatchExperiment) => {
        const config = Object.fromEntries(Object.entries(record).filter(([key]) => key !== 'name'));
        return (
          <pre style={{ maxHeight: 100, overflow: 'auto', fontSize: 11 }}>
            {JSON.stringify(config, null, 2)}
          </pre>
        );
      },
    },
  ];

  return (
    <div>
      <h2 className="page-title">批量实验</h2>
      <Card>
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Alert
            message="上传 JSON 配置文件"
            description='文件格式示例：{ "experiments": [ { "name": "exp1", "attack": "badnets" } ], "common": { "dataset": "mnist" } }'
            type="info"
            showIcon
          />
          <Row gutter={16} align="middle">
            <Col span={12}>
              <Space>
                <Upload
                  fileList={fileList}
                  beforeUpload={() => false}
                  onChange={({ fileList }) => setFileList(fileList)}
                  maxCount={1}
                >
                  <Button icon={<UploadOutlined />}>选择 JSON 文件</Button>
                </Upload>
                <Button type="primary" onClick={handleUpload} loading={loading}>
                  解析文件
                </Button>
              </Space>
            </Col>
            <Col span={12} style={{ textAlign: 'right' }}>
              <Button
                icon={editMode ? <EyeOutlined /> : <EditOutlined />}
                onClick={() => setEditMode(!editMode)}
                disabled={!jsonContent}
              >
                {editMode ? '预览表格' : '编辑 JSON'}
              </Button>
            </Col>
          </Row>

          {jsonContent && (editMode ? (
            <Card size="small" title="编辑 JSON 配置">
              <TextArea
                value={jsonContent}
                onChange={(e) => setJsonContent(e.target.value)}
                rows={12}
                style={{ fontFamily: 'monospace' }}
              />
              <Space style={{ marginTop: 12 }}>
                <Button type="primary" onClick={handleParseContent} loading={loading}>
                  重新解析
                </Button>
                <Button onClick={() => setEditMode(false)}>取消</Button>
              </Space>
            </Card>
          ) : (
            experiments.length > 0 && (
              <>
                <h3>实验列表 ({experiments.length})</h3>
                <Table
                  dataSource={experiments}
                  columns={columns}
                  rowKey="name"
                  pagination={false}
                  size="small"
                />
              </>
            )
          ))}

          {experiments.length > 0 && !batchRunning && (
            <div style={{ textAlign: 'center', marginTop: 16 }}>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleRun}
                loading={loading}
                danger
                size="large"
              >
                运行批量实验
              </Button>
            </div>
          )}
          {batchRunning && <Button danger onClick={handleCancel} loading={loading}>取消整批／重试清理</Button>}
        </Space>
      </Card>

      {batchId && (
        <BatchMonitor batchId={batchId} onComplete={handleBatchComplete} />
      )}
    </div>
  );
};

export default BatchPage;