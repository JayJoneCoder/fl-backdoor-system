import React, { useState } from 'react';
import { Card, Upload, Button, Space, message, Table, Progress, Alert, Input, Row, Col } from 'antd';
import { UploadOutlined, PlayCircleOutlined, EditOutlined, EyeOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import { parseBatchConfig, runBatchExperiments } from '../api/client';
import BatchMonitor from '../components/BatchMonitor';

const { TextArea } = Input;

const BatchPage: React.FC = () => {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [jsonContent, setJsonContent] = useState<string>('');
  const [editMode, setEditMode] = useState(false);

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
      message.error('解析失败，请检查 JSON 格式');
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
      message.error('解析失败，请检查 JSON 格式');
    } finally {
      setLoading(false);
    }
  };

  const handleRun = async () => {
    if (experiments.length === 0) return;
    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/batch/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiments }),
      });
      const data = await res.json();
      console.log('[Fetch] Response:', data);
      setBatchId(data.batch_id);
      setBatchRunning(true);
      message.success(`批量实验已启动，批次ID: ${data.batch_id}`);
    } catch (error: any) {
      console.error('[Fetch] Error:', error);
      message.error('启动失败: ' + (error?.message || '未知错误'));
    } finally {
      setLoading(false);
    }
  };

  const handleBatchComplete = () => {
    setBatchRunning(false);
    message.success('批量实验已完成');
  };

  const columns = [
    { title: '实验名称', dataIndex: 'name', key: 'name' },
    {
      title: '配置',
      dataIndex: 'config',
      key: 'config',
      render: (_: any, record: any) => {
        const { name, ...config } = record;
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
        </Space>
      </Card>

      {batchRunning && batchId && (
        <BatchMonitor batchId={batchId} onComplete={handleBatchComplete} />
      )}
    </div>
  );
};

export default BatchPage;