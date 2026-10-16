import React, { useEffect, useRef, useState } from 'react';
import {
  Card,
  Row,
  Col,
  Spin,
  message,
  Button,
  Modal,
  Table,
  Space,
  Divider,
  Image,
  Statistic,
  Descriptions,
  Tooltip,
  Dropdown,
  Typography,
  Tag,
  Alert,
} from 'antd';
import type { MenuProps } from 'antd';
import {
  BarChartOutlined,
  DownloadOutlined,
  FileTextOutlined,
  ExperimentOutlined,
  AimOutlined,
  FileImageOutlined,
  FileZipOutlined,
  DeleteOutlined,
} from '@ant-design/icons';
import {
  listExperiments,
  getExperimentDetail,
  generatePlots,
  downloadExperimentImages,
  downloadExperimentAllFiles,
  deleteExperiment,
  downloadExperimentFile,
} from '../../api/client';
import { getBackendUrl } from '../../api/endpoints';
import { getUserErrorMessage } from '../../api/checkedFetch';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';

// 指标中文映射
const METRIC_LABELS: Record<string, string> = {
  accuracy: '准确率',
  asr: '攻击成功率',
  loss: '损失',
  precision: '精确率',
  recall: '召回率',
  fpr: '假正例率',
  auc: 'AUC',
  tp: '真阳性',
  fp: '假阳性',
  fn: '假阴性',
  tn: '真阴性',
  avg_malicious_removal_rate: '平均恶意移除率',
  avg_benign_removal_rate: '平均良性移除率',
  avg_kept_malicious: '平均保留恶意数',
  avg_kept_benign: '平均保留良性数',
  last_round: '最终轮次',
  dataset: '数据集',
};

interface ExperimentSummary {
  name: string;
  run_id?: string | null;
  accuracy: number | null;
  asr: number | null;
  loss: number | null;
  last_round: number | null;
  status?: string;
  error?: string | null;
  has_metrics?: boolean;
  can_plot?: boolean;
  warnings?: string[];
  precision?: number;
  recall?: number;
  fpr?: number;
  auc?: number;
  dataset?: string;
  created?: string;
  csv_files?: string[];
}

const RUN_STATUS: Record<string, string> = {
  starting: '启动中', running: '运行中', stopping: '停止中', cleaning: '清理中',
  cleanup_failed: '清理失败', finished: '已完成', failed: '失败', cancelled: '已取消', unknown: '未记录',
};

const ACTIVE_STATUSES = new Set(['starting', 'running', 'stopping', 'cleaning', 'cleanup_failed']);

const ExperimentHistory: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedExp, setSelectedExp] = useState<string | null>(null);
  const [detailData, setDetailData] = useState<any>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [plotGenerating, setPlotGenerating] = useState<string[]>([]);
  const generating = useRef(new Set<string>());
  const [deleting, setDeleting] = useState<string[]>([]);
  const removing = useRef(new Set<string>());
  const selected = useRef<string | null>(null);

  const loadExperiments = async () => {
    setLoading(true);
    try {
      const res = await listExperiments();
      setExperiments(res.data);
    } catch (error) {
      message.error(getUserErrorMessage(error, '加载实验列表失败，请稍后重试。'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadExperiments();
  }, []);

  const handleViewDetail = async (name: string) => {
    selected.current = name;
    setSelectedExp(name);
    setDetailData(null);
    setDetailModalVisible(true);
    setDetailLoading(true);
    try {
      const res = await getExperimentDetail(name);
      if (selected.current === name) setDetailData(res.data);
    } catch (error) {
      message.error(getUserErrorMessage(error, '加载实验详情失败，请稍后重试。'));
    } finally {
      if (selected.current === name) setDetailLoading(false);
    }
  };

  const handleGeneratePlots = async (expName: string) => {
    if (generating.current.has(expName) || removing.current.has(expName)) return false;
    generating.current.add(expName);
    setPlotGenerating([...generating.current]);
    const key = `history-plot-${expName}`;
    message.loading({ key, content: '正在生成图表，请稍候…', duration: 0 });
    try {
      await generatePlots(expName);
      message.success({ key, content: '图表生成成功', duration: 3 });
      if (selected.current === expName) {
        const res = await getExperimentDetail(expName);
        if (selected.current === expName) setDetailData(res.data);
      }
      return true;
    } catch (error) {
      message.error({ key, content: getUserErrorMessage(error, '图表生成失败，请查看后端日志。'), duration: 5 });
      return false;
    } finally {
      generating.current.delete(expName);
      setPlotGenerating([...generating.current]);
    }
  };

  const handleDelete = (exp: ExperimentSummary) => {
    Modal.confirm({
      title: `删除实验“${exp.name}”？`,
      content: '将永久删除该实验的指标、日志、图表、配置快照和模型，无法撤销。已有总结不会一并删除，请先下载需要保留的文件。',
      okText: '确认删除',
      okButtonProps: { danger: true },
      cancelText: '取消',
      onOk: async () => {
        if (removing.current.has(exp.name) || generating.current.has(exp.name)) return;
        removing.current.add(exp.name);
        setDeleting([...removing.current]);
        try {
          await deleteExperiment(exp.name, exp.run_id);
          setExperiments(current => current.filter(item => item.name !== exp.name));
          if (selected.current === exp.name) {
            selected.current = null;
            setSelectedExp(null);
            setDetailModalVisible(false);
            setDetailData(null);
          }
          message.success('实验已删除');
        } catch (error) {
          message.error(getUserErrorMessage(error, '删除实验失败，请稍后重试。'));
          throw error;
        } finally {
          removing.current.delete(exp.name);
          setDeleting([...removing.current]);
        }
      },
    });
  };

  const saveBlob = (data: BlobPart, filename: string, type?: string) => {
    const url = window.URL.createObjectURL(new Blob([data], { type }));
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  const handleDownloadFile = async (name: string, filename: string) => {
    try {
      const res = await downloadExperimentFile(name, filename);
      saveBlob(res.data, filename);
    } catch (error) {
      message.error(getUserErrorMessage(error, '下载文件失败，请稍后重试。'));
    }
  };

  const handleDownloadZip = async (
    fetchFn: () => Promise<any>,
    defaultFilename: string,
    onSuccess?: () => void
  ) => {
    try {
      const res = await fetchFn();
      saveBlob(res.data, defaultFilename, 'application/zip');
      onSuccess?.();
    } catch (error: any) {
      if (error.response?.status === 404) {
        message.error('文件不存在');
      } else {
        message.error(getUserErrorMessage(error, '下载失败，请稍后重试。'));
      }
    }
  };

  const handleDownloadImages = async (exp: ExperimentSummary) => {
    try {
      const res = await downloadExperimentImages(exp.name);
      await handleDownloadZip(() => Promise.resolve(res), `${exp.name}_images.zip`);
    } catch (error: any) {
      if (error.response?.status === 404) {
        if (exp.can_plot === false) {
          message.error('该实验暂无图表且指标不完整，无法生成；仍可下载全部文件或查看日志。');
          return;
        }
        Modal.confirm({
          title: '该实验暂无图表',
          content: '是否先生成图表？',
          okText: '生成并下载',
          cancelText: '取消',
          onOk: async () => {
            try {
              if (!await handleGeneratePlots(exp.name)) return;
              await handleDownloadZip(
                () => downloadExperimentImages(exp.name),
                `${exp.name}_images.zip`
              );
            } catch (err) {
              message.error(getUserErrorMessage(err, '图表生成失败，请查看后端日志。'));
            }
          },
        });
      } else {
        message.error(getUserErrorMessage(error, '下载失败，请稍后重试。'));
      }
    }
  };

  const getDownloadActions = (exp: ExperimentSummary) => [
    {
      key: 'main-csv',
      label: '主日志 CSV',
      icon: <FileTextOutlined />,
      disabled: exp.csv_files ? !exp.csv_files.includes(`${exp.name}.csv`) : false,
      onClick: () => handleDownloadFile(exp.name, `${exp.name}.csv`),
    },
    {
      key: 'clients-csv',
      label: '客户端日志 CSV',
      icon: <FileTextOutlined />,
      disabled: exp.csv_files ? !exp.csv_files.includes(`${exp.name}_clients.csv`) : false,
      onClick: () => handleDownloadFile(exp.name, `${exp.name}_clients.csv`),
    },
    {
      key: 'metrics-csv',
      label: '检测指标 CSV',
      icon: <FileTextOutlined />,
      disabled: exp.csv_files ? !exp.csv_files.includes(`${exp.name}_metrics.csv`) : false,
      onClick: () => handleDownloadFile(exp.name, `${exp.name}_metrics.csv`),
    },
    {
      key: 'run-log',
      label: '下载运行日志',
      icon: <FileTextOutlined />,
      onClick: () => handleDownloadFile(exp.name, 'run.log'),
    },
    {
      key: 'images-zip',
      label: '下载图表 (ZIP)',
      icon: <FileImageOutlined />,
      onClick: () => handleDownloadImages(exp),
    },
    {
      key: 'all-files-zip',
      label: '下载全部文件 (ZIP)',
      icon: <FileZipOutlined />,
      onClick: () => handleDownloadZip(
          () => downloadExperimentAllFiles(exp.name),
          `${exp.name}_all_files.zip`
        ),
    },
  ];
  const getDownloadMenuItems = (exp: ExperimentSummary): MenuProps['items'] =>
    getDownloadActions(exp).map(action => ({
      ...action,
      onClick: event => { event.domEvent.stopPropagation(); return action.onClick(); },
    }));
  const columns = [
    { title: '指标', dataIndex: 'metric', key: 'metric' },
    {
      title: '中文',
      dataIndex: 'metric',
      key: 'label',
      render: (metric: string) => METRIC_LABELS[metric] || metric,
    },
    { title: '值', dataIndex: 'value', key: 'value' },
  ];

  const metricTableData = detailData?.metrics
    ? Object.entries(detailData.metrics)
        .filter(([key]) => !key.startsWith('curve_data'))
        .map(([key, value]) => ({ metric: key, value: value == null ? '未记录' : String(value) }))
    : [];

  const downloadFile = (url: string, filename: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <>
      <Card title="历史实验" extra={<Button onClick={loadExperiments}>刷新</Button>}>
        <Spin spinning={loading}>
          <Row gutter={[16, 16]}>
            {experiments.map((exp) => (
              <Col key={exp.name} xs={24} sm={12} md={8} lg={6}>
                <Card
                  hoverable
                  style={{ cursor: 'pointer' }}
                  onClick={() => handleViewDetail(exp.name)}
                  actions={[
                    <Tooltip title={exp.can_plot === false ? '指标不完整，仍可查看详情和下载日志' : '生成图表'} key="plot">
                      <Button type="text" size="small" icon={<BarChartOutlined />}
                        loading={plotGenerating.includes(exp.name)}
                        disabled={exp.can_plot === false || deleting.includes(exp.name)}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (exp.can_plot !== false) return handleGeneratePlots(exp.name);
                        }}
                      >{plotGenerating.includes(exp.name) ? '生成中…' : '生成图表'}</Button>
                    </Tooltip>,
                    <Tooltip title="下载文件" key="download">
                      <Dropdown
                        menu={{ items: getDownloadMenuItems(exp) }}
                        trigger={['click']}
                      >
                        <Button type="text" size="small" icon={<DownloadOutlined />}
                          disabled={deleting.includes(exp.name)}
                          onClick={(e) => e.stopPropagation()}>下载文件</Button>
                      </Dropdown>
                    </Tooltip>,
                    <Tooltip title={ACTIVE_STATUSES.has(exp.status || '') ? '实验运行或清理结束后才可删除' : '永久删除该实验的全部文件'} key="delete">
                      <Button type="text" size="small" danger icon={<DeleteOutlined />}
                        loading={deleting.includes(exp.name)}
                        disabled={ACTIVE_STATUSES.has(exp.status || '') || plotGenerating.includes(exp.name)}
                        onClick={event => { event.stopPropagation(); handleDelete(exp); }}>删除</Button>
                    </Tooltip>,
                  ]}
                >
                  <div
                    style={{
                      fontSize: 16,
                      fontWeight: 500,
                      marginBottom: 8,
                      wordBreak: 'break-word',
                    }}
                  >
                    {exp.name}
                  </div>
                  <Tooltip title={exp.run_id || '旧实验未记录 UUID'}>
                    <Typography.Text
                      type="secondary"
                      ellipsis
                      style={{ display: 'block', fontSize: 11, fontFamily: 'monospace', marginBottom: 12 }}
                    >
                      UUID：{exp.run_id || '未记录'}
                    </Typography.Text>
                  </Tooltip>
                  {exp.status && <Tag color={exp.status === 'failed' || exp.status === 'cleanup_failed' ? 'error' : 'default'}>
                    {RUN_STATUS[exp.status] || '未记录'}
                  </Tag>}
                  {!!exp.warnings?.length && <Tooltip title={exp.warnings.join(' ')}><Tag color="warning">指标不完整</Tag></Tooltip>}
                  <Row gutter={8}>
                    <Col span={12}>
                      <Statistic
                        title="ACC"
                        value={
                          exp.accuracy != null
                            ? `${(exp.accuracy * 100).toFixed(2)}%`
                            : '未记录'
                        }
                        prefix={<ExperimentOutlined />}
                        valueStyle={{ fontSize: 20 }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="ASR"
                        value={
                          exp.asr != null
                            ? `${(exp.asr * 100).toFixed(2)}%`
                            : '未记录'
                        }
                        prefix={<AimOutlined />}
                        valueStyle={{ fontSize: 20 }}
                      />
                    </Col>
                  </Row>
                  <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
                    <div>Loss: {exp.loss?.toFixed(4) ?? '未记录'}</div>
                    <div>数据集: {exp.dataset || 'N/A'}</div>
                    <div>
                      AUC: {exp.auc?.toFixed(3) ?? '未记录'} | 轮次: {exp.last_round ?? '未记录'}
                    </div>
                    {exp.created && <div>创建: {exp.created.replace('T', ' ').substring(0, 22)}</div>}
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </Spin>
      </Card>

      <Modal
        title={`实验详情: ${selectedExp}`}
        open={detailModalVisible}
        onCancel={() => { selected.current = null; setDetailModalVisible(false); }}
        footer={null}
        width={1100}
      >
        <Spin spinning={detailLoading}>
          {detailData && (
            <>
              <div style={{ marginBottom: 20 }}>
                <Typography.Text type="secondary">实验 UUID：</Typography.Text>
                <Typography.Text
                  copyable={detailData.run_id ? { text: detailData.run_id, tooltips: ['复制 UUID', '已复制'] } : false}
                  style={{ wordBreak: 'break-all', color: '#000' }}
                >
                  {detailData.run_id ? (
                    <code style={{ color: '#000', background: '#f5f5f5' }}>
                      {detailData.run_id}
                    </code>
                  ) : '未记录 UUID（旧实验）'}
                </Typography.Text>
              </div>
              {detailData.status && <Tag>{RUN_STATUS[detailData.status] || '未记录'}</Tag>}
              {detailData.error && <Alert type="error" showIcon
                title={getUserErrorMessage(detailData.error, '实验未正常完成，请查看运行日志。')}
                style={{ marginBottom: 12 }} />}
              {detailData.warnings?.map((warning: string) => <Alert key={warning} type="warning" showIcon
                title={warning} style={{ marginBottom: 8 }} />)}
              <Row gutter={16} style={{ marginBottom: 24 }}>
                <Col span={12}>
                  <Statistic
                    title="准确率 (ACC)"
                    value={
                      detailData.metrics.accuracy != null
                        ? `${(detailData.metrics.accuracy * 100).toFixed(2)}%`
                        : '未记录'
                    }
                    prefix={<ExperimentOutlined />}
                    valueStyle={{ fontSize: 32 }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="攻击成功率 (ASR)"
                    value={
                      detailData.metrics.asr != null
                        ? `${(detailData.metrics.asr * 100).toFixed(2)}%`
                        : '未记录'
                    }
                    prefix={<AimOutlined />}
                    valueStyle={{ fontSize: 32 }}
                  />
                </Col>
              </Row>
              <Descriptions size="small" column={4} style={{ marginBottom: 16 }}>
                <Descriptions.Item label="Loss">
                  {detailData.metrics.loss?.toFixed(4) ?? '未记录'}
                </Descriptions.Item>
                <Descriptions.Item label="数据集">
                  {detailData.metrics.dataset || 'N/A'}
                </Descriptions.Item>
                <Descriptions.Item label="最终轮次">
                  {detailData.metrics.last_round ?? '未记录'}
                </Descriptions.Item>
                <Descriptions.Item label="AUC">
                  {detailData.metrics.auc?.toFixed(3) ?? '未记录'}
                </Descriptions.Item>
              </Descriptions>

              {!!detailData.metrics?.curve_data?.round?.length && (
                <div style={{ marginBottom: 24 }}>
                  <h4>ACC / ASR 曲线</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={detailData.metrics.curve_data.round.map(
                        (r: number, i: number) => ({
                          round: r,
                          accuracy: detailData.metrics.curve_data.accuracy[i],
                          asr: detailData.metrics.curve_data.asr[i],
                        })
                      )}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        stroke="#8884d8"
                        name="ACC"
                      />
                      <Line
                        type="monotone"
                        dataKey="asr"
                        stroke="#82ca9d"
                        name="ASR"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              <h4>指标汇总</h4>
              <Table
                columns={columns}
                dataSource={metricTableData}
                pagination={false}
                size="small"
                style={{ marginBottom: 16 }}
              />

              <Space wrap style={{ marginBottom: 16 }}>
                <Button
                  icon={<BarChartOutlined />}
                  onClick={() => handleGeneratePlots(selectedExp!)}
                  loading={selectedExp != null && plotGenerating.includes(selectedExp)}
                  disabled={detailData.can_plot === false}
                >
                  {selectedExp && plotGenerating.includes(selectedExp) ? '正在生成，请稍候…' : '生成图表'}
                </Button>
                {getDownloadActions({ ...detailData.metrics, ...detailData }).map(action => (
                  <Button key={action.key} icon={action.icon} disabled={action.disabled}
                    onClick={action.onClick}>{action.label}</Button>
                ))}
                <Button onClick={() => window.open(
                  getBackendUrl(`/results/${encodeURIComponent(selectedExp!)}/run.log`)
                )}>
                  查看运行日志
                </Button>
              </Space>

              {detailData.images && detailData.images.length > 0 && (
                <>
                  <Divider />
                  <h4>生成图表</h4>
                  <Image.PreviewGroup>
                    <Row gutter={[16, 16]}>
                      {detailData.images.map((img: string) => (
                        <Col span={12} key={img}>
                          <Card
                            size="small"
                            title={img}
                            extra={
                              <Button
                                size="small"
                                icon={<DownloadOutlined />}
                                onClick={() =>
                                  downloadFile(
                                    getBackendUrl(`/results/${encodeURIComponent(selectedExp!)}/${encodeURIComponent(img)}`),
                                    img
                                  )
                                }
                              />
                            }
                          >
                            <Image
                              src={getBackendUrl(`/results/${encodeURIComponent(selectedExp!)}/${encodeURIComponent(img)}`)}
                              alt={img}
                              style={{ width: '100%', cursor: 'pointer' }}
                            />
                          </Card>
                        </Col>
                      ))}
                    </Row>
                  </Image.PreviewGroup>
                </>
              )}
            </>
          )}
        </Spin>
      </Modal>
    </>
  );
};

export default ExperimentHistory;