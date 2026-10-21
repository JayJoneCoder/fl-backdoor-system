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
  Input,
  Pagination,
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
  owner?: { uid: string; username: string } | null;
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
  files_available?: boolean;
  available_files?: string[];
  attachments_missing?: boolean;
}

const RUN_STATUS: Record<string, string> = {
  starting: '启动中', running: '运行中', stopping: '停止中', cleaning: '清理中',
  cleanup_failed: '清理失败', finished: '已完成', failed: '失败', cancelled: '已取消', unknown: '未记录',
};

const ACTIVE_STATUSES = new Set(['starting', 'running', 'stopping', 'cleaning', 'cleanup_failed']);

const ExperimentHistory: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchInput, setSearchInput] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(100);
  const [total, setTotal] = useState(0);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedExp, setSelectedExp] = useState<string | null>(null);
  const [detailData, setDetailData] = useState<any>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [plotGenerating, setPlotGenerating] = useState<string[]>([]);
  const generating = useRef(new Set<string>());
  const [deleting, setDeleting] = useState<string[]>([]);
  const removing = useRef(new Set<string>());
  const selected = useRef<string | null>(null);
  const loadSequence = useRef(0);

  const loadExperiments = async (query = searchTerm, nextPage = page, nextSize = pageSize) => {
    const sequence = ++loadSequence.current;
    setLoading(true);
    try {
      const res = await listExperiments({
        search: query,
        limit: nextSize,
        offset: (nextPage - 1) * nextSize,
        paginated: true,
      });
      if (sequence !== loadSequence.current) return;
      const payload = res.data;
      if (Array.isArray(payload)) {
        // 兼容旧版后端：未分页时直接在本地展示全部记录。
        setExperiments(payload);
        setTotal(payload.length);
      } else {
        setExperiments(Array.isArray(payload?.items) ? payload.items : []);
        setTotal(Number(payload?.total) || 0);
      }
    } catch (error) {
      if (sequence !== loadSequence.current) return;
      message.error(getUserErrorMessage(error, '加载实验列表失败，请稍后重试。'));
    } finally {
      if (sequence === loadSequence.current) setLoading(false);
    }
  };

  const applySearch = (value: string) => {
    const next = value.trim();
    setSearchTerm(next);
    setPage(1);
    loadExperiments(next, 1, pageSize);
  };

  const handlePageChange = (nextPage: number, nextSize: number) => {
    const targetPage = nextSize === pageSize ? nextPage : 1;
    setPage(targetPage);
    setPageSize(nextSize);
    loadExperiments(searchTerm, targetPage, nextSize);
  };

  useEffect(() => {
    loadExperiments();
    // 只在首次加载时读取第一页；后续查询由搜索和翻页显式触发。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const copyOwnerUid = async (uid: string) => {
    try {
      if (!window.navigator?.clipboard) throw new Error('clipboard unavailable');
      await window.navigator.clipboard.writeText(uid);
      message.success('已复制实验者 UID');
    } catch {
      message.error('复制失败，请手动选择 UID。');
    }
  };

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
      content: '将永久删除数据库中的实验记录、指标和配置，以及仍存在的日志、图表和模型，无法撤销。已有总结不会一并删除，请先下载需要保留的文件。',
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
          setTotal(current => Math.max(0, current - 1));
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
      disabled: exp.available_files ? !exp.available_files.includes('run.log') : exp.files_available === false,
      onClick: () => handleDownloadFile(exp.name, 'run.log'),
    },
    {
      key: 'images-zip',
      label: '下载图表 (ZIP)',
      icon: <FileImageOutlined />,
      disabled: exp.files_available === false,
      onClick: () => handleDownloadImages(exp),
    },
    {
      key: 'all-files-zip',
      label: '下载全部文件 (ZIP)',
      icon: <FileZipOutlined />,
      disabled: exp.files_available === false,
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
      <Card
        title="历史实验"
        extra={
          <Space wrap>
            <Input.Search
              allowClear
              enterButton="搜索"
              aria-label="搜索历史实验"
              placeholder="搜索实验名称或 UUID"
              value={searchInput}
              onChange={(event) => {
                const value = event.target.value;
                setSearchInput(value);
                if (!value) applySearch('');
              }}
              onSearch={applySearch}
              style={{ width: 280 }}
            />
            <Button onClick={() => loadExperiments(searchTerm, page, pageSize)}>刷新</Button>
          </Space>
        }
      >
        <Spin spinning={loading}>
          <Row className="history-grid">
            {experiments.map((exp) => (
              <Col key={exp.name}>
                <Card
                  className="history-experiment-card"
                  hoverable
                  style={{ cursor: 'pointer' }}
                  onClick={() => handleViewDetail(exp.name)}
                  actions={[
                    <Tooltip title={exp.can_plot === false ? '缺少可用的原始指标文件，已入库数据仍可在详情查看' : '生成图表'} key="plot">
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
                          disabled={deleting.includes(exp.name) || exp.files_available === false}
                          onClick={(e) => e.stopPropagation()}>下载文件</Button>
                      </Dropdown>
                    </Tooltip>,
                    <Tooltip title={ACTIVE_STATUSES.has(exp.status || '') ? '实验运行或清理结束后才可删除' : '永久删除该实验记录及剩余附件'} key="delete">
                      <Button type="text" size="small" danger icon={<DeleteOutlined />}
                        loading={deleting.includes(exp.name)}
                        disabled={ACTIVE_STATUSES.has(exp.status || '') || plotGenerating.includes(exp.name)}
                        onClick={event => { event.stopPropagation(); handleDelete(exp); }}>删除</Button>
                    </Tooltip>,
                  ]}
                >
                  <div className="history-experiment-card__name" title={exp.name}>
                    {exp.name}
                  </div>
                  <Tooltip title={exp.run_id || '旧实验未记录 UUID'}>
                    <Typography.Text
                      type="secondary"
                      ellipsis
                      style={{ display: 'block', fontSize: 11, fontFamily: 'monospace', marginBottom: 4 }}
                    >
                      UUID：{exp.run_id || '未记录'}
                    </Typography.Text>
                  </Tooltip>
                  <Tooltip title={exp.owner ? `UID：${exp.owner.uid}（点击复制）` : '未记录实验者 UID'}>
                    <Typography.Text
                      type="secondary"
                      ellipsis
                      style={{ display: 'block', fontSize: 11, marginBottom: 12, cursor: exp.owner ? 'pointer' : 'default' }}
                      onClick={(event) => {
                        event.stopPropagation();
                        if (exp.owner) copyOwnerUid(exp.owner.uid);
                      }}
                    >
                      实验者：{exp.owner?.username || '未记录'}
                    </Typography.Text>
                  </Tooltip>
                  {exp.status && <Tag color={exp.status === 'failed' || exp.status === 'cleanup_failed' ? 'error' : 'default'}>
                    {RUN_STATUS[exp.status] || '未记录'}
                  </Tag>}
                  {!!exp.warnings?.length && <Tooltip title={exp.warnings.join(' ')}><Tag color="warning">{exp.attachments_missing ? '附件缺失' : '指标不完整'}</Tag></Tooltip>}
                  <Row gutter={8}>
                    <Col span={12}>
                      <Statistic
                        title="ACC"
                        value={
                          exp.accuracy != null
                            ? `${(exp.accuracy * 100).toFixed(2)}%`
                            : '未记录'
                        }
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
                        valueStyle={{ fontSize: 20 }}
                      />
                    </Col>
                  </Row>
                  <div className="history-experiment-card__meta">
                    <div>Loss: {exp.loss?.toFixed(4) ?? '未记录'}</div>
                    <div>数据集: {exp.dataset || 'N/A'}</div>
                    <div>
                      AUC: {exp.auc?.toFixed(3) ?? '未记录'} | 轮次: {exp.last_round ?? '未记录'}
                    </div>
                    {exp.created && <div className="history-experiment-card__date">创建: {exp.created.replace('T', ' ').substring(0, 19)}</div>}
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
          {!loading && experiments.length === 0 && (
            <div style={{ padding: '40px 0', color: '#7b8798', textAlign: 'center' }}>
              {searchTerm ? '没有匹配的实验，请尝试其他名称或 UUID。' : '暂无实验记录。'}
            </div>
          )}
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 20 }}>
            <Pagination
              current={page}
              pageSize={pageSize}
              total={total}
              showSizeChanger
              showQuickJumper
              totalBoundaryShowSizeChanger={0}
              pageSizeOptions={[20, 50, 100, 200]}
              showTotal={(count) => `共 ${count} 个实验`}
              onChange={handlePageChange}
            />
          </div>
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
              <Row gutter={[0, 16]} style={{ marginBottom: 24 }}>
                <Col xs={24} sm={12}>
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
                <Col xs={24} sm={12}>
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
              <Descriptions size="small" column={{ xs: 1, sm: 2, md: 4 }} style={{ marginBottom: 16 }}>
                <Descriptions.Item label="实验者">
                  {detailData.owner ? (
                    <Tooltip title={`UID：${detailData.owner.uid}（点击复制）`}>
                      <Typography.Text style={{ cursor: 'pointer' }}
                        onClick={() => copyOwnerUid(detailData.owner.uid)}>
                        {detailData.owner.username}
                      </Typography.Text>
                    </Tooltip>
                  ) : '未记录'}
                </Descriptions.Item>
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
                  <h4>准确率 / 攻击成功率曲线</h4>
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
                      <XAxis dataKey="round" label={{ value: '轮次', position: 'insideBottom', offset: -4 }} />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        stroke="#8884d8"
                        name="准确率 (ACC)"
                      />
                      <Line
                        type="monotone"
                        dataKey="asr"
                        stroke="#82ca9d"
                        name="攻击成功率 (ASR)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              <h4>指标汇总</h4>
              <Table
                columns={columns}
                rowKey="metric"
                scroll={{ x: 480 }}
                dataSource={metricTableData}
                pagination={false}
                size="small"
                style={{ marginBottom: 16 }}
              />

              {detailData.config_toml && (
                <details style={{ marginBottom: 16 }}>
                  <summary style={{ cursor: 'pointer' }}>实验配置快照（数据库保存）</summary>
                  <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxHeight: 320, overflow: 'auto' }}>
                    {detailData.config_toml}
                  </pre>
                </details>
              )}

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
                <Button disabled={detailData.available_files ? !detailData.available_files.includes('run.log') : detailData.files_available === false} onClick={() => window.open(
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
                    <Row className="detail-image-grid">
                      {detailData.images.map((img: string) => (
                        <Col key={img}>
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