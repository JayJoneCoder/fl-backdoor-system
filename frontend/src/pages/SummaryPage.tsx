import React, { useEffect, useState } from 'react';
import {
  Card,
  Button,
  Table,
  Modal,
  Checkbox,
  message,
  Space,
  Input,
  Image,
  Row,
  Col,
  Spin,
  Divider,
  Dropdown,
} from 'antd';
import type { MenuProps } from 'antd';
import {
  FileTextOutlined,
  PlusOutlined,
  BarChartOutlined,
  CodeOutlined,
  DownloadOutlined,
  FileZipOutlined,
  FileImageOutlined,
  DeleteOutlined,
} from '@ant-design/icons';
import {
  listExperiments,
  listSummaries,
  createSummary,
  generateSummaryPlots,
  downloadSummaryZip,
  getSummaryDetail,
  deleteSummary,
} from '../api/client';
import { getBackendUrl } from '../api/endpoints';
import { getUserErrorMessage } from '../api/checkedFetch';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';

interface ExperimentItem {
  name: string;
  run_id?: string | null;
  has_metrics?: boolean;
}

interface SummaryItem {
  name: string;
  experiments: string[];
  created: string | null;
}

// 汇总表格的列名来自 summary.csv，展示时统一转换为中文。
const SUMMARY_COLUMN_LABELS: Record<string, string> = {
  experiment: '实验',
  accuracy: '准确率',
  asr: '攻击成功率',
  loss: '损失',
  last_round: '最终轮次',
  precision: '精确率',
  recall: '召回率',
  fpr: '假正例率',
  auc: 'AUC',
  dataset: '数据集',
  avg_malicious_removal_rate: '平均恶意移除率',
  avg_benign_removal_rate: '平均良性移除率',
  avg_kept_malicious: '平均保留恶意数',
  avg_selected_experiments: '平均选中实验数',
  avg_selected_clients: '平均选中客户端数',
};

const SummaryPage: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentItem[]>([]);
  const [summaries, setSummaries] = useState<SummaryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedExps, setSelectedExps] = useState<string[]>([]);
  const [summaryName, setSummaryName] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [summarySearch, setSummarySearch] = useState('');
  const [summaryPage, setSummaryPage] = useState(1);
  const [summaryPageSize, setSummaryPageSize] = useState(50);
  const [experimentSearch, setExperimentSearch] = useState('');

  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [detailData, setDetailData] = useState<any>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [plotGenerating, setPlotGenerating] = useState<string | null>(null);

  const summaryKeyword = summarySearch.trim().toLowerCase();
  const filteredSummaries = summaryKeyword
    ? summaries.filter((summary) =>
        summary.name.toLowerCase().includes(summaryKeyword) ||
        (summary.experiments || []).some((name) => name.toLowerCase().includes(summaryKeyword)))
    : summaries;

  const experimentKeyword = experimentSearch.trim().toLowerCase();
  const filteredExperiments = experimentKeyword
    ? experiments.filter((experiment) =>
        experiment.name.toLowerCase().includes(experimentKeyword) ||
        (experiment.run_id || '').toLowerCase().includes(experimentKeyword))
    : experiments;

  const loadData = async () => {
    setLoading(true);
    try {
      const [expRes, sumRes] = await Promise.all([listExperiments(), listSummaries()]);
      setExperiments(expRes.data.filter((item: ExperimentItem) => item.has_metrics !== false));
      setSummaries(sumRes.data);
    } catch (error) {
      message.error(getUserErrorMessage(error, '加载数据失败，请稍后重试。'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleCreateSummary = async () => {
    if (selectedExps.length === 0) {
      message.warning('请至少选择一个实验');
      return;
    }
    setSubmitting(true);
    try {
      await createSummary({
        experiments: selectedExps,
        output_name: summaryName || undefined,
      });
      message.success('总结生成成功');
      setModalVisible(false);
      setSelectedExps([]);
      setSummaryName('');
      loadData();
    } catch (error) {
      message.error(getUserErrorMessage(error, '创建总结失败，请查看后端日志。'));
    } finally {
      setSubmitting(false);
    }
  };

  const handleGeneratePlots = async (summaryName: string) => {
    setPlotGenerating(summaryName);
    try {
      await generateSummaryPlots(summaryName);
      message.success('图表生成成功');
      if (detailData && detailData.name === summaryName) {
        const res = await getSummaryDetail(summaryName);
        setDetailData(res.data);
      }
    } catch (error) {
      message.error(getUserErrorMessage(error, '图表生成失败，请查看后端日志。'));
    } finally {
      setPlotGenerating(null);
    }
  };

  const downloadFile = (url: string, filename: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadZip = async (summaryName: string, response?: { data: Blob }) => {
    try {
      const res = response ?? await downloadSummaryZip(summaryName);
      const blob = new Blob([res.data], { type: 'application/zip' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${summaryName}_images.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      message.success('开始下载压缩包');
    } catch (error) {
      message.error(getUserErrorMessage(error, '下载失败，请稍后重试。'));
    }
  };

  const handleDownloadImages = async (summaryName: string) => {
    try {
      const res = await downloadSummaryZip(summaryName);
      await handleDownloadZip(summaryName, res);
    } catch (error: any) {
      if (error.response?.status === 404) {
        Modal.confirm({
          title: '该总结暂无图表',
          content: '是否先生成图表？',
          okText: '生成并下载',
          cancelText: '取消',
          onOk: async () => {
            try {
              await generateSummaryPlots(summaryName);
              message.success('图表生成成功，开始下载');
              await handleDownloadZip(summaryName);
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

  const handleDeleteSummary = (summaryName: string) => {
    Modal.confirm({
      title: '删除总结',
      content: `确定要删除总结“${summaryName}”吗？该总结目录中的 CSV、LaTeX 表格和图表都会被删除，且无法恢复。`,
      okText: '删除',
      okButtonProps: { danger: true },
      cancelText: '取消',
      onOk: async () => {
        try {
          await deleteSummary(summaryName);
          message.success('总结已删除');
          if (detailData?.name === summaryName) {
            setDetailModalVisible(false);
            setDetailData(null);
          }
          const remaining = filteredSummaries.filter((item) => item.name !== summaryName).length;
          if (summaryPage > 1 && remaining <= (summaryPage - 1) * summaryPageSize) {
            setSummaryPage(summaryPage - 1);
          }
          await loadData();
        } catch (error) {
          message.error(getUserErrorMessage(error, '删除总结失败，请稍后重试。'));
        }
      },
    });
  };

  const handleViewDetail = async (summaryName: string) => {
    setDetailLoading(true);
    setDetailModalVisible(true);
    try {
      const res = await getSummaryDetail(summaryName);
      setDetailData(res.data);
    } catch (error) {
      message.error(getUserErrorMessage(error, '加载详情失败，请稍后重试。'));
      setDetailModalVisible(false);
    } finally {
      setDetailLoading(false);
    }
  };

  const getDownloadMenuItems = (summaryName: string): MenuProps['items'] => [
    {
      key: 'csv',
      label: '总结 CSV',
      icon: <FileTextOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        window.open(getBackendUrl(`/results/summaries/${encodeURIComponent(summaryName)}/summary.csv`));
      },
    },
    {
      key: 'latex',
      label: 'LaTeX 表格',
      icon: <CodeOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        window.open(getBackendUrl(`/results/summaries/${encodeURIComponent(summaryName)}/summary_table.tex`));
      },
    },
    {
      type: 'divider',
    },
    {
      key: 'images-zip',
      label: '下载图表 (ZIP)',
      icon: <FileImageOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        handleDownloadImages(summaryName);
      },
    },
    {
      key: 'all-zip',
      label: '下载全部文件 (ZIP)',
      icon: <FileZipOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        handleDownloadZip(summaryName);
      },
    },
  ];

  const columns = [
    { title: '总结名称', dataIndex: 'name', key: 'name', width: 200 },
    {
      title: '包含实验',
      dataIndex: 'experiments',
      key: 'experiments',
      render: (list: string[]) => list?.join(', ') || '',
    },
    { title: '创建时间', dataIndex: 'created', key: 'created', width: 200,  render: (v: string) => v ? v.replace('T', ' ').substring(0, 22) : '-' },
    {
      title: '操作',
      key: 'action',
      width: 320,
      render: (_: any, record: SummaryItem) => (
        <Space wrap size="small">
          <Button
            size="small"
            icon={<BarChartOutlined />}
            onClick={(e) => {
              e.stopPropagation();
              handleGeneratePlots(record.name);
            }}
            loading={plotGenerating === record.name}
          >
            生成图表
          </Button>
          <Dropdown menu={{ items: getDownloadMenuItems(record.name) }} trigger={['click']}>
            <Button size="small" icon={<DownloadOutlined />} onClick={(e) => e.stopPropagation()}>
              下载
            </Button>
          </Dropdown>
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={(e) => {
              e.stopPropagation();
              handleDeleteSummary(record.name);
            }}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  const prepareChartData = () => {
    if (!detailData?.experiments_curve_data) return [];
    const expNames = Object.keys(detailData.experiments_curve_data);
    if (expNames.length === 0) return [];
    const firstExp = expNames[0];
    const rounds = detailData.experiments_curve_data[firstExp].round;
    const data = rounds.map((r: number, idx: number) => {
      const point: any = { round: r };
      expNames.forEach((name) => {
        point[`${name}_acc`] = detailData.experiments_curve_data[name].accuracy[idx];
        point[`${name}_asr`] = detailData.experiments_curve_data[name].asr[idx];
      });
      return point;
    });
    return data;
  };

  const chartData = prepareChartData();
  const expNames = detailData ? Object.keys(detailData.experiments_curve_data || {}) : [];

  return (
    <div className="workspace-page">
      <div className="page-heading">
        <h2 className="page-title">实验比较总结</h2>
        <p className="page-description">对比不同实验的准确率与攻击成功率，生成汇总报告和可视化图表。</p>
      </div>
      <Card>
        <Space wrap style={{ marginBottom: 20 }}>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
            新建总结
          </Button>
          <Input.Search
            allowClear
            aria-label="搜索比较总结"
            placeholder="搜索总结名称或包含的实验名称"
            value={summarySearch}
            onChange={(event) => {
              setSummarySearch(event.target.value);
              setSummaryPage(1);
            }}
            style={{ width: 320 }}
          />
          <Button onClick={loadData}>刷新</Button>
        </Space>
        <Table
          dataSource={filteredSummaries}
          columns={columns}
          rowKey="name"
          loading={loading}
          scroll={{ x: 1000 }}
          pagination={{
            current: summaryPage,
            pageSize: summaryPageSize,
            total: filteredSummaries.length,
            showSizeChanger: true,
            showQuickJumper: true,
            totalBoundaryShowSizeChanger: 0,
            pageSizeOptions: [20, 50, 100, 200],
            showTotal: (count) => `共 ${count} 条总结`,
            onChange: (nextPage, nextSize) => {
              setSummaryPage(nextSize === summaryPageSize ? nextPage : 1);
              setSummaryPageSize(nextSize);
            },
          }}
          onRow={(record) => ({
            style: { cursor: 'pointer' },
            onClick: () => handleViewDetail(record.name),
          })}
        />
      </Card>

      <Modal
        title="选择实验并命名总结"
        open={modalVisible}
        onOk={handleCreateSummary}
        onCancel={() => setModalVisible(false)}
        confirmLoading={submitting}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <label htmlFor="summary-name">总结名称</label>
          <Input
            id="summary-name"
            placeholder="总结名称（可选，默认时间戳）"
            value={summaryName}
            onChange={(e) => setSummaryName(e.target.value)}
          />
          <Input.Search
            allowClear
            aria-label="搜索可选实验"
            placeholder="搜索实验名称或 UUID"
            value={experimentSearch}
            onChange={(event) => setExperimentSearch(event.target.value)}
          />
          <Checkbox.Group
            className="summary-experiment-picker"
            aria-label="选择参与比较的实验"
            options={filteredExperiments.map((e) => ({ label: e.name, value: e.name }))}
            value={selectedExps}
            onChange={(values) => setSelectedExps(values as string[])}
          />
          <span style={{ color: '#7b8798', fontSize: 12 }}>
            已选择 {selectedExps.length} 个实验
            {experimentKeyword ? `，筛选出 ${filteredExperiments.length} 个` : ''}
          </span>
        </Space>
      </Modal>

      <Modal
        title={`总结详情: ${detailData?.name}`}
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={1200}
      >
        <Spin spinning={detailLoading}>
          {detailData && (
            <>
              <Space wrap style={{ marginBottom: 16 }}>
                <Dropdown menu={{ items: getDownloadMenuItems(detailData.name) }} trigger={['click']}>
                  <Button icon={<DownloadOutlined />}>下载</Button>
                </Dropdown>
                <Button danger icon={<DeleteOutlined />} onClick={() => handleDeleteSummary(detailData.name)}>
                  删除本总结
                </Button>
              </Space>
              {expNames.length > 0 && (
                <>
                  <h4>所有实验准确率（ACC）曲线对比</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      {expNames.map((name, index) => (
                        <Line
                          key={name}
                          type="monotone"
                          dataKey={`${name}_acc`}
                          name={name}
                          stroke={['#2563b8', '#c17a37', '#398673', '#8467a7', '#b95263', '#64778f'][index % 6]}
                          strokeWidth={2}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                  <div style={{ textAlign: 'right', color: '#8592a5', fontSize: 12, paddingRight: 18, marginTop: 2 }}>轮次</div>

                  <h4 style={{ marginTop: 24 }}>所有实验攻击成功率（ASR）曲线对比</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      {expNames.map((name, index) => (
                        <Line
                          key={name}
                          type="monotone"
                          dataKey={`${name}_asr`}
                          name={name}
                          stroke={['#2563b8', '#c17a37', '#398673', '#8467a7', '#b95263', '#64778f'][index % 6]}
                          strokeWidth={2}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                  <div style={{ textAlign: 'right', color: '#8592a5', fontSize: 12, paddingRight: 18, marginTop: 2 }}>轮次</div>
                  <Divider />
                </>
              )}

              <h4>汇总表格</h4>
              <Table<Record<string, unknown>>
                dataSource={detailData.summary_table}
                rowKey={record => String(record.experiment ?? record.name ?? JSON.stringify(record))}
                columns={Object.keys(detailData.summary_table[0] || {}).map((k) => ({
                  title: SUMMARY_COLUMN_LABELS[k] || k,
                  dataIndex: k,
                  key: k,
                }))}
                pagination={false}
                size="small"
                scroll={{ x: 1000 }}
              />

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
                                onClick={() => downloadFile(getBackendUrl(`/results/summaries/${encodeURIComponent(detailData.name)}/${encodeURIComponent(img)}`), img)}
                              />
                            }
                          >
                            <Image
                              src={getBackendUrl(`/results/summaries/${encodeURIComponent(detailData.name)}/${encodeURIComponent(img)}`)}
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
    </div>
  );
};

export default SummaryPage;