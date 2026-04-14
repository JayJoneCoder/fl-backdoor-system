import React, { useEffect, useState } from 'react';
import {
  Form,
  Select,
  InputNumber,
  Switch,
  Input,
  Button,
  Space,
  message,
  Card,
  Row,
  Col,
  Tooltip,
  Divider,
  Modal,
  Upload,
  Alert,
} from 'antd';
import { QuestionCircleOutlined, SettingOutlined, DownloadOutlined, UploadOutlined, SaveOutlined } from '@ant-design/icons';
import { getConfigSchema, getConfig, updateConfig } from '../../api/client';
import type { UploadFile } from 'antd/es/upload/interface';

interface FieldSchema {
  type: string;
  default: any;
  options?: string[];
  option_labels?: Record<string, string>;
  depends_on?: Record<string, string[]>;
  group: string;
  description?: string;
  ui_only?: boolean;
  linked_to?: string;
}

// 字段名到中文标签的映射 (保持不变)
const FIELD_LABELS: Record<string, string> = {
  'attack-malicious-mode': '恶意客户端选择模式',
  'attack-fixed-clients': '固定恶意客户端ID列表',
  'malicious-ratio': '恶意客户端比例',
  'malicious-count': '恶意客户端数量',
  'attack': '攻击类型',
  'client-defense': '客户端防御',
  'detection': '服务端检测',
  'defense': '聚合防御',
  'num-clients': '客户端总数',
  'num-server-rounds': '训练轮数',
  'fraction-evaluate': '评估客户端比例',
  'local-epochs': '本地训练轮次',
  'learning-rate': '学习率',
  'batch-size': '批次大小',
  'seed': '随机种子',
  'dataset': '数据集',
  'poison-rate': '投毒比例',
  'target-label': '目标标签',
  'trigger-size': '触发器尺寸',
  'wanet-noise': 'WaNet 噪声强度',
  'frequency-mode': '频域变换类型',
  'frequency-band': '频带选择',
  'frequency-window-size': '频域窗口大小',
  'frequency-intensity': '频域强度',
  'frequency-mix-alpha': '频域混合系数',
  'dba-num-sub-patterns': 'DBA 子模式数量',
  'dba-sub-pattern-size': 'DBA 子模式尺寸',
  'dba-global-trigger-value': 'DBA 触发器值',
  'dba-split-strategy': 'DBA 分割策略',
  'dba-global-trigger-location': 'DBA 触发器位置',
  'fcba-num-sub-blocks': 'FCBA 子块数量',
  'fcba-sub-block-size': 'FCBA 子块尺寸',
  'fcba-global-trigger-value': 'FCBA 触发器值',
  'fcba-split-strategy': 'FCBA 分割策略',
  'fcba-global-trigger-location': 'FCBA 触发器位置',
  'client-defense-filter-ratio': '过滤比例',
  'client-defense-min-keep': '最少保留样本',
  'client-defense-scoring-batch-size': '评分批大小',
  'client-defense-use-label-centroids': '使用标签中心',
  'client-defense-label-blend-alpha': '标签混合系数',
  'client-defense-min-class-samples': '每类最少样本',
  'detection-z-threshold': 'Z-Score 阈值',
  'detection-top-k': '最多剔除数',
  'detection-min-clients': '最少参与客户端',
  'detection-cosine-floor': '余弦相似度下限',
  'detection-min-kept-clients': '最少保留客户端',
  'detection-max-reject-fraction': '最大拒绝比例',
  'detection-enable-filter': '启用过滤',
  'detection-percentile': '百分位数阈值',
  'detection-weight-norm': '范数权重',
  'detection-weight-cosine': '余弦权重',
  'detection-min-silhouette': '最小轮廓系数',
  'detection-cluster-score-gap': '聚类分数差距',
  'defense-clip-norm': '裁剪阈值',
  'defense-trim-ratio': '修剪比例',
  'defense-trim-k': '修剪数量',
  'defense-num-malicious': '预估恶意数',
  'defense-krum-k': 'Krum 选择数',
  'results-dir': '结果目录',
  'run-name': '实验名称',
};

// 分组中文名
const GROUP_LABELS: Record<string, string> = {
  core: '核心配置',
  federated: '联邦学习参数',
  attack: '攻击参数',
  defense: '防御参数',
  management: '实验管理',
  other: '其他配置',
};

const ConfigForm: React.FC = () => {
  const [schema, setSchema] = useState<Record<string, FieldSchema>>({});
  const [groups, setGroups] = useState<string[]>([]);
  const [currentConfig, setCurrentConfig] = useState<Record<string, any>>({});
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [, forceUpdate] = useState({});

  // 高级模式相关
  const [advancedModalVisible, setAdvancedModalVisible] = useState(false);
  const [tomlContent, setTomlContent] = useState('');
  const [uploadFileList, setUploadFileList] = useState<UploadFile[]>([]);
  const [advancedLoading, setAdvancedLoading] = useState(false);

  // 加载 schema 和配置
  const loadConfig = async () => {
    try {
      const [schemaRes, configRes] = await Promise.all([
        getConfigSchema(),
        getConfig(),
      ]);
      setSchema(schemaRes.data.fields);
      setGroups([...schemaRes.data.groups, 'other']); // 确保 other 分组存在
      const newConfig = configRes.data;
      setCurrentConfig(newConfig);
      form.setFieldsValue(newConfig);
    } catch (error: any) {
      const detail = error?.response?.data?.detail || error?.message || '未知错误';
      message.error(`加载配置失败：${detail}`);
    }
  };

  useEffect(() => {
    loadConfig();
  }, []);

  // 获取当前 toml 原始内容（用于高级模式）
  const fetchTomlContent = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/config/raw');
      const data = await response.json();
      setTomlContent(data.content);
    } catch (error) {
      message.error('获取配置文件失败');
    }
  };

  const handleOpenAdvanced = () => {
    Modal.confirm({
      title: '高级模式警告',
      content: '您正在进入高级配置模式。直接编辑 TOML 文件可能导致配置错误，请确保您了解所做的更改。',
      okText: '进入',
      cancelText: '取消',
      onOk: async () => {
        await fetchTomlContent();
        setAdvancedModalVisible(true);
      },
    });
  };

  const handleSaveToml = async () => {
    setAdvancedLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/config/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: tomlContent }),
      });
      if (!response.ok) throw new Error('保存失败');
      message.success('配置已保存');
      setAdvancedModalVisible(false);
      await loadConfig();
    } catch (error) {
      message.error('保存失败');
    } finally {
      setAdvancedLoading(false);
    }
  };

  const handleDownloadToml = () => {
    const blob = new Blob([tomlContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pyproject.toml';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImportToml = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setTomlContent(e.target?.result as string);
    };
    reader.readAsText(file);
    return false; // 阻止自动上传
  };

  const isFieldVisible = (fieldName: string): boolean => {
    const fieldSchema = schema[fieldName];
    if (!fieldSchema?.depends_on) return true;

    for (const [depField, allowedValues] of Object.entries(fieldSchema.depends_on)) {
      const currentValue = form.getFieldValue(depField);
      if (!allowedValues.includes(currentValue)) {
        return false;
      }
    }
    return true;
  };

  const handleValuesChange = (changedValues: any) => {
    if ('malicious-count' in changedValues) {
      const count = changedValues['malicious-count'];
      const numClients = form.getFieldValue('num-clients') || 10;
      const ratio = numClients > 0 ? count / numClients : 0;
      form.setFieldValue('malicious-ratio', ratio);
    }
    if ('num-clients' in changedValues) {
      const numClients = changedValues['num-clients'];
      const count = form.getFieldValue('malicious-count') || 2;
      const ratio = numClients > 0 ? count / numClients : 0;
      form.setFieldValue('malicious-ratio', ratio);
    }
    forceUpdate({});
  };

  const onFinish = async (values: any) => {
    setLoading(true);
    try {
      const updates = { ...values };
      Object.keys(schema).forEach((key) => {
        if (schema[key]?.ui_only) delete updates[key];
      });
      await updateConfig(updates);
      message.success('配置已保存');
    } catch (error) {
      message.error('保存失败');
    } finally {
      setLoading(false);
    }
  };

  const renderField = (fieldName: string) => {
    if (!isFieldVisible(fieldName)) return null;

    const fieldSchema = schema[fieldName];
    const commonProps = {
      style: { width: '100%' },
    };

    // 未知 schema 的字段（来自 other 分组）统一渲染为输入框
    if (!fieldSchema) {
      return <Input {...commonProps} placeholder={FIELD_LABELS[fieldName] || fieldName} />;
    }

    switch (fieldSchema.type) {
      case 'string':
        if (fieldSchema.options) {
          return (
            <Select {...commonProps} placeholder={`请选择${FIELD_LABELS[fieldName] || ''}`}>
              {fieldSchema.options.map((opt) => {
                const label = fieldSchema.option_labels?.[opt] ?? opt;
                return (
                  <Select.Option key={opt} value={opt}>
                    {label}
                  </Select.Option>
                );
              })}
            </Select>
          );
        }
        return <Input {...commonProps} placeholder={fieldSchema.description} />;

      case 'integer':
      case 'float':
        return <InputNumber {...commonProps} placeholder={fieldSchema.description} />;

      case 'boolean':
        return <Switch />;

      default:
        return <Input {...commonProps} />;
    }
  };

  // 专门渲染防御组的逻辑
  const renderDefenseGroup = (defenseFields: string[]) => {
    const mainFields = ['client-defense', 'detection', 'defense'];
    const mainFieldItems = defenseFields.filter(f => mainFields.includes(f));
    const otherFields = defenseFields.filter(f => !mainFields.includes(f));

    const clientFields = otherFields.filter(f => f.startsWith('client-defense-'));
    const detectionFields = otherFields.filter(f => f.startsWith('detection-'));
    const aggregationFields = otherFields.filter(f => f.startsWith('defense-'));

    const clientVisible = clientFields.some(f => isFieldVisible(f));
    const detectionVisible = detectionFields.some(f => isFieldVisible(f));
    const aggregationVisible = aggregationFields.some(f => isFieldVisible(f));

    return (
      <Card
        title={GROUP_LABELS.defense || '防御参数'}
        style={{ marginBottom: 16 }}
        extra={
          <Tooltip title="配置组: defense">
            <QuestionCircleOutlined />
          </Tooltip>
        }
      >
        <Row gutter={16}>
          {mainFieldItems.map(key => {
            const visible = isFieldVisible(key);
            return visible ? (
              <Col span={8} key={key}>
                <Form.Item
                  name={key}
                  label={FIELD_LABELS[key] || key}
                  tooltip={`${key}: ${schema[key]?.description || ''}`}
                >
                  {renderField(key)}
                </Form.Item>
              </Col>
            ) : null;
          })}
        </Row>
        {clientVisible && (
          <>
            <Divider style={{ margin: '16px 0 8px' }}>客户端防御参数</Divider>
            <Row gutter={16}>
              {clientFields.map(key => {
                const visible = isFieldVisible(key);
                return visible ? (
                  <Col span={8} key={key}>
                    <Form.Item
                      name={key}
                      label={FIELD_LABELS[key] || key}
                      tooltip={`${key}: ${schema[key]?.description || ''}`}
                    >
                      {renderField(key)}
                    </Form.Item>
                  </Col>
                ) : null;
              })}
            </Row>
          </>
        )}
        {detectionVisible && (
          <>
            <Divider style={{ margin: '16px 0 8px' }}>服务端检测参数</Divider>
            <Row gutter={16}>
              {detectionFields.map(key => {
                const visible = isFieldVisible(key);
                return visible ? (
                  <Col span={8} key={key}>
                    <Form.Item
                      name={key}
                      label={FIELD_LABELS[key] || key}
                      tooltip={`${key}: ${schema[key]?.description || ''}`}
                    >
                      {renderField(key)}
                    </Form.Item>
                  </Col>
                ) : null;
              })}
            </Row>
          </>
        )}
        {aggregationVisible && (
          <>
            <Divider style={{ margin: '16px 0 8px' }}>聚合防御参数</Divider>
            <Row gutter={16}>
              {aggregationFields.map(key => {
                const visible = isFieldVisible(key);
                return visible ? (
                  <Col span={8} key={key}>
                    <Form.Item
                      name={key}
                      label={FIELD_LABELS[key] || key}
                      tooltip={`${key}: ${schema[key]?.description || ''}`}
                    >
                      {renderField(key)}
                    </Form.Item>
                  </Col>
                ) : null;
              })}
            </Row>
          </>
        )}
      </Card>
    );
  };

  // 收集未定义在 schema 中的字段（用于 other 分组）
const getUnknownFields = (): string[] => {
  return Object.keys(currentConfig).filter(key => !schema[key]);
};

  return (
    <>
      <div style={{ marginBottom: 16, textAlign: 'right' }}>
        <Button icon={<SettingOutlined />} onClick={handleOpenAdvanced}>
          高级模式
        </Button>
      </div>
      <Form
        form={form}
        layout="vertical"
        onFinish={onFinish}
        onValuesChange={(changed, _all) => {
          handleValuesChange(changed);
        }}
        initialValues={currentConfig}
      >
        {groups.map((group) => {
          // 处理 other 分组：只展示未知字段
          if (group === 'other') {
            const unknownFields = getUnknownFields();
            if (unknownFields.length === 0) return null;
            return (
              <Card
                key="other"
                title={GROUP_LABELS.other}
                style={{ marginBottom: 16 }}
                extra={
                  <Tooltip title="无法被标准表单解析的字段，可直接编辑">
                    <QuestionCircleOutlined />
                  </Tooltip>
                }
              >
                <Row gutter={16}>
                  {unknownFields.map((key) => (
                    <Col span={8} key={key}>
                      <Form.Item
                        name={key}
                        label={FIELD_LABELS[key] || key}
                        tooltip={`${key} (自定义字段)`}
                      >
                        {renderField(key)}
                      </Form.Item>
                    </Col>
                  ))}
                </Row>
              </Card>
            );
          }

          const groupFields = Object.keys(schema).filter(key => schema[key].group === group);
          if (group === 'defense') {
            return renderDefenseGroup(groupFields);
          }
          return (
            <Card
              key={group}
              title={GROUP_LABELS[group] || group}
              style={{ marginBottom: 16 }}
              extra={
                <Tooltip title={`配置组: ${group}`}>
                  <QuestionCircleOutlined />
                </Tooltip>
              }
            >
              <Row gutter={16}>
                {groupFields.map((key) => {
                  const visible = isFieldVisible(key);
                  return visible ? (
                    <Col span={8} key={key}>
                      <Form.Item
                        name={key}
                        label={FIELD_LABELS[key] || key}
                        tooltip={`${key}: ${schema[key]?.description || ''}`}
                      >
                        {renderField(key)}
                      </Form.Item>
                    </Col>
                  ) : null;
                })}
              </Row>
            </Card>
          );
        })}
        <Form.Item>
          <Space>
            <Button type="primary" htmlType="submit" loading={loading}>
              保存配置
            </Button>
            <Button onClick={() => form.resetFields()}>重置</Button>
          </Space>
        </Form.Item>
      </Form>

      {/* 高级模式弹窗 */}
      <Modal
        title="高级配置（直接编辑 pyproject.toml）"
        open={advancedModalVisible}
        onCancel={() => setAdvancedModalVisible(false)}
        width={900}
        footer={[
          <Button key="download" icon={<DownloadOutlined />} onClick={handleDownloadToml}>
            下载
          </Button>,
          <Upload
            key="upload"
            fileList={uploadFileList}
            beforeUpload={handleImportToml}
            onChange={({ fileList }) => setUploadFileList(fileList)}
            maxCount={1}
            showUploadList={false}
          >
            <Button icon={<UploadOutlined />}>导入</Button>
          </Upload>,
          <Button key="cancel" onClick={() => setAdvancedModalVisible(false)}>
            取消
          </Button>,
          <Button key="save" type="primary" icon={<SaveOutlined />} loading={advancedLoading} onClick={handleSaveToml}>
            保存并应用
          </Button>,
        ]}
      >
        <Alert
          message="警告"
          description="直接编辑配置文件可能导致系统无法正常运行。请确保您熟悉 TOML 语法和配置项含义。"
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Input.TextArea
          value={tomlContent}
          onChange={(e) => setTomlContent(e.target.value)}
          rows={20}
          style={{ fontFamily: 'monospace' }}
          placeholder="加载中..."
        />
      </Modal>
    </>
  );
};

export default ConfigForm;