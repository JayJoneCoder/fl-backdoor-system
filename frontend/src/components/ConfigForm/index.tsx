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
  Dropdown,
  Table,
} from 'antd';
import {
  QuestionCircleOutlined,
  SettingOutlined,
  DownloadOutlined,
  UploadOutlined,
  SaveOutlined,
  HistoryOutlined,
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { getConfigSchema, getConfig, updateConfig } from '../../api/client';
import type { UploadFile } from 'antd/es/upload/interface';
import type { MenuProps } from 'antd';

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

  // 备份管理相关
  const [backupModalVisible, setBackupModalVisible] = useState(false);
  const [backups, setBackups] = useState<any[]>([]);
  const [backupLoading, setBackupLoading] = useState(false);
  const [selectedBackup, setSelectedBackup] = useState<string | null>(null);
  const [backupContent, setBackupContent] = useState('');
  const [viewBackupModalVisible, setViewBackupModalVisible] = useState(false);
  const [editingBackupContent, setEditingBackupContent] = useState('');

  // 加载 schema 和配置
  const loadConfig = async () => {
    try {
      const [schemaRes, configRes] = await Promise.all([
        getConfigSchema(),
        getConfig(),
      ]);
      setSchema(schemaRes.data.fields);
      setGroups([...schemaRes.data.groups, 'other']);
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

  const handleSaveToml = async (apply: boolean = true) => {
    setAdvancedLoading(true);
    try {
      if (apply) {
        // 保存并应用：询问是否备份当前生效的配置
        Modal.confirm({
          title: '保存并应用配置',
          content: (
            <div>
              <p>是否备份当前正在生效的配置？</p>
              <Input
                placeholder="可选：输入备份名称（留空则使用时间戳）"
                id="backupNameInput"
                style={{ marginTop: 8 }}
              />
            </div>
          ),
          okText: '保存应用并备份',
          cancelText: '仅保存并应用',
          onOk: async () => {
            const input = document.getElementById('backupNameInput') as HTMLInputElement;
            const customName = input?.value?.trim();
            // 1. 写入 pyproject.toml（应用配置）
            await fetch('http://localhost:8000/api/config/raw', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ content: tomlContent }),
            });
            // 2. 创建当前配置的备份
            await fetch('http://localhost:8000/api/config/backups', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ name: customName || undefined }),
            });
            message.success('配置已保存并应用，备份已创建');
            setAdvancedModalVisible(false);
            await loadConfig();
            loadBackups();
          },
          onCancel: async () => {
            // 仅保存并应用，不备份
            await fetch('http://localhost:8000/api/config/raw', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ content: tomlContent }),
            });
            message.success('配置已保存并应用');
            setAdvancedModalVisible(false);
            await loadConfig();
          },
          // 取消
          footer: (_, { OkBtn, CancelBtn }) => (
            <>
              <Button onClick={() => Modal.destroyAll()}>取消</Button>
              <CancelBtn />
              <OkBtn />
            </>
          ),
        });
      } else {
        // 仅创建备份：将编辑区内容保存为独立备份文件，不修改 pyproject.toml
        Modal.confirm({
          title: '创建配置草稿备份',
          content: (
            <div>
              <p>将当前编辑器中的内容保存为备份文件，不会影响正在运行的配置。</p>
              <Input
                placeholder="可选：输入备份名称（留空则使用时间戳）"
                id="backupNameInput"
                style={{ marginTop: 8 }}
              />
            </div>
          ),
          okText: '创建备份',
          cancelText: '取消',
          onOk: async () => {
            const input = document.getElementById('backupNameInput') as HTMLInputElement;
            const customName = input?.value?.trim();
            
            const res = await fetch('http://localhost:8000/api/config/backups/content', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ 
                content: tomlContent,
                name: customName || undefined,
                on_conflict: 'ask'
              }),
            });
            
            if (res.status === 409) {
              // 重名冲突处理（可复用类似逻辑）
              const filename = customName ? `${customName}.toml` : '默认名称';
              Modal.confirm({
                title: '文件已存在',
                content: `备份文件 "${filename}" 已存在，请选择处理方式：`,
                okText: '自动增加后缀',
                cancelText: '覆盖',
                onOk: async () => {
                  // 重新请求，使用 auto 策略
                  const retryRes = await fetch('http://localhost:8000/api/config/backups/content', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                      content: tomlContent,
                      name: customName || undefined,
                      on_conflict: 'auto'
                    }),
                  });
                  const retryData = await retryRes.json();
                  message.success(`备份已创建: ${retryData.filename}`);
                  loadBackups();
                },
                onCancel: async () => {
                  // 覆盖
                  const retryRes = await fetch('http://localhost:8000/api/config/backups/content', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                      content: tomlContent,
                      name: customName || undefined,
                      on_conflict: 'overwrite'
                    }),
                  });
                  const retryData = await retryRes.json();
                  message.success(`备份已创建: ${retryData.filename}`);
                  loadBackups();
                },
                footer: (_, { OkBtn, CancelBtn }) => (
                  <>
                    <Button onClick={() => Modal.destroyAll()}>取消</Button>
                    <CancelBtn />
                    <OkBtn />
                  </>
                ),
              });
              return;
            }
            
            if (!res.ok) throw new Error('创建失败');
            const data = await res.json();
            message.success(`备份已创建: ${data.filename}`);
            loadBackups();
          },
        });
      }
    } catch (error) {
      message.error('操作失败');
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

  const proceedImport = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      let content = e.target?.result as string;
      // 标准化换行符
      content = content.replace(/\r\n/g, '\n');
      setTomlContent(content);
    };
    reader.readAsText(file);
    return false;
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

  const getUnknownFields = (): string[] => {
    return Object.keys(currentConfig).filter(key => !schema[key]);
  };

  // -------------------- 备份管理功能 --------------------
  const loadBackups = async () => {
    setBackupLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/config/backups');
      const data = await res.json();
      setBackups(data.backups);
    } catch (error) {
      message.error('加载备份列表失败');
    } finally {
      setBackupLoading(false);
    }
  };

  const handleCreateBackup = async (customName?: string, onConflict: string = 'ask'): Promise<void> => {
    try {
      const res = await fetch('http://localhost:8000/api/config/backups', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: customName, on_conflict: onConflict }),
      });
      
      if (res.status === 409) {
        // 重名冲突，弹窗询问
        Modal.confirm({
          title: '文件已存在',
          content: `备份文件 "${customName || '默认名称'}" 已存在，请选择处理方式：`,
          okText: '自动增加后缀',
          cancelText: '覆盖',
          onOk: async () => {
            await handleCreateBackup(customName, 'auto');
          },
          onCancel: async () => {
            await handleCreateBackup(customName, 'overwrite');
          },
          footer: (_, { OkBtn, CancelBtn }) => (
            <>
              <Button onClick={() => Modal.destroyAll()}>取消</Button>
              <CancelBtn />
              <OkBtn />
            </>
          ),
        });
        return;
      }
      
      if (!res.ok) throw new Error('创建失败');
      const data = await res.json();
      message.success(`备份已创建: ${data.filename}`);
      await loadBackups();
    } catch (error) {
      message.error('创建备份失败');
      throw error;
    }
  };

  const handleImportToml = (file: File) => {
    Modal.confirm({
      title: '导入配置',
      content: '是否备份当前配置？',
      okText: '备份并导入',
      cancelText: '直接导入',
      onOk: async () => {
        try {
          await handleCreateBackup();
          proceedImport(file);
        } catch (error) {
          message.error('备份失败，取消导入');
        }
      },
      onCancel: () => proceedImport(file),
    });
    return false;
  };

  const handleViewBackup = async (filename: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/config/backups/${filename}`);
      const data = await res.json();
      setSelectedBackup(filename);
      setBackupContent(data.content);
      setEditingBackupContent(data.content);
      setViewBackupModalVisible(true);
    } catch (error) {
      message.error('读取备份失败');
    }
  };

  const handleSaveBackupContent = async () => {
    if (!selectedBackup) return;
    try {
      const response = await fetch(`http://localhost:8000/api/config/backups/${selectedBackup}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: editingBackupContent }),
      });
      if (!response.ok) throw new Error('保存失败');
      message.success('备份已更新');
      setViewBackupModalVisible(false);
      loadBackups();
    } catch (error) {
      message.error('更新备份失败');
    }
  };

  const handleRestoreBackup = async (filename: string) => {
    let backupNameInput: string | undefined = undefined;

    Modal.confirm({
      title: '恢复配置',
      content: (
        <div>
          <p>从备份 <strong>{filename}</strong> 恢复配置。</p>
          <p>是否备份当前配置？</p>
          <Input
            placeholder="可选：输入备份名称（留空则使用时间戳）"
            onChange={(e) => { backupNameInput = e.target.value; }}
            style={{ marginTop: 8 }}
          />
        </div>
      ),
      okText: '恢复并备份',
      cancelText: '仅恢复不备份',
      onOk: async () => {
        try {
          const res = await fetch('http://localhost:8000/api/config/restore', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename, backup_current: true, backup_name: backupNameInput || undefined }),
          });
          if (!res.ok) throw new Error('恢复失败');
          message.success('配置已恢复，当前配置已备份');
          await loadConfig();
          loadBackups();
        } catch (error) {
          message.error('恢复失败');
        }
      },
      onCancel: async () => {
        try {
          const res = await fetch('http://localhost:8000/api/config/restore', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename, backup_current: false }),
          });
          if (!res.ok) throw new Error('恢复失败');
          message.success('配置已恢复');
          await loadConfig();
          loadBackups();
        } catch (error) {
          message.error('恢复失败');
        }
      },
      // 添加第三个按钮：取消
      footer: (_, { OkBtn, CancelBtn }) => (
        <>
          <Button onClick={() => Modal.destroyAll()}>取消</Button>
          <CancelBtn />
          <OkBtn />
        </>
      ),
    });
  };

  const handleDeleteBackup = async (filename: string) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除备份 ${filename} 吗？`,
      okText: '删除',
      okButtonProps: { danger: true },
      onOk: async () => {
        try {
          await fetch(`http://localhost:8000/api/config/backups/${filename}`, { method: 'DELETE' });
          message.success('备份已删除');
          loadBackups();
        } catch (error) {
          message.error('删除失败');
        }
      },
    });
  };

  const resetMenuItems: MenuProps['items'] = [
    {
      key: 'form',
      label: '撤销未保存的更改',
      onClick: () => form.resetFields(),
    },
    {
      key: 'last',
      label: '恢复到上次备份 (.bak)',
      onClick: () => {
        const filename = 'pyproject.toml.bak';
        let backupNameInput: string | undefined = undefined;

        Modal.confirm({
          title: '恢复配置',
          content: (
            <div>
              <p>从备份 <strong>{filename}</strong> 恢复配置。</p>
              <p>是否备份当前配置？</p>
              <Input
                placeholder="可选：输入备份名称（留空则使用时间戳）"
                onChange={(e) => { backupNameInput = e.target.value; }}
                style={{ marginTop: 8 }}
              />
            </div>
          ),
          okText: '恢复并备份',
          cancelText: '仅恢复不备份',
          onOk: async () => {
            try {
              const res = await fetch('http://localhost:8000/api/config/restore', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, backup_current: true, backup_name: backupNameInput || undefined }),
              });
              if (!res.ok) throw new Error('恢复失败');
              message.success('配置已恢复，当前配置已备份');
              await loadConfig();
              loadBackups();
            } catch (error) {
              message.error('恢复失败');
            }
          },
          onCancel: async () => {
            try {
              const res = await fetch('http://localhost:8000/api/config/restore', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, backup_current: false }),
              });
              if (!res.ok) throw new Error('恢复失败');
              message.success('配置已恢复');
              await loadConfig();
              loadBackups();
            } catch (error) {
              message.error('恢复失败');
            }
          },
          footer: (_, { OkBtn, CancelBtn }) => (
            <>
              <Button onClick={() => Modal.destroyAll()}>取消</Button>
              <CancelBtn />
              <OkBtn />
            </>
          ),
        });
      },
    },
    {
      key: 'backup',
      label: '从备份列表恢复...',
      onClick: () => {
        setBackupModalVisible(true);
        loadBackups();
      },
    },
  ];

  return (
    <>
      <div style={{ marginBottom: 16, textAlign: 'right' }}>
        <Space>
          <Button icon={<HistoryOutlined />} onClick={() => { setBackupModalVisible(true); loadBackups(); }}>
            备份管理
          </Button>
          <Button icon={<SettingOutlined />} onClick={handleOpenAdvanced}>
            高级模式
          </Button>
        </Space>
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
            <Tooltip title="保存当前用户设置的配置，并生成 .bak 备份供用户恢复">
              <Button type="primary" htmlType="submit" loading={loading}>
                保存配置
              </Button>
            </Tooltip>
            <Dropdown menu={{ items: resetMenuItems }}>
              <Button>重置 ▼</Button>
            </Dropdown>
          </Space>
        </Form.Item>
      </Form>

      {/* 高级模式弹窗 */}
      <Modal
        title="高级配置模式（直接编辑 pyproject.toml）"
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
          <Button
            key="backup-only"
            onClick={() => handleSaveToml(false)}
          >
            仅创建备份
          </Button>,
          <Button
            key="save"
            type="primary"
            icon={<SaveOutlined />}
            loading={advancedLoading}
            onClick={() => handleSaveToml(true)}
          >
            保存并应用...
          </Button>,
        ]}
      >
        <Alert
          message="警告"
          description="直接编辑配置文件可能导致系统无法正常运行。请确保您熟悉 TOML 语法和配置项含义。强烈建议仅更改[tool.flwr.app.config]下的字段的值，避免修改其他部分。"
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

      {/* 备份管理弹窗 */}
      <Modal
        title="配置备份管理"
        open={backupModalVisible}
        onCancel={() => setBackupModalVisible(false)}
        footer={[
          <Button key="refresh" onClick={loadBackups}>刷新</Button>,
          <Button
            key="new"
            type="primary"
            onClick={() => {
              Modal.confirm({
                title: '新建备份',
                content: (
                  <Input
                    placeholder="可选：输入备份名称"
                    id="backupNameInput"
                  />
                ),
                onOk: () => {
                  const input = document.getElementById('backupNameInput') as HTMLInputElement;
                  handleCreateBackup(input?.value || undefined);
                },
              });
            }}
          >
            立即备份当前配置
          </Button>,
          <Button key="close" onClick={() => setBackupModalVisible(false)}>关闭</Button>,
        ]}
        width={900}
      >
        <Table
          loading={backupLoading}
          dataSource={backups}
          rowKey="filename"
          columns={[
            { title: '文件名', dataIndex: 'filename', key: 'filename' },
            { title: '创建时间', dataIndex: 'timestamp', key: 'timestamp', render: (v) => new Date(v).toLocaleString() },
            { title: '大小', dataIndex: 'size', key: 'size', render: (v) => `${(v / 1024).toFixed(1)} KB` },
            {
              title: '操作',
              key: 'action',
              render: (_, record) => (
                <Space>
                  <Button size="small" icon={<EyeOutlined />} onClick={() => handleViewBackup(record.filename)}>查看</Button>
                  <Button size="small" icon={<ReloadOutlined />} onClick={() => handleRestoreBackup(record.filename)}>恢复</Button>
                  <Button size="small" danger icon={<DeleteOutlined />} onClick={() => handleDeleteBackup(record.filename)}>删除</Button>
                </Space>
              ),
            },
          ]}
        />
      </Modal>

      {/* 查看/编辑备份内容弹窗 */}
      <Modal
        title={`查看备份: ${selectedBackup}`}
        open={viewBackupModalVisible}
        onCancel={() => setViewBackupModalVisible(false)}
        footer={[
          <Button key="restore" type="primary" onClick={() => { handleRestoreBackup(selectedBackup!); setViewBackupModalVisible(false); }}>
            恢复此备份
          </Button>,
          <Button key="save" onClick={handleSaveBackupContent}>
            保存修改
          </Button>,
          <Button key="close" onClick={() => setViewBackupModalVisible(false)}>关闭</Button>,
        ]}
        width={900}
      >
        <Input.TextArea
          value={editingBackupContent}
          onChange={(e) => setEditingBackupContent(e.target.value)}
          rows={20}
          style={{ fontFamily: 'monospace' }}
        />
      </Modal>
    </>
  );
};

export default ConfigForm;