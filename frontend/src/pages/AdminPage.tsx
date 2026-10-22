import { useEffect, useState } from 'react';
import { Alert, Avatar, Button, Card, Empty, Form, Input, Modal, Popconfirm, Table, Tag, message } from 'antd';
import { ReloadOutlined, UserOutlined } from '@ant-design/icons';
import { apiClient } from '../api/client';
import type { PlatformUser } from '../api/auth';
import { getUserErrorMessage } from '../api/checkedFetch';
import { passwordRules } from './LoginPage';

interface UserRow extends PlatformUser { experiment_count: number }
interface TaskRow { uid: string; kind: string; name: string; status: string; current_experiment?: string }
const statuses: Record<string, string> = { pending: '等待中', running: '运行中', paused: '已暂停', stopping: '停止中', starting: '启动中', cleaning: '清理中', cleanup_failed: '清理失败' };

export default function AdminPage() {
  const [data, setData] = useState<{ users: UserRow[]; tasks: TaskRow[] }>({ users: [], tasks: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [resetUser, setResetUser] = useState<UserRow | null>(null);
  const [resetBusy, setResetBusy] = useState(false);
  const [resetForm] = Form.useForm<{ new_password: string; confirm: string }>();
  const [stopOpen, setStopOpen] = useState(false);
  const [stopBusy, setStopBusy] = useState(false);
  const [stopped, setStopped] = useState(false);
  const [stopForm] = Form.useForm<{ password: string }>();
  useEffect(() => {
    let active = true;
    const load = async () => {
      if (stopped) return;
      setLoading(true);
      try { const response = await apiClient.get('/api/admin/overview'); if (active) { setData(response.data); setError(''); } }
      catch (e) { if (active) setError(getUserErrorMessage(e)); }
      finally { if (active) setLoading(false); }
    };
    void load(); const timer = window.setInterval(load, 10000);
    return () => { active = false; clearInterval(timer); };
  }, [refresh, stopped]);
  const nameFor = (uid: string) => data.users.find(user => user.uid === uid)?.username ?? uid;
  const submitReset = async (values: { new_password: string }) => {
    if (!resetUser) return;
    setResetBusy(true);
    try {
      await apiClient.put(`/api/admin/users/${resetUser.uid}/password`, { new_password: values.new_password });
      message.success(`已重置 ${resetUser.username} 的密码，该用户下次登录需修改密码。`);
      setResetUser(null); resetForm.resetFields(); setRefresh(n => n + 1);
    } catch (e) { message.error(getUserErrorMessage(e)); }
    finally { setResetBusy(false); }
  };
  const submitStop = async (values: { password: string }) => {
    setStopBusy(true);
    try {
      await apiClient.post('/api/admin/shutdown', { password: values.password });
      setStopped(true); setStopOpen(false); stopForm.resetFields();
      message.success('平台正在停止，可以关闭此页面。');
    } catch (e) { message.error(getUserErrorMessage(e)); }
    finally { setStopBusy(false); }
  };
  return <div className="workspace-page"><div className="page-heading admin-heading"><div><h2 className="page-title">用户与任务总览</h2><p className="page-description">管理员专属：查看账户、最近活动和当前实验任务，每 10 秒刷新。</p></div><Button icon={<ReloadOutlined/>} loading={loading} onClick={() => setRefresh(n => n + 1)}>刷新</Button></div>
    {error && <Alert type="error" showIcon title={error} style={{ marginBottom: 20 }}/>}
    <Card title="当前任务" style={{ marginBottom: 24 }}>{data.tasks.length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="当前没有正在执行的实验任务"/> : <Table<TaskRow> rowKey={row => row.kind + row.name} pagination={false} dataSource={data.tasks} scroll={{ x: 650 }} columns={[
      { title: '所属用户', dataIndex: 'uid', render: nameFor }, { title: '任务类型', dataIndex: 'kind' }, { title: '任务名称', dataIndex: 'name' }, { title: '当前实验', dataIndex: 'current_experiment', render: value => value || '—' }, { title: '状态', dataIndex: 'status', render: value => <Tag color="blue">{statuses[value] ?? value}</Tag> },
    ]}/>}</Card>
    <Card title="平台控制" style={{ marginBottom: 24 }}>
      <Alert type="warning" showIcon title="停止平台" description="停止后会终止当前实验与批量任务，并关闭后台服务。再次使用需要重新启动联安智盾。" style={{ marginBottom: 16 }}/>
      <Button danger type="primary" disabled={stopped} onClick={() => { stopForm.resetFields(); setStopOpen(true); }}>停止平台</Button>
      {stopped && <Tag color="red" style={{ marginLeft: 12 }}>已请求停止</Tag>}
    </Card>
    <Card title={`系统用户 · ${data.users.length}`}><Table<UserRow> rowKey="uid" loading={loading} dataSource={data.users} scroll={{ x: 850 }} pagination={{ pageSize: 10, showSizeChanger: false }} columns={[
      { title: '用户', dataIndex: 'username', render: (name, user) => <span><Avatar size="small" src={user.avatar || undefined} icon={<UserOutlined/>} style={{ marginRight: 8 }}/>{name}</span> },
      { title: '唯一 UID', dataIndex: 'uid', ellipsis: true, width: 230 },
      { title: '角色', dataIndex: 'role', render: value => <Tag color={value === 'admin' ? 'gold' : 'blue'}>{value === 'admin' ? '管理员' : '用户'}</Tag> },
      { title: '实验数', dataIndex: 'experiment_count' },
      { title: '最近活动', dataIndex: 'last_seen', render: value => value ? new Date(value * 1000).toLocaleString('zh-CN') : '尚未登录' },
      { title: '状态', dataIndex: 'disabled', render: value => <Tag color={value ? 'default' : 'green'}>{value ? '已停用' : '正常'}</Tag> },
      { title: '操作', render: (_, user) => user.role === 'admin' ? '—' : <><Button type="link" size="small" onClick={() => { setResetUser(user); resetForm.resetFields(); }}>重置密码</Button><Popconfirm title={user.disabled ? '启用此用户？' : '停用此用户并撤销其登录会话？'} description="已有实验结果会保留，正在运行的训练不会被强制终止。" onConfirm={async () => { try { await apiClient.put(`/api/admin/users/${user.uid}/disabled`, { disabled: !user.disabled }); setRefresh(n => n + 1); } catch (e) { message.error(getUserErrorMessage(e)); } }}><Button type="link" danger={!user.disabled} size="small">{user.disabled ? '启用' : '停用'}</Button></Popconfirm></> },
    ]}/></Card>
  <Modal title={`重置密码 · ${resetUser?.username ?? ''}`} open={Boolean(resetUser)} onCancel={() => { if (!resetBusy) { setResetUser(null); resetForm.resetFields(); } }} onOk={() => resetForm.submit()} okText="确认重置" confirmLoading={resetBusy} maskClosable={!resetBusy} width={420} destroyOnHidden>
    <Form form={resetForm} layout="vertical" onFinish={submitReset} requiredMark={false}>
      <Form.Item name="new_password" label="新密码" rules={passwordRules}><Input.Password autoComplete="new-password" placeholder="至少 12 位，包含两类字符"/></Form.Item>
      <Form.Item name="confirm" label="确认新密码" dependencies={['new_password']} rules={[{ required: true, message: '请确认新密码' }, ({ getFieldValue }) => ({ validator: (_, value) => value === getFieldValue('new_password') ? Promise.resolve() : Promise.reject(new Error('两次密码不一致')) })]}><Input.Password autoComplete="new-password" placeholder="再次输入新密码"/></Form.Item>
      <p style={{ color: '#8592a5', fontSize: 12, margin: 0 }}>重置后该用户需使用新密码登录，并会被要求再次修改密码。</p>
    </Form>
  </Modal>
  <Modal title="停止联安智盾" open={stopOpen} onCancel={() => { if (!stopBusy) { setStopOpen(false); stopForm.resetFields(); } }} onOk={() => stopForm.submit()} okText="确认停止" okButtonProps={{ danger: true }} confirmLoading={stopBusy} maskClosable={!stopBusy} width={440} destroyOnHidden>
    <Alert type="error" showIcon title="此操作会立即停止平台" description="正在运行的实验和批量任务会被终止；所有页面随后无法访问。" style={{ marginBottom: 16 }}/>
    <Form form={stopForm} layout="vertical" onFinish={submitStop} requiredMark={false}>
      <Form.Item name="password" label="管理员登录密码" rules={[{ required: true, message: '请输入管理员登录密码' }]}>
        <Input.Password autoComplete="current-password" placeholder="输入当前管理员的登录密码"/>
      </Form.Item>
      <p style={{ color: '#8592a5', fontSize: 12, margin: 0 }}>密码仅用于本地校验，不会保存。</p>
    </Form>
  </Modal>
  </div>;
}