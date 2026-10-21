import { useState } from 'react';
import { Avatar, Button, Descriptions, Dropdown, Form, Input, Modal, Tabs, message } from 'antd';
import { DownOutlined, LogoutOutlined, SafetyOutlined, UserOutlined } from '@ant-design/icons';
import type { PlatformUser } from '../../api/auth';
import { changeAvatar, changePassword, logout } from '../../api/auth';
import { getUserErrorMessage } from '../../api/checkedFetch';
import { passwordRules } from '../../pages/LoginPage';

interface Props { user: PlatformUser; onUpdate: (user: PlatformUser) => void; onLogout: () => void }

export default function UserMenu({ user, onUpdate, onLogout }: Props) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [form] = Form.useForm();
  const signOut = async () => {
    setBusy(true);
    try { await logout(); onLogout(); } catch (e) { message.error(getUserErrorMessage(e)); }
    finally { setBusy(false); }
  };
  const uploadAvatar = async (file?: File) => {
    if (!file) return;
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || file.size > 256 * 1024) {
      message.error('请选择不超过 256 KB 的 PNG、JPEG 或 WebP 图片。'); return;
    }
    setBusy(true);
    try {
      const value = await new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = reject; reader.readAsDataURL(file); });
      await changeAvatar(value); onUpdate({ ...user, avatar: value }); message.success('头像已更新');
    } catch (e) { message.error(getUserErrorMessage(e)); }
    finally { setBusy(false); }
  };
  return <>
    <Dropdown trigger={['click']} menu={{ items: [
      { key: 'profile', icon: <UserOutlined/>, label: '个人资料与头像', onClick: () => setOpen(true) },
      { type: 'divider' }, { key: 'logout', icon: <LogoutOutlined/>, label: '安全退出', disabled: busy, onClick: () => { void signOut(); } },
    ] }}><button type="button" className="platform-user" aria-label={`用户菜单：${user.username}`}><Avatar size={34} src={user.avatar || undefined} icon={<UserOutlined/>}/><span><strong>{user.username}</strong><small>{user.role === 'admin' ? '管理员' : '研究用户'}</small></span><DownOutlined/></button></Dropdown>
    <Modal title="账户设置" open={open} onCancel={() => { if (!busy) { setOpen(false); form.resetFields(); } }} footer={null} width={520} destroyOnHidden>
      <Tabs items={[{ key: 'profile', label: '个人资料', children: <div className="profile-settings"><div className="profile-avatar"><Avatar size={72} src={user.avatar || undefined} icon={<UserOutlined/>}/><div><label className="avatar-upload">{busy ? '正在更新…' : '更换头像'}<input type="file" accept="image/png,image/jpeg,image/webp" disabled={busy} onChange={event => { void uploadAvatar(event.target.files?.[0]); event.target.value = ''; }}/></label><p>PNG / JPEG / WebP，最大 256 KB</p><Button size="small" type="text" disabled={busy || !user.avatar} onClick={async () => { setBusy(true); try { await changeAvatar(''); onUpdate({ ...user, avatar: '' }); } catch (e) { message.error(getUserErrorMessage(e)); } finally { setBusy(false); } }}>恢复默认头像</Button></div></div><Descriptions column={1} items={[{ key: 'name', label: '用户名', children: user.username }, { key: 'uid', label: '唯一 UID', children: <span style={{ overflowWrap: 'anywhere' }}>{user.uid}</span> }, { key: 'role', label: '账户角色', children: <span><SafetyOutlined/> {user.role === 'admin' ? '管理员 · 可查询全部实验' : '用户 · 仅可访问自己的实验'}</span> }]}/></div> }, { key: 'security', label: '修改密码', children: <Form form={form} layout="vertical" onFinish={async values => { setBusy(true); try { await changePassword(values.current_password, values.new_password); message.success('密码已更新，请重新登录'); onLogout(); } catch (e) { message.error(getUserErrorMessage(e)); } finally { setBusy(false); } }}><Form.Item name="current_password" label="当前密码" rules={[{ required: true, message: '请输入当前密码' }]}><Input.Password autoComplete="current-password"/></Form.Item><Form.Item name="new_password" label="新密码" rules={passwordRules}><Input.Password autoComplete="new-password"/></Form.Item><Form.Item name="confirm" label="确认新密码" dependencies={['new_password']} rules={[{ required: true, message: '请确认新密码' }, ({ getFieldValue }) => ({ validator: (_, value) => value === getFieldValue('new_password') ? Promise.resolve() : Promise.reject(new Error('两次密码不一致')) })]}><Input.Password autoComplete="new-password"/></Form.Item><p className="profile-note">修改密码后，所有设备上的会话将退出。</p><Button type="primary" htmlType="submit" loading={busy}>更新密码并重新登录</Button></Form> }]}/>
    </Modal>
  </>;
}