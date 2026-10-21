import { useEffect, useState } from 'react';
import { Alert, Button, Checkbox, Form, Input, Spin } from 'antd';
import { ArrowLeftOutlined, LockOutlined, SafetyCertificateOutlined, UserOutlined } from '@ant-design/icons';
import { forceResetPassword, forgotPassword, getAuthStatus, login, register } from '../api/auth';
import type { SessionResponse } from '../api/auth';
import { getUserErrorMessage } from '../api/checkedFetch';
import { setCsrfToken } from '../api/session';
import './LoginPage.css';

export const passwordRules = [
  { required: true, message: '请设置密码' },
  { min: 12, max: 128, message: '密码须为 12–128 个字符' },
  { validator: (_: unknown, value: string) => !value || [/[A-Za-z]/, /\d/, /[^A-Za-z\d]/].filter(rule => rule.test(value)).length >= 2
    ? Promise.resolve() : Promise.reject(new Error('请至少包含字母、数字、符号中的两类')) },
];

interface Props { onHome: () => void; onAuthenticated: (session: SessionResponse) => void }
interface Values { username: string; password: string; confirm: string; rememberUsername: boolean; rememberMe: boolean }

export default function LoginPage({ onHome, onAuthenticated }: Props) {
  const [form] = Form.useForm<Values>();
  const [mode, setMode] = useState<'login' | 'register' | 'forgot' | 'reset'>('login');
  const [checking, setChecking] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [status, setStatus] = useState<{ admin_must_change_password: boolean; default_admin_password: string } | null>(null);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let active = true;
    getAuthStatus().then(({ data }) => { if (active) { setStatus(data); setError(''); } })
      .catch(e => { if (active) setError(getUserErrorMessage(e, '无法读取登录状态，请确认后端已启动。')); })
      .finally(() => { if (active) setChecking(false); });
    return () => { active = false; };
  }, [retry]);
  useEffect(() => {
    const saved = localStorage.getItem('lianan:remembered_username');
    if (saved) { form.setFieldValue('username', saved); form.setFieldValue('rememberUsername', true); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const switchMode = (next: 'login' | 'register' | 'forgot' | 'reset') => { setMode(next); setError(''); setNotice(''); form.resetFields(['password', 'confirm']); };

  const submit = async (values: Values) => {
    setLoading(true); setError(''); setNotice('');
    try {
      if (mode === 'register') {
        await register(values.username, values.password);
        form.resetFields(['password', 'confirm']); setMode('login'); form.setFieldValue('username', values.username);
        setNotice('账户已创建，请登录。新实验将自动归属你的账户。');
      } else if (mode === 'forgot') {
        const { data } = await forgotPassword(values.username);
        setNotice(data.message);
      } else if (mode === 'reset') {
        const { data } = await forceResetPassword(values.password);
        onAuthenticated(data);
      } else {
        const { data } = await login(values.username, values.password, Boolean(values.rememberMe));
        if (values.rememberUsername) localStorage.setItem('lianan:remembered_username', values.username);
        else localStorage.removeItem('lianan:remembered_username');
        if (data.user.must_change_password) {
          setCsrfToken(data.csrf_token);
          form.resetFields(['password', 'confirm']);
          setMode('reset');
          setNotice('首次登录需修改默认密码后才能进入系统。');
        } else {
          onAuthenticated(data);
        }
      }
    } catch (e) { setError(getUserErrorMessage(e)); }
    finally { setLoading(false); }
  };

  const title = { login: '登录系统', register: '创建研究账户', forgot: '找回密码', reset: '设置新密码' }[mode];
  const intro = { login: '登录你的账户，开始联邦学习安全实验。', register: '创建你的专属实验空间，开始安全验证。', forgot: '输入用户名发起找回。管理员密码将被重置为默认密码，普通用户请联系管理员。', reset: '首次登录需设置新密码后方可进入系统。' }[mode];
  const submitLabel = { login: '登录系统', register: '创建账户', forgot: '找回密码', reset: '确认并进入系统' }[mode];

  return <main className="auth-page">
    <section className="auth-story"><a href="#/" className="landing-logo"><img src="/favicon.svg" alt="" width={40} height={40}/><span>联安智盾</span></a><div className="auth-story__copy"><span className="auth-kicker">可信身份 · 安全实验</span><h1>以协同防护，<br/>连接可信的联邦智能。</h1><p>面向联邦学习的多层协同安全防护与可视化验证平台</p><div className="auth-story__features"><span><SafetyCertificateOutlined/> 数据、更新与模型三层协同</span><span><LockOutlined/> 独立账户与实验结果权限隔离</span></div></div></section>
    <section className="auth-panel"><Button type="text" icon={<ArrowLeftOutlined/>} onClick={onHome} className="auth-back">返回首页</Button><div className="auth-form-wrap"><span className="auth-kicker">欢迎使用联安智盾</span><h2>{title}</h2><p className="auth-form-intro">{intro}</p>
      {checking ? <Spin tip="正在连接身份服务…"><div style={{ minHeight: 160 }}/></Spin> : <>
        {error && <Alert type="error" showIcon title={error} style={{ marginBottom: 18 }}/>}
        {notice && <Alert type="success" showIcon title={notice} style={{ marginBottom: 18 }}/>}
        {status?.admin_must_change_password && <Alert type="info" showIcon title="首次部署" description={`管理员账户 admin 已自动创建，初始密码为 ${status.default_admin_password}。首次登录后需立即修改密码。`} style={{ marginBottom: 22 }}/>}
        <Form form={form} layout="vertical" onFinish={submit} requiredMark={false} disabled={loading} size="large">
          {mode !== 'reset' && <Form.Item name="username" label="用户名" rules={[{ required: true, message: '请输入用户名' }, ...(mode === 'register' ? [{ pattern: /^[A-Za-z0-9_\-\u3400-\u9fff]{3,32}$/, message: '须为 3–32 位中文、字母、数字、下划线或短横线' }] : [])]}><Input prefix={<UserOutlined/>} autoComplete="username" placeholder="输入用户名" maxLength={32}/></Form.Item>}
          {mode !== 'forgot' && <Form.Item name="password" label={mode === 'login' ? '密码' : '设置密码'} rules={mode === 'login' ? [{ required: true, message: '请输入密码' }] : passwordRules}><Input.Password prefix={<LockOutlined/>} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} maxLength={128} placeholder={mode === 'login' ? '输入密码' : '至少 12 位，包含两类字符'}/></Form.Item>}
          {(mode === 'register' || mode === 'reset') && <Form.Item name="confirm" label="确认密码" dependencies={['password']} rules={[{ required: true, message: '请再次输入密码' }, ({ getFieldValue }) => ({ validator: (_, value) => !value || value === getFieldValue('password') ? Promise.resolve() : Promise.reject(new Error('两次输入的密码不一致')) })]}><Input.Password autoComplete="new-password" placeholder="再次输入密码"/></Form.Item>}
          {mode === 'login' && <div className="auth-options"><Form.Item name="rememberUsername" valuePropName="checked" noStyle><Checkbox>记住用户名</Checkbox></Form.Item><Form.Item name="rememberMe" valuePropName="checked" noStyle><Checkbox>记住我</Checkbox></Form.Item><Button type="link" className="auth-forgot" disabled={loading} onClick={() => switchMode('forgot')}>忘记密码？</Button></div>}
          <Button type="primary" htmlType="submit" block loading={loading}>{submitLabel}</Button>
        </Form>
        {mode === 'login' && <div className="auth-switch">还没有账户？<Button type="link" disabled={loading} onClick={() => switchMode('register')}>创建账户</Button></div>}
        {mode === 'register' && <div className="auth-switch">已经有账户？<Button type="link" disabled={loading} onClick={() => switchMode('login')}>返回登录</Button></div>}
        {mode === 'forgot' && <div className="auth-switch">想起密码了？<Button type="link" disabled={loading} onClick={() => switchMode('login')}>返回登录</Button></div>}
        {error && <Button type="link" onClick={() => { setChecking(true); setRetry(v => v + 1); }}>重新连接身份服务</Button>}
        <div className="auth-security-note"><LockOutlined/> 会话受保护 · 普通用户仅可访问自己的实验结果</div>
      </>}
    </div><p className="auth-panel__footer">联安智盾 · 联邦学习安全研究与验证</p></section>
  </main>;
}