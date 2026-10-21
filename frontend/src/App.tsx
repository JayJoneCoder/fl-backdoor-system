import React from 'react';
import { Alert, Button, Layout, Menu, Spin } from 'antd';
import {
  HistoryOutlined,
  SettingOutlined,
  AppstoreOutlined,
  FileTextOutlined,
  HomeOutlined,
  TeamOutlined,
} from '@ant-design/icons';
import { useEffect, useState } from 'react';
import ConfigPage from './pages/ConfigPage';
import HistoryPage from './pages/HistoryPage';
import BatchPage from './pages/BatchPage';
import SummaryPage from './pages/SummaryPage';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import AdminPage from './pages/AdminPage';
import UserMenu from './components/UserMenu';
import { getSession } from './api/auth';
import type { PlatformUser, SessionResponse } from './api/auth';
import { setCsrfToken } from './api/session';
import './App.css';

const { Header, Content } = Layout;

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState('config');
  const [route, setRoute] = useState(() => window.location.hash === '#/workspace' ? 'workspace' : window.location.hash === '#/login' ? 'login' : 'home');
  const [user, setUser] = useState<PlatformUser | null>(null);
  const [checking, setChecking] = useState(true);
  const [sessionError, setSessionError] = useState(false);
  const [visited, setVisited] = useState(false);
  const [retry, setRetry] = useState(0);
  const navigate = (next: string) => { window.location.hash = next === 'home' ? '/' : `/${next}`; setRoute(next); };
  useEffect(() => {
    const update = () => setRoute(window.location.hash === '#/workspace' ? 'workspace' : window.location.hash === '#/login' ? 'login' : 'home');
    const expired = () => { setUser(null); setCsrfToken(''); setVisited(false); setCurrentPage('config'); navigate('login'); };
    window.addEventListener('hashchange', update);
    window.addEventListener('lianan:session-expired', expired);
    return () => { window.removeEventListener('hashchange', update); window.removeEventListener('lianan:session-expired', expired); };
  }, []);
  useEffect(() => {
    let active = true;
    getSession().then(({ data }) => { if (active) { setUser(data.user); setCsrfToken(data.csrf_token); setSessionError(false); } })
      .catch(error => { if (active) { setUser(null); setSessionError(error.response?.status !== 401); } })
      .finally(() => { if (active) setChecking(false); });
    return () => { active = false; };
  }, [retry]);
  useEffect(() => { if (route === 'workspace' && user) setVisited(true); }, [route, user]);
  const authenticated = (session: SessionResponse) => { setUser(session.user); setCsrfToken(session.csrf_token); setSessionError(false); setCurrentPage('config'); navigate('workspace'); };
  const enter = () => navigate(user ? 'workspace' : 'login');

  const renderPage = () => {
    switch (currentPage) {
      case 'config': return <ConfigPage />;
      case 'history': return <HistoryPage />;
      case 'batch': return <BatchPage />;
      case 'summary': return <SummaryPage />;
      case 'admin': return user?.role === 'admin' ? <AdminPage /> : <ConfigPage />;
      default: return <ConfigPage />;
    }
  };

  return (
    <>
    {route === 'home' && <LandingPage onEnter={enter} authenticated={Boolean(user)} />}
    {route !== 'home' && checking && <div className="session-loading"><Spin size="large"/><p>正在恢复安全会话…</p></div>}
    {route !== 'home' && !checking && sessionError && <div className="session-loading"><Alert type="error" showIcon title="无法连接身份服务" description="请确认后端服务已启动，现有实验数据不会受到影响。"/><Button onClick={() => { setChecking(true); setRetry(n => n + 1); }}>重新连接</Button><Button type="link" onClick={() => navigate('home')}>返回首页</Button></div>}
    {route !== 'home' && !checking && !sessionError && (!user || route === 'login') && <LoginPage onHome={() => navigate('home')} onAuthenticated={authenticated}/>}
    {user && (visited || route === 'workspace') && <div hidden={route !== 'workspace'} className="workspace-host" key={user.uid}>
    <Layout className="platform-shell">
      {/* 固定头部 */}
      <Header className="platform-header">
        <div className="platform-brand">
          <img className="platform-brand__mark" src="/favicon.svg" alt="" width={48} height={48} />
          <div className="platform-brand__text">
            <span className="platform-brand__name">
              联安<span className="platform-brand__accent">智盾</span>
            </span>
            <span className="platform-brand__description">
              联邦学习多层协同安全防护与可视化验证平台
            </span>
          </div>
        </div>

        <div className="platform-header__actions">
          <Button type="text" icon={<HomeOutlined/>} onClick={() => navigate('home')} className="platform-home">返回首页</Button>
          <UserMenu user={user} onUpdate={setUser} onLogout={() => { setUser(null); setCsrfToken(''); setVisited(false); setCurrentPage('config'); navigate('login'); }}/>
        </div>
      </Header>

      {/* 主体区域：侧栏 + 内容 */}
      <div className="platform-workspace">
        {/* 固定侧栏 */}
        <aside className="platform-sidebar" aria-label="主导航">
          <div className="platform-sidebar__label">实验工作台</div>
          <Menu
            className="platform-navigation"
            mode="inline"
            selectedKeys={[currentPage]}
            onClick={({ key }) => setCurrentPage(key)}
            items={[
              { key: 'config', icon: <SettingOutlined />, label: '单个实验配置' },
              { key: 'batch', icon: <AppstoreOutlined />, label: '批量实验' },
              { key: 'history', icon: <HistoryOutlined />, label: '历史实验记录' },
              { key: 'summary', icon: <FileTextOutlined />, label: '实验比较总结' },
              ...(user.role === 'admin' ? [{ key: 'admin', icon: <TeamOutlined/>, label: '用户与任务总览' }] : []),
            ]}
          />
          <div className="platform-sidebar__footer">
            <span>联安智盾</span>
            <span>联邦学习安全验证</span>
          </div>
        </aside>

        {/* 可滚动的内容区 */}
        <main className="platform-main" aria-label="实验工作区" id="main-content" key={currentPage}>
          <Content className="platform-content">
            {renderPage()}
          </Content>
        </main>
      </div>
    </Layout>
    </div>}
    </>
  );
};

export default App;