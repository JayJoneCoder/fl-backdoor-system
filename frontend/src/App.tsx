import React from 'react';
import { Layout, Menu } from 'antd';
import { HistoryOutlined, SettingOutlined, AppstoreOutlined, FileTextOutlined } from '@ant-design/icons';
import { useState } from 'react';
import ConfigPage from './pages/ConfigPage';
import HistoryPage from './pages/HistoryPage';
import BatchPage from './pages/BatchPage';
import SummaryPage from './pages/SummaryPage';

const { Header, Content, Sider } = Layout;

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState('config');

  const renderPage = () => {
    switch (currentPage) {
      case 'config': return <ConfigPage />;
      case 'history': return <HistoryPage />;
      case 'batch': return <BatchPage />;
      case 'summary': return <SummaryPage />;
      default: return <ConfigPage />;
    }
  };

  return (
    <Layout style={{ minHeight: '100vh', width: '100%' }}>
      <Header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px' }}>
        <h1 style={{ color: 'white', margin: 0 }}>FL Backdoor Platform</h1>
        <div style={{ color: 'white' }}>联邦学习后门攻防系统</div>
      </Header>
      <Layout style={{ width: '100%' }}>
        <Sider width={200} style={{ background: '#fff' }}>
          <Menu
            mode="inline"
            selectedKeys={[currentPage]}
            onClick={({ key }) => setCurrentPage(key)}
            items={[
              { key: 'config', icon: <SettingOutlined />, label: '单个实验配置' },
              { key: 'batch', icon: <AppstoreOutlined />, label: '批量实验' },
              { key: 'history', icon: <HistoryOutlined />, label: '历史实验记录' },
              { key: 'summary', icon: <FileTextOutlined />, label: '历史实验总结' },
            ]}
          />
        </Sider>
        <Layout style={{ padding: '24px', width: '100%', background: '#f0f2f5' }}>
          <Content style={{ background: '#fff', padding: 24, margin: 0, minHeight: 280, width: '100%' }}>
            {renderPage()}
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default App;