import React from 'react';
import { Layout, Menu } from 'antd';
import {
  HistoryOutlined,
  SettingOutlined,
  AppstoreOutlined,
  FileTextOutlined,
  SafetyOutlined,
  GithubOutlined,
} from '@ant-design/icons';
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
    <Layout style={{ height: '100vh', width: '100%', overflow: 'hidden' }}>
      {/* 固定头部 */}
      <Header style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between', 
        padding: '0 24px',
        flexShrink: 0 
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <SafetyOutlined style={{ fontSize: 28, color: '#fff' }} />
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <span style={{ color: 'white', fontSize: 18, fontWeight: 600, lineHeight: 1.4 }}>
              联邦学习后门攻击防御系统
            </span>
            <span style={{ color: 'rgba(255,255,255,0.75)', fontSize: 12, lineHeight: 1.4 }}>
              FL Backdoor Attack & Defense System
            </span>
          </div>
        </div>

        <a
          href="https://github.com/JayJoneCoder/fl-backdoor-system"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'white', fontSize: 24, display: 'flex', alignItems: 'center' }}
        >
          <GithubOutlined />
        </a>
      </Header>

      {/* 主体区域：侧栏 + 内容 */}
      <Layout style={{ flex: 1, overflow: 'hidden' }}>
        {/* 固定侧栏 */}
        <Sider 
          width={200} 
          style={{ 
            background: '#fff', 
            height: '100%',
            overflowY: 'auto'  // 侧栏菜单过多时可独立滚动
          }}
        >
          <Menu
            mode="inline"
            selectedKeys={[currentPage]}
            onClick={({ key }) => setCurrentPage(key)}
            items={[
              { key: 'config', icon: <SettingOutlined />, label: '单个实验配置' },
              { key: 'batch', icon: <AppstoreOutlined />, label: '批量实验' },
              { key: 'history', icon: <HistoryOutlined />, label: '历史实验记录' },
              { key: 'summary', icon: <FileTextOutlined />, label: '实验比较总结' },
            ]}
          />
        </Sider>

        {/* 可滚动的内容区 */}
        <Layout style={{ padding: '24px', background: '#f0f2f5', overflowY: 'auto' }}>
          <Content style={{ 
            background: '#fff', 
            padding: 24, 
            margin: 0, 
            minHeight: 280,
            width: '100%' 
          }}>
            <style>{`
              .page-title {
                font-size: 24px;
                font-weight: 600;
                color: #1a1a1a;
                margin-top: 0;
                margin-bottom: 24px;
                padding-bottom: 12px;
                border-bottom: 1px solid #e8e8e8;
                display: flex;
                align-items: center;
              }
              .page-title::before {
                content: '';
                display: inline-block;
                width: 4px;
                height: 22px;
                background-color: #1890ff;
                margin-right: 12px;
                border-radius: 2px;
              }
            `}</style>
            {renderPage()}
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default App;