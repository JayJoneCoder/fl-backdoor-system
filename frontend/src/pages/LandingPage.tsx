import { Button } from 'antd';
import {
  ArrowRightOutlined, SafetyCertificateOutlined,
  FilterOutlined, RadarChartOutlined, ClusterOutlined, CheckOutlined,
} from '@ant-design/icons';
import './LandingPage.css';

interface Props { onEnter: () => void; authenticated: boolean }

const capabilities = [
  { number: '01', icon: <FilterOutlined />, layer: '数据层 · 客户端', title: '客户端样本特征过滤',
    text: '在本地训练前分析样本特征，识别并过滤可疑数据，从源头减少后门样本对联邦训练的影响。', tags: ['特征分析', '可疑样本过滤'] },
  { number: '02', icon: <RadarChartOutlined />, layer: '更新层 · 服务端', title: '服务端异常更新检测',
    text: '结合更新范数、方向相似度与聚类分析，发现异常客户端更新，验证不同检测策略的防护效果。', tags: ['多策略检测', '客户端级评估'] },
  { number: '03', icon: <SafetyCertificateOutlined />, layer: '模型层 · 聚合端', title: '鲁棒聚合与模型保护',
    text: '通过范数裁剪、修剪平均与 Krum 等聚合策略抑制恶意更新，兼顾全局模型的性能与安全。', tags: ['鲁棒聚合', '全局模型保护'] },
];

function DefenseVisual() {
  return <div className="defense-visual" aria-label="客户端、异常检测与安全聚合的三层协同防护示意图">
    <div className="defense-visual__heading"><span>联邦安全协同架构</span><span className="visual-caption">三层协同 / 全程验证</span></div>
    <svg className="defense-network" viewBox="0 0 520 300" fill="none" aria-hidden="true">
      <defs><linearGradient id="secure-line"><stop stopColor="#3681db"/><stop offset="1" stopColor="#59d0c0"/></linearGradient></defs>
      <circle cx="260" cy="150" r="121" stroke="#244362" strokeDasharray="3 8"/>
      <circle cx="260" cy="150" r="86" stroke="#254e70"/>
      {[ [70,65], [70,235], [450,65], [450,235], [260,27], [260,273] ].map(([x,y], i) =>
        <g key={i}><path d={`M${x} ${y} L260 150`} stroke="url(#secure-line)" strokeWidth="1.5"/>
          <rect x={x-23} y={y-19} width="46" height="38" rx="10" fill="#122c48" stroke="#3e7599"/>
          <rect x={x-9} y={y-8} width="18" height="13" rx="3" stroke="#91c4dd"/>
          <path d={`M${x-12} ${y+10}h24`} stroke="#91c4dd"/>
          <circle cx={x+18} cy={y-15} r="4" fill="#63cebd" stroke="#112841" strokeWidth="2"/>
        </g>)}
      <path d="M260 96l47 18v38c0 31-24 47-47 57-23-10-47-26-47-57v-38l47-18z" fill="#123951" stroke="#65d6ca" strokeWidth="2"/>
      <path d="M241 151l13 13 27-30" stroke="#8ae1d6" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
    <div className="defense-visual__layers">
      {['样本过滤', '异常检测', '安全聚合'].map((item, i) => <div key={item}><span>0{i+1}</span><strong>{item}</strong><CheckOutlined /></div>)}
    </div>
    <div className="defense-visual__foot"><span className="visual-dot"/> 以实验指标验证每一层防护 <span>ACC / ASR / FPR</span></div>
  </div>;
}

export default function LandingPage({ onEnter, authenticated }: Props) {
  return <div className="landing-page">
    <a className="landing-skip" href="#landing-main">跳转到主要内容</a>
    <header className="landing-nav">
      <a href="#/" className="landing-logo" aria-label="联安智盾首页"><img src="/favicon.svg" alt="" width={38} height={38}/><span>联安智盾</span><span className="landing-logo__divider"/><small>联邦学习安全验证平台</small></a>
      <nav aria-label="首页导航"><Button onClick={onEnter}>进入系统 <ArrowRightOutlined/></Button></nav>
    </header>
    <main id="landing-main">
      <section className="landing-hero">
        <div className="landing-container landing-hero__grid">
          <div className="landing-hero__copy">
            <div className="landing-eyebrow"><span/> 联邦智能 · 协同防护 · 可信验证</div>
            <h1>联安<span>智盾</span></h1>
            <h2>面向联邦学习的多层协同安全防护<br className="hero-break"/>与可视化验证平台</h2>
            <p className="landing-intro">让联邦训练中的安全风险可发现，让协同防护的效果可验证。<br/>贯通客户端、服务端检测与模型聚合，以可复现的实验和可视化指标，支撑联邦学习安全研究。</p>
            <div className="landing-actions"><Button type="primary" size="large" onClick={onEnter}>{authenticated ? '进入实验工作台' : '登录系统'} <ArrowRightOutlined/></Button><a href="#core-capabilities" onClick={e => { e.preventDefault(); document.getElementById('core-capabilities')?.scrollIntoView({ behavior: 'smooth' }); }}>了解核心功能 <span aria-hidden="true">↓</span></a></div>
            <div className="hero-assurances"><span><CheckOutlined/> 多层协同防护</span><span><CheckOutlined/> 全流程实验追溯</span><span><CheckOutlined/> 实时可视化验证</span></div>
          </div>
          <DefenseVisual/>
        </div>
        <div className="landing-container landing-workflow"><span>构建完整的安全验证闭环</span><div>{['实验配置', '联邦训练', '协同防护', '可视化评估'].map((item, i) => <span key={item}>{i > 0 && <ArrowRightOutlined/>}<strong>{item}</strong></span>)}</div></div>
      </section>
      <section id="core-capabilities" className="landing-capabilities landing-container">
        <div className="landing-section-heading"><div><span className="section-kicker">核心防护体系</span><h2>三层协同，守护联邦训练全链路</h2></div><p>从本地样本到全局模型，<br/>将防御机制与实验验证紧密连接。</p></div>
        <div className="capability-grid">{capabilities.map(item => <article className="capability-card" key={item.number}><div className="capability-card__top"><span className="capability-icon">{item.icon}</span><span>{item.number}</span></div><span className="capability-layer">{item.layer}</span><h3>{item.title}</h3><p>{item.text}</p><div className="capability-tags">{item.tags.map(tag => <span key={tag}>{tag}</span>)}</div></article>)}</div>
      </section>
      <section className="landing-tools landing-container"><div><span className="section-kicker">研究与工程一体化</span><h2>从安全场景，到可追溯的实验结论</h2><p>保留熟悉的实验工作流，将配置、监控、结果与对比统一在一个平台。</p></div><div className="landing-tool-grid">{[
        ['IID / Non-IID 数据划分', '验证不同数据分布下的防护表现'], ['典型安全场景验证', '支持多种后门攻击与防御策略对照'], ['实时训练监控', '联动查看训练日志与逐轮指标'], ['批量实验与结果追溯', '保留配置快照、指标与实验附件'],
      ].map(([name, text]) => <div key={name}><ClusterOutlined/><div><h3>{name}</h3><p>{text}</p></div></div>)}</div></section>
      <section className="landing-cta landing-container"><div><h2>让每一次防护，都有实验依据。</h2><p>进入联安智盾，开启联邦学习安全验证。</p></div><Button type="primary" size="large" onClick={onEnter}>进入系统 <ArrowRightOutlined/></Button></section>
    </main>
    <footer className="landing-footer landing-container"><span>联安智盾 <span className="footer-muted"> / 联邦学习安全研究与验证</span></span><span>基于 Flower · PyTorch · FastAPI · React 构建</span></footer>
  </div>;
}