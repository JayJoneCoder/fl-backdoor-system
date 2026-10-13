// 默认使用同源代理；独立部署时可指定后端地址。
export const API_BASE = import.meta.env?.VITE_API_BASE_URL?.trim() ?? '';

export const getBackendUrl = (path: string, baseUrl: string = API_BASE): string =>
  `${baseUrl.replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`;

export const getWebSocketUrl = (
  path: string,
  baseUrl: string = API_BASE,
  pageUrl: string = window.location.href,
): string => {
  const url = new URL(getBackendUrl(path, baseUrl), pageUrl);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
};