import { apiClient } from './client';

export interface PlatformUser {
  uid: string;
  username: string;
  role: 'admin' | 'user';
  avatar: string;
  created_at: number;
  last_seen: number | null;
  disabled: number;
  must_change_password: number;
}

export interface SessionResponse { user: PlatformUser; csrf_token: string }
export interface AuthStatus { admin_must_change_password: boolean; default_admin_password: string }
export const getAuthStatus = () => apiClient.get<AuthStatus>('/api/auth/status');
export const getSession = () => apiClient.get<SessionResponse>('/api/auth/me');
export const login = (username: string, password: string, remember = false) =>
  apiClient.post<SessionResponse>('/api/auth/login', { username, password, remember });
export const register = (username: string, password: string) => apiClient.post('/api/auth/register', { username, password });
export const forceResetPassword = (new_password: string) => apiClient.post<SessionResponse>('/api/auth/force-reset', { new_password });
export const forgotPassword = (username: string) => apiClient.post<{ admin_reset: boolean; message: string }>('/api/auth/forgot', { username });
export const logout = () => apiClient.post('/api/auth/logout');
export const changePassword = (current_password: string, new_password: string) => apiClient.post('/api/auth/password', { current_password, new_password });
export const changeAvatar = (avatar: string) => apiClient.put('/api/auth/avatar', { avatar });