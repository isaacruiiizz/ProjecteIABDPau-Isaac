import apiClient from './client';
import type { LoginRequest, TokenResponse } from '../types/auth';

export async function login(email: string, password: string): Promise<TokenResponse> {
  const body: LoginRequest = { email, password };
  const { data } = await apiClient.post<TokenResponse>('/auth/login', body);
  return data;
}

export async function refreshToken(): Promise<TokenResponse> {
  const { data } = await apiClient.post<TokenResponse>('/auth/refresh');
  return data;
}
