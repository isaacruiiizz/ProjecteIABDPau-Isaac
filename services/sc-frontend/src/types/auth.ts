export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface AuthState {
  token: string | null;
  setToken: (token: string) => void;
  clearToken: () => void;
}
