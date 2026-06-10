import apiClient from './client';

export interface RoiPoint {
  x: number;
  y: number;
}

interface MatchCreateResponse {
  match_id: string;
  status: string;
}

interface MatchConfigResponse {
  match_id: string;
  roi_polygon: RoiPoint[];
  start_seconds: number;
  end_seconds: number;
}

export interface MatchListItem {
  match_id: string;
  title: string;
  status: string;
  created_at: string;
  start_seconds: number | null;
  end_seconds: number | null;
  has_roi: boolean;
}

export async function getMatches(): Promise<MatchListItem[]> {
  const { data } = await apiClient.get<MatchListItem[]>('/api/v1/matches');
  return data;
}

export async function createMatch(
  video: File,
  title: string,
  onProgress?: (pct: number) => void,
): Promise<MatchCreateResponse> {
  const form = new FormData();
  form.append('video', video);
  form.append('title', title);
  const { data } = await apiClient.post<MatchCreateResponse>('/api/v1/matches', form, {
    onUploadProgress(e) {
      if (e.total && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
  return data;
}

export async function updateMatchConfig(
  matchId: string,
  roiPolygon: RoiPoint[],
  startSeconds: number,
  endSeconds: number,
): Promise<MatchConfigResponse> {
  const { data } = await apiClient.patch<MatchConfigResponse>(
    `/api/v1/matches/${matchId}/config`,
    { roi_polygon: roiPolygon, start_seconds: startSeconds, end_seconds: endSeconds },
  );
  return data;
}

export async function processMatch(matchId: string): Promise<{ match_id: string; status: string }> {
  const { data } = await apiClient.post<{ match_id: string; status: string }>(
    `/api/v1/matches/${matchId}/process`,
  );
  return data;
}

export interface MatchDetail {
  match_id: string;
  title: string;
  status: string;
  created_at: string;
  start_seconds: number | null;
  end_seconds: number | null;
  download_url: string | null;
}

export async function getMatch(matchId: string): Promise<MatchDetail> {
  const { data } = await apiClient.get<MatchDetail>(`/api/v1/matches/${matchId}`);
  return data;
}
