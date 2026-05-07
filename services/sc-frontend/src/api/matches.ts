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

export async function createMatch(video: File, title: string): Promise<MatchCreateResponse> {
  const form = new FormData();
  form.append('video', video);
  form.append('title', title);
  const { data } = await apiClient.post<MatchCreateResponse>('/api/v1/matches', form);
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
