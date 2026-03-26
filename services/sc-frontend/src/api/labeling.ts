import apiClient from './client';

export interface LabelingUploadResponse {
  session_id: string;
  minio_key: string;
  status: string;
}

export interface LabelingFrameResponse {
  frame_url: string;
  total_frames: number;
}

export interface LabelingStartResponse {
  status: string;
  frames_queued: number;
}

export async function uploadLabelingVideo(
  file: File,
  frameInterval = 2,
): Promise<LabelingUploadResponse> {
  const form = new FormData();
  form.append('video', file);
  const { data } = await apiClient.post<LabelingUploadResponse>(
    `/api/v1/labeling/upload?frame_interval=${frameInterval}`,
    form,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  );
  return data;
}

export async function getLabelingFrame(
  videoKey: string,
  frameNumber: number,
): Promise<LabelingFrameResponse> {
  const { data } = await apiClient.get<LabelingFrameResponse>(
    `/api/v1/labeling/frame`,
    { params: { video_key: videoKey, frame_number: frameNumber } },
  );
  return data;
}

export async function startLabeling(
  videoKey: string,
  jerseyOwnColorHsv: string | null,
  jerseyColorThreshold: number,
): Promise<LabelingStartResponse> {
  const { data } = await apiClient.post<LabelingStartResponse>(
    `/api/v1/labeling/start`,
    {
      video_key: videoKey,
      jersey_own_color_hsv: jerseyOwnColorHsv,
      jersey_color_threshold: jerseyColorThreshold,
    },
  );
  return data;
}
