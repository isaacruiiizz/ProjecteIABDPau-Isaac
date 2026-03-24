import apiClient from './client';

export interface LabelingUploadResponse {
  session_id: string;
  minio_key: string;
  status: string;
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
