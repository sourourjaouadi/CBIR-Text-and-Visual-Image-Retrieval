const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api';

export type Descriptor = { name: string; weight: number };
export type Result = {
  image_id: number;
  url: string;
  distance: number;
  rank: number;
  keywords?: string[];
  descriptor_distances?: Record<string, number | null>;
};
export type ImageInfo = { image_id: number; url: string; label: string | null; keywords: string[] };
export type SearchResponse = { query: Record<string, unknown>; query_image_url: string | null; results: Result[]; took_ms: number };
export type SearchPayload = {
  dataset: string;
  keywords: string[];
  keyword_mode: string;
  query_image_id: number | null;
  descriptors: Descriptor[];
  k: number;
};

async function checked<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = await response.json();
      message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    } catch { message = await response.clone().text(); }
    throw new Error(message || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export async function search(payload: SearchPayload): Promise<SearchResponse> {
  return checked(await fetch(`${API}/search`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }));
}

export async function searchUpload(payload: SearchPayload, image: File): Promise<SearchResponse> {
  const form = new FormData();
  form.append('image', image);
  form.append('dataset', payload.dataset);
  form.append('keywords', JSON.stringify(payload.keywords));
  form.append('keyword_mode', payload.keyword_mode);
  form.append('descriptors', JSON.stringify(payload.descriptors));
  form.append('k', String(payload.k));
  return checked(await fetch(`${API}/search/upload`, { method: 'POST', body: form }));
}

export async function descriptors(dataset: string) {
  return checked<{ descriptors: { name: string; metric: string; kind?: string }[] }>(
    await fetch(`${API}/descriptors?dataset=${encodeURIComponent(dataset)}`));
}

export async function getImage(dataset: string, imageId: number): Promise<ImageInfo> {
  return checked(await fetch(`${API}/images/${imageId}?dataset=${encodeURIComponent(dataset)}`));
}
