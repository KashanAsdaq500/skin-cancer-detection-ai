/**
 * Dermora AI - Frontend API Configuration & Utilities
 * Provides configurable base URL resolution with trailing slash protection
 * and fallback to http://localhost:8000 for local development.
 */

export function getApiBaseUrl(): string {
  const envUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (envUrl && typeof envUrl === "string" && envUrl.trim() !== "") {
    return envUrl.trim().replace(/\/+$/, "");
  }
  return "http://localhost:8000";
}

export function getApiUrl(path: string): string {
  const base = getApiBaseUrl();
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${base}${cleanPath}`;
}
