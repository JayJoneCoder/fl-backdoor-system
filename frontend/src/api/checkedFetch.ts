export class HttpRequestError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'HttpRequestError';
    this.status = status;
  }
}

// fetch rejects network failures, not HTTP errors. Preserve explicit conflict
// handling while requiring every other response to be successful.
export const checkedFetch = async (
  input: RequestInfo | URL,
  init?: RequestInit,
  allowedStatuses: number[] = [],
): Promise<Response> => {
  const response = await fetch(input, init);
  if (!response.ok && !allowedStatuses.includes(response.status)) {
    let message = `请求失败（HTTP ${response.status}）`;
    try {
      const data: unknown = await response.json();
      if (data && typeof data === 'object' && 'detail' in data) {
        const detail = (data as { detail: unknown }).detail;
        if (typeof detail === 'string') message = detail;
      }
    } catch {
      // A non-JSON error response is still an HTTP failure.
    }
    throw new HttpRequestError(response.status, message);
  }
  return response;
};

export const getConfigErrorMessage = (error: unknown): string => {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof response?.data?.detail === 'string') return response.data.detail;
  }
  return error instanceof Error ? error.message : '配置操作失败';
};