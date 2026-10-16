export type ApiFailureKind = 'connection' | 'validation' | 'backend' | 'response'

export class ApiFailure extends Error {
  readonly kind: ApiFailureKind

  constructor(kind: ApiFailureKind, message: string) {
    super(message)
    this.name = 'ApiFailure'
    this.kind = kind
  }
}

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || '/api'
export const apiBaseUrl = configuredBaseUrl.replace(/\/$/, '')

export class ApiClient {
  private readonly baseUrl: string

  constructor(baseUrl = apiBaseUrl) {
    this.baseUrl = baseUrl
  }

  async get(path: string, signal?: AbortSignal): Promise<unknown> {
    return this.request(path, { signal })
  }

  async post(path: string, body: unknown, signal?: AbortSignal): Promise<unknown> {
    return this.request(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
  }

  private async request(path: string, options: RequestInit): Promise<unknown> {
    let response: Response
    try {
      response = await fetch(`${this.baseUrl}${path}`, options)
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') throw error
      throw new ApiFailure('connection', 'No se pudo conectar con el servicio. Comprueba que esté disponible e inténtalo de nuevo.')
    }

    if (!response.ok) {
      if (response.status === 422) {
        throw new ApiFailure('validation', 'El servicio rechazó el texto. Revisa el título y el contenido.')
      }
      if ([502, 503, 504].includes(response.status) || !response.headers.get('Content-Type')?.includes('application/json')) {
        throw new ApiFailure('connection', 'No se pudo conectar con el servicio. Comprueba que esté disponible e inténtalo de nuevo.')
      }
      throw new ApiFailure('backend', 'El servicio no pudo completar el análisis. Inténtalo de nuevo en unos momentos.')
    }

    try {
      return await response.json() as unknown
    } catch {
      throw new ApiFailure('response', 'El servicio devolvió una respuesta que no se pudo leer.')
    }
  }
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
