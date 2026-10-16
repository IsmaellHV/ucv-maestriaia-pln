import type { ServiceInfo } from '../../domain/entities/ServiceInfo'
import type { ServiceInfoRepository } from '../../application/ports/ServiceInfoRepository'
import { ApiClient, ApiFailure, isRecord } from '../api/ApiClient'

export class FastApiServiceInfoRepository implements ServiceInfoRepository {
  private readonly api: ApiClient

  constructor(api: ApiClient) {
    this.api = api
  }

  async load(signal?: AbortSignal): Promise<ServiceInfo> {
    const health = await this.api.get('/health', signal)
    if (!isRecord(health) || health.status !== 'ok' || health.model_loaded !== true) {
      throw new ApiFailure('backend', 'El modelo todavía no está disponible.')
    }

    const model = await this.api.get('/model-info', signal)
    if (!isRecord(model) || typeof model.model_name !== 'string' ||
        typeof model.model_version !== 'string' || typeof model.max_length !== 'number') {
      throw new ApiFailure('response', 'La información del modelo está incompleta.')
    }

    return {
      modelName: model.model_name,
      modelVersion: model.model_version,
      maxLength: model.max_length,
    }
  }
}
