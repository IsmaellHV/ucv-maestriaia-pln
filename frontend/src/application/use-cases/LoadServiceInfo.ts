import type { ServiceInfo } from '../../domain/entities/ServiceInfo'
import type { ServiceInfoRepository } from '../ports/ServiceInfoRepository'

export class LoadServiceInfo {
  private readonly repository: ServiceInfoRepository

  constructor(repository: ServiceInfoRepository) {
    this.repository = repository
  }

  execute(signal?: AbortSignal): Promise<ServiceInfo> {
    return this.repository.load(signal)
  }
}
