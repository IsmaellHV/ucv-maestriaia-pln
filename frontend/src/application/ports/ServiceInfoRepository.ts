import type { ServiceInfo } from '../../domain/entities/ServiceInfo'

export interface ServiceInfoRepository {
  load(signal?: AbortSignal): Promise<ServiceInfo>
}
