import type { NewsArticle } from '../../domain/entities/NewsArticle'
import type { Prediction } from '../../domain/entities/Prediction'
import type { PredictionRepository } from '../../domain/repositories/PredictionRepository'
import { ApiClient, ApiFailure, isRecord } from '../api/ApiClient'

export class FastApiPredictionRepository implements PredictionRepository {
  private readonly api: ApiClient

  constructor(api: ApiClient) {
    this.api = api
  }

  async predict(article: NewsArticle, signal?: AbortSignal): Promise<Prediction> {
    const payload = await this.api.post('/predict', article, signal)
    if (!isRecord(payload)) throw invalidPrediction()

    const { prediction, label, probability_fake: fakeProbability, model_version: modelVersion } = payload
    if ((label !== 0 && label !== 1) || (prediction !== 'True' && prediction !== 'Fake') || (label === 0 && prediction !== 'True') ||
        (label === 1 && prediction !== 'Fake') || typeof fakeProbability !== 'number' ||
        !Number.isFinite(fakeProbability) || fakeProbability < 0 || fakeProbability > 1 ||
        typeof modelVersion !== 'string' || !modelVersion) {
      throw invalidPrediction()
    }

    return {
      classification: prediction,
      label,
      fakeProbability,
      trueProbability: 1 - fakeProbability,
      modelVersion,
    }
  }
}

function invalidPrediction(): ApiFailure {
  return new ApiFailure('response', 'El servicio devolvió un resultado incompleto o inesperado.')
}
