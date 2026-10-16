import { createNewsArticle, type NewsArticle } from '../../domain/entities/NewsArticle'
import type { Prediction } from '../../domain/entities/Prediction'
import type { PredictionRepository } from '../../domain/repositories/PredictionRepository'

export class AnalyzeNews {
  private readonly repository: PredictionRepository

  constructor(repository: PredictionRepository) {
    this.repository = repository
  }

  execute(input: NewsArticle, signal?: AbortSignal): Promise<Prediction> {
    return this.repository.predict(createNewsArticle(input), signal)
  }
}
