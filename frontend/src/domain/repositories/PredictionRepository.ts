import type { NewsArticle } from '../entities/NewsArticle'
import type { Prediction } from '../entities/Prediction'

export interface PredictionRepository {
  predict(article: NewsArticle, signal?: AbortSignal): Promise<Prediction>
}
