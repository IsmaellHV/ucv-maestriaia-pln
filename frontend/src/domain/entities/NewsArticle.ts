export interface NewsArticle {
  headline: string
  text: string
}

export type ArticleField = keyof NewsArticle

export class ArticleValidationError extends Error {
  readonly field: ArticleField

  constructor(field: ArticleField, message: string) {
    super(message)
    this.name = 'ArticleValidationError'
    this.field = field
  }
}

export function createNewsArticle(input: NewsArticle): NewsArticle {
  const headline = input.headline.trim()
  const text = input.text.trim()

  if (headline.length > 1000) {
    throw new ArticleValidationError('headline', 'El título no puede superar los 1.000 caracteres.')
  }
  if (!text) {
    throw new ArticleValidationError('text', 'Ingresa el contenido de la noticia para analizarla.')
  }
  if (text.length > 100_000) {
    throw new ArticleValidationError('text', 'El contenido no puede superar los 100.000 caracteres.')
  }

  return { headline, text }
}
