import { describe, expect, it } from 'vitest'
import { ArticleValidationError, createNewsArticle } from './NewsArticle'

describe('createNewsArticle', () => {
  it('recorta los extremos y conserva el contenido interior', () => {
    expect(createNewsArticle({ headline: '  Un titular  ', text: '  Primer párrafo.\nSegundo párrafo.  ' }))
      .toEqual({ headline: 'Un titular', text: 'Primer párrafo.\nSegundo párrafo.' })
  })

  it('permite un titular vacío y exige contenido', () => {
    expect(createNewsArticle({ headline: ' ', text: 'Texto.' })).toEqual({ headline: '', text: 'Texto.' })
    expect(() => createNewsArticle({ headline: '', text: '   ' })).toThrow(ArticleValidationError)
  })

  it('respeta los límites del contrato FastAPI', () => {
    expect(() => createNewsArticle({ headline: 'a'.repeat(1001), text: 'Texto.' })).toThrow('1.000')
    expect(() => createNewsArticle({ headline: '', text: 'a'.repeat(100_001) })).toThrow('100.000')
  })
})
