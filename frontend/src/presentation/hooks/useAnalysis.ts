import { useRef, useState } from 'react'
import { AnalyzeNews } from '../../application/use-cases/AnalyzeNews'
import { ArticleValidationError, type NewsArticle } from '../../domain/entities/NewsArticle'
import type { Prediction } from '../../domain/entities/Prediction'
import { ApiFailure } from '../../infrastructure/api/ApiClient'

type AnalysisStatus = 'idle' | 'loading' | 'success' | 'error'
type FieldErrors = Partial<Record<keyof NewsArticle, string>>

interface AnalysisEvents {
  onConnectionError: () => void
  onSuccess: () => void
}

export function useAnalysis(analyzeNews: AnalyzeNews, events: AnalysisEvents) {
  const [article, setArticle] = useState<NewsArticle>({ headline: '', text: '' })
  const [status, setStatus] = useState<AnalysisStatus>('idle')
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [requestError, setRequestError] = useState<string | null>(null)
  const pending = useRef<AbortController | null>(null)

  function update(field: keyof NewsArticle, value: string) {
    pending.current?.abort()
    pending.current = null
    setArticle((current) => ({ ...current, [field]: value }))
    setFieldErrors((current) => ({ ...current, [field]: undefined }))
    setPrediction(null)
    setRequestError(null)
    setStatus('idle')
  }

  async function submit() {
    if (status === 'loading') return
    const controller = new AbortController()
    pending.current = controller
    setFieldErrors({})
    setRequestError(null)

    try {
      const resultPromise = analyzeNews.execute(article, controller.signal)
      setStatus('loading')
      const result = await resultPromise
      if (pending.current !== controller) return
      setPrediction(result)
      setStatus('success')
      events.onSuccess()
    } catch (error) {
      if (pending.current !== controller || (error instanceof DOMException && error.name === 'AbortError')) return
      setStatus('error')
      if (error instanceof ArticleValidationError) {
        setFieldErrors({ [error.field]: error.message })
      } else if (error instanceof ApiFailure) {
        setRequestError(error.message)
        if (error.kind === 'connection') events.onConnectionError()
      } else {
        setRequestError('Ocurrió un error inesperado. Inténtalo de nuevo.')
      }
    } finally {
      if (pending.current === controller) pending.current = null
    }
  }

  function reset() {
    pending.current?.abort()
    pending.current = null
    setArticle({ headline: '', text: '' })
    setStatus('idle')
    setPrediction(null)
    setFieldErrors({})
    setRequestError(null)
  }

  return { article, status, prediction, fieldErrors, requestError, update, submit, reset }
}
