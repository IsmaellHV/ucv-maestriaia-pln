import { ArrowRight, RotateCcw } from 'lucide-react'
import type { FormEvent, RefObject } from 'react'
import type { NewsArticle } from '../../domain/entities/NewsArticle'

interface NewsFormProps {
  article: NewsArticle
  fieldErrors: Partial<Record<keyof NewsArticle, string>>
  loading: boolean
  headlineRef: RefObject<HTMLInputElement | null>
  onChange: (field: keyof NewsArticle, value: string) => void
  onSubmit: () => void
  onReset: () => void
}

export function NewsForm({ article, fieldErrors, loading, headlineRef, onChange, onSubmit, onReset }: NewsFormProps) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    onSubmit()
  }

  return (
    <section className="form-card" aria-labelledby="form-heading">
      <div className="card-topline">
        <span className="section-index">01 / ENTRADA</span>
        <span className="card-topline-note">Análisis de texto</span>
      </div>
      <div className="card-heading-row">
        <div>
          <h2 id="form-heading">Noticia a analizar</h2>
          <p>Introduce el contenido tal como aparece en la noticia.</p>
        </div>
      </div>

      <form noValidate onSubmit={handleSubmit}>
        <div className="field-group">
          <div className="field-heading">
            <label htmlFor="headline">Título de la noticia</label>
            <span>Opcional</span>
          </div>
          <input
            ref={headlineRef}
            id="headline"
            name="headline"
            type="text"
            autoComplete="off"
            placeholder="Escribe el titular de la noticia"
            maxLength={1000}
            value={article.headline}
            disabled={loading}
            aria-invalid={Boolean(fieldErrors.headline)}
            aria-describedby={fieldErrors.headline ? 'headline-error' : undefined}
            onChange={(event) => onChange('headline', event.target.value)}
          />
          {fieldErrors.headline && <p id="headline-error" className="field-error" role="alert">{fieldErrors.headline}</p>}
        </div>

        <div className="field-group body-field">
          <div className="field-heading">
            <label htmlFor="text">Contenido de la noticia <span className="required">*</span></label>
            <span>{article.text.length.toLocaleString('es-CO')} / 100.000</span>
          </div>
          <textarea
            id="text"
            name="text"
            rows={9}
            placeholder="Pega o escribe el cuerpo completo de la noticia aquí..."
            maxLength={100_000}
            value={article.text}
            disabled={loading}
            required
            aria-invalid={Boolean(fieldErrors.text)}
            aria-describedby={fieldErrors.text ? 'text-error' : 'text-hint'}
            onChange={(event) => onChange('text', event.target.value)}
          />
          {fieldErrors.text ? (
            <p id="text-error" className="field-error" role="alert">{fieldErrors.text}</p>
          ) : (
            <p id="text-hint" className="field-hint">El análisis utiliza el título y el contenido. No se admiten enlaces como entrada.</p>
          )}
        </div>

        <div className="form-actions">
          <button className="primary-button" type="submit" disabled={loading} aria-busy={loading}>
            {loading ? <span className="button-spinner" aria-hidden="true" /> : <ArrowRight size={19} strokeWidth={2} aria-hidden="true" />}
            {loading ? 'Analizando noticia…' : 'Analizar noticia'}
          </button>
          <button className="text-button" type="button" disabled={loading || (!article.headline && !article.text)} onClick={onReset}>
            <RotateCcw size={15} aria-hidden="true" /> Limpiar formulario
          </button>
        </div>
      </form>
    </section>
  )
}
