import { ArrowUpRight, FileSearch, RotateCcw, ShieldAlert } from 'lucide-react'
import type { Prediction } from '../../domain/entities/Prediction'
import type { ServiceInfo } from '../../domain/entities/ServiceInfo'

interface PredictionPanelProps {
  prediction: Prediction | null
  loading: boolean
  error: string | null
  model: ServiceInfo | null
  onReset: () => void
}

function percent(value: number): string {
  return new Intl.NumberFormat('es-CO', {
    style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2,
  }).format(value)
}

export function PredictionPanel({ prediction, loading, error, model, onReset }: PredictionPanelProps) {
  const state = prediction?.classification === 'Fake' ? 'fake' : 'true'
  return (
    <section className={`result-card ${prediction ? `result-card--${state}` : ''}`} aria-labelledby="result-heading" aria-live="polite">
      <div className="card-topline">
        <span className="section-index">02 / RESULTADO</span>
        <span className="card-topline-note">Modelo BETO</span>
      </div>
      <h2 id="result-heading">Resultado del análisis</h2>

      {loading ? (
        <div className="result-state result-state--loading" role="status">
          <div className="processing-symbol"><span /></div>
          <h3>Procesando contenido</h3>
          <p>El modelo está analizando los patrones del texto. Esto puede tomar unos segundos.</p>
          <div className="loading-track"><span /></div>
        </div>
      ) : error ? (
        <div className="result-state result-state--error" role="alert">
          <ShieldAlert size={36} strokeWidth={1.5} aria-hidden="true" />
          <h3>No se pudo completar el análisis</h3>
          <p>{error}</p>
          <span className="error-advice">El texto permanece en el formulario para que puedas reintentar.</span>
        </div>
      ) : prediction ? (
        <div className="prediction-content">
          <div className="classification-head">
            <div className="classification-icon"><FileSearch size={25} strokeWidth={1.7} aria-hidden="true" /></div>
            <span className="classification-label">CLASIFICACIÓN DEL MODELO</span>
            <h3>{prediction.classification === 'Fake' ? 'Clasificada como falsa' : 'Clasificada como verdadera'}</h3>
            <p>Resultado de clasificación textual. No constituye una verificación factual.</p>
          </div>

          <div className="score-overview">
            <div>
              <span className="score-kicker">PUNTUACIÓN ASOCIADA A FAKE</span>
              <strong>{percent(prediction.fakeProbability)}</strong>
            </div>
            <ArrowUpRight size={26} strokeWidth={1.4} aria-hidden="true" />
          </div>

          <div className="score-distribution" aria-label="Distribución de puntuaciones del modelo">
            <div className="score-row">
              <div className="score-row-label"><span>Fake</span><strong>{percent(prediction.fakeProbability)}</strong></div>
              <div className="score-track" role="meter" aria-label="Puntuación Fake" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(prediction.fakeProbability * 100)}><span className="score-fill score-fill--fake" style={{ width: `${prediction.fakeProbability * 100}%` }} /></div>
            </div>
            <div className="score-row">
              <div className="score-row-label"><span>True</span><strong>{percent(prediction.trueProbability)}</strong></div>
              <div className="score-track" role="meter" aria-label="Puntuación True" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(prediction.trueProbability * 100)}><span className="score-fill score-fill--true" style={{ width: `${prediction.trueProbability * 100}%` }} /></div>
            </div>
          </div>

          <div className="result-footer">
            <div><span>Modelo utilizado</span><strong>{model?.modelName ?? 'BETO Fake News ES'}</strong></div>
            <button type="button" className="again-button" onClick={onReset}><RotateCcw size={15} aria-hidden="true" /> Analizar otra noticia</button>
          </div>
        </div>
      ) : (
        <div className="result-state result-state--empty">
          <div className="empty-illustration"><FileSearch size={44} strokeWidth={1.2} aria-hidden="true" /></div>
          <h3>Tu resultado aparecerá aquí</h3>
          <p>Completa el contenido de una noticia y selecciona <strong>Analizar noticia</strong> para iniciar.</p>
          <div className="empty-steps"><span>01&nbsp; Ingresa el texto</span><span>02&nbsp; Ejecuta el análisis</span><span>03&nbsp; Interpreta el resultado</span></div>
        </div>
      )}
    </section>
  )
}
