import { Activity, ArrowUpRight, Info, Layers3, ShieldCheck } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { AnalyzeNews } from './application/use-cases/AnalyzeNews'
import { LoadServiceInfo } from './application/use-cases/LoadServiceInfo'
import type { ServiceInfo } from './domain/entities/ServiceInfo'
import { ApiClient } from './infrastructure/api/ApiClient'
import { FastApiPredictionRepository } from './infrastructure/repositories/FastApiPredictionRepository'
import { FastApiServiceInfoRepository } from './infrastructure/repositories/FastApiServiceInfoRepository'
import { NewsForm } from './presentation/components/NewsForm'
import { PredictionPanel } from './presentation/components/PredictionPanel'
import { useAnalysis } from './presentation/hooks/useAnalysis'
import './App.css'

export interface AppDependencies {
  analyzeNews: AnalyzeNews
  loadServiceInfo: LoadServiceInfo
}

const api = new ApiClient()
const defaultDependencies: AppDependencies = {
  analyzeNews: new AnalyzeNews(new FastApiPredictionRepository(api)),
  loadServiceInfo: new LoadServiceInfo(new FastApiServiceInfoRepository(api)),
}

export default function App({ dependencies = defaultDependencies }: { dependencies?: AppDependencies }) {
  const [service, setService] = useState<ServiceInfo | null>(null)
  const [serviceStatus, setServiceStatus] = useState<'checking' | 'online' | 'offline'>('checking')
  const analysis = useAnalysis(dependencies.analyzeNews, {
    onConnectionError: () => setServiceStatus('offline'),
    onSuccess: () => setServiceStatus('online'),
  })
  const headlineRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const controller = new AbortController()
    dependencies.loadServiceInfo.execute(controller.signal)
      .then((info) => { setService(info); setServiceStatus('online') })
      .catch(() => { if (!controller.signal.aborted) setServiceStatus('offline') })
    return () => controller.abort()
  }, [dependencies])

  function resetAndFocus() {
    analysis.reset()
    headlineRef.current?.focus()
  }

  const modelVersion = analysis.prediction?.modelVersion ?? service?.modelVersion
  const modelDescription = modelVersion
    ? `${service?.modelName ?? 'BETO Fake News ES'} · Versión ${modelVersion}`
    : 'Información del modelo disponible al conectar con el servicio.'

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Ir al contenido principal</a>
      <header className="site-header">
        <div className="header-inner">
          <div className="brand">
            <div className="brand-mark"><Layers3 size={24} strokeWidth={1.8} aria-hidden="true" /></div>
            <div><strong>Sistema de Detección<br />de Noticias Falsas</strong><span>PLN EN ESPAÑOL</span></div>
          </div>
          <div className="header-right">
            <span className="header-caption">Clasificación textual · Modelo BETO</span>
            <span className={`service-pill service-pill--${serviceStatus}`} aria-live="polite">
              <span className="service-dot" />
              {serviceStatus === 'online' ? 'Servicio disponible' : serviceStatus === 'offline' ? 'Servicio no disponible' : 'Comprobando servicio'}
            </span>
          </div>
        </div>
      </header>

      <main id="main-content" className="page-content">
        <section className="intro-section" aria-labelledby="page-title">
          <div className="intro-copy">
            <div className="eyebrow"><span className="eyebrow-line" /> SISTEMA DE CLASIFICACIÓN</div>
            <h1 id="page-title">Una lectura más informada empieza aquí<span>.</span></h1>
            <p>Clasificación de contenido informativo mediante Procesamiento de Lenguaje Natural. Analiza el texto de una noticia y consulta la respuesta del modelo.</p>
          </div>
          <div className="intro-note">
            <span className="intro-note-icon"><ShieldCheck size={21} strokeWidth={1.8} aria-hidden="true" /></span>
            <span>Herramienta académica de análisis textual<br /><strong>Resultado orientativo, no verificación factual.</strong></span>
          </div>
        </section>

        <div className="workspace-grid">
          <NewsForm
            article={analysis.article}
            fieldErrors={analysis.fieldErrors}
            loading={analysis.status === 'loading'}
            headlineRef={headlineRef}
            onChange={analysis.update}
            onSubmit={analysis.submit}
            onReset={resetAndFocus}
          />
          <PredictionPanel
            prediction={analysis.prediction}
            loading={analysis.status === 'loading'}
            error={analysis.requestError}
            model={service}
            onReset={resetAndFocus}
          />
        </div>

        <section className="context-section" aria-label="Información sobre la herramienta">
          <div className="context-item"><div className="context-icon"><Activity size={20} strokeWidth={1.7} aria-hidden="true" /></div><div><h3>¿Qué analiza el sistema?</h3><p>Patrones del titular y del cuerpo de la noticia. No consulta sitios externos ni comprueba hechos.</p></div></div>
          <div className="context-item"><div className="context-icon"><Info size={20} strokeWidth={1.7} aria-hidden="true" /></div><div><h3>Cómo interpretar la puntuación</h3><p>Es la salida estimada por el clasificador para cada clase. No mide la veracidad objetiva.</p></div></div>
          <div className="context-item context-item--model"><div className="context-icon"><ArrowUpRight size={20} strokeWidth={1.7} aria-hidden="true" /></div><div><h3>Modelo en uso</h3><p>{modelDescription}</p></div></div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="footer-inner"><span>Proyecto académico · Procesamiento de Lenguaje Natural</span><span>El resultado no constituye una verificación factual de la noticia.</span></div>
      </footer>
    </div>
  )
}
