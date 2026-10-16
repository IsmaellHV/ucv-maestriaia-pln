import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App, { type AppDependencies } from './App'
import { AnalyzeNews } from './application/use-cases/AnalyzeNews'
import { LoadServiceInfo } from './application/use-cases/LoadServiceInfo'
import type { NewsArticle } from './domain/entities/NewsArticle'
import type { Prediction } from './domain/entities/Prediction'
import { ApiFailure } from './infrastructure/api/ApiClient'

const fakePrediction: Prediction = {
  classification: 'Fake', label: 1, fakeProbability: 0.8734,
  trueProbability: 0.1266, modelVersion: 'beto-512',
}

function createDependencies(predict: (article: NewsArticle) => Promise<Prediction>): AppDependencies {
  return {
    analyzeNews: new AnalyzeNews({ predict }),
    loadServiceInfo: new LoadServiceInfo({ load: async () => ({
      modelName: 'BETO Fake News ES', modelVersion: 'beto-512', maxLength: 512,
    }) }),
  }
}

async function fillForm(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText('Título de la noticia'), '  Un titular  ')
  await user.type(screen.getByLabelText(/Contenido de la noticia/), '  El cuerpo de la noticia.  ')
}

describe('App', () => {
  it('muestra el formulario, el estado inicial y la advertencia factual', async () => {
    render(<App dependencies={createDependencies(async () => fakePrediction)} />)
    expect(screen.getByRole('button', { name: 'Analizar noticia' })).toBeVisible()
    expect(screen.getByText('Tu resultado aparecerá aquí')).toBeVisible()
    expect(screen.getByText(/no constituye una verificación factual de la noticia/i)).toBeVisible()
    expect(await screen.findByText('Servicio disponible')).toBeVisible()
  })

  it('valida el contenido antes de llamar al caso de uso externo', async () => {
    const predict = vi.fn().mockResolvedValue(fakePrediction)
    const user = userEvent.setup()
    render(<App dependencies={createDependencies(predict)} />)
    await user.click(screen.getByRole('button', { name: 'Analizar noticia' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Ingresa el contenido')
    expect(predict).not.toHaveBeenCalled()
  })

  it('ejecuta el caso de uso y muestra una predicción Fake con ambas puntuaciones', async () => {
    const predict = vi.fn().mockResolvedValue(fakePrediction)
    const user = userEvent.setup()
    render(<App dependencies={createDependencies(predict)} />)
    await fillForm(user)
    await user.click(screen.getByRole('button', { name: 'Analizar noticia' }))
    expect(await screen.findByText('Clasificada como falsa')).toBeVisible()
    expect(screen.getAllByText(/87,34/).length).toBeGreaterThan(0)
    expect(screen.getByText(/12,66/)).toBeVisible()
    expect(predict).toHaveBeenCalledWith({ headline: 'Un titular', text: 'El cuerpo de la noticia.' }, expect.any(AbortSignal))
    await user.click(screen.getByRole('button', { name: 'Analizar otra noticia' }))
    expect(screen.getByLabelText('Título de la noticia')).toHaveValue('')
    expect(screen.getByText('Tu resultado aparecerá aquí')).toBeVisible()
  })

  it('muestra la clasificación True sin confundirla con verificación factual', async () => {
    const user = userEvent.setup()
    render(<App dependencies={createDependencies(async () => ({
      ...fakePrediction, classification: 'True', label: 0,
      fakeProbability: 0.15, trueProbability: 0.85,
    }))} />)
    await user.type(screen.getByLabelText(/Contenido de la noticia/), 'Noticia con cuerpo.')
    await user.click(screen.getByRole('button', { name: 'Analizar noticia' }))
    expect(await screen.findByText('Clasificada como verdadera')).toBeVisible()
    expect(screen.getAllByText(/15,00/).length).toBeGreaterThan(0)
    expect(screen.getByText(/85,00/)).toBeVisible()
  })

  it('muestra el estado de carga y un error de conexión en español', async () => {
    let rejectRequest: (error: Error) => void = () => undefined
    const pending = new Promise<Prediction>((_resolve, reject) => { rejectRequest = reject })
    const user = userEvent.setup()
    render(<App dependencies={createDependencies(async () => pending)} />)
    await user.type(screen.getByLabelText(/Contenido de la noticia/), 'Noticia con cuerpo.')
    await user.click(screen.getByRole('button', { name: 'Analizar noticia' }))
    expect(screen.getByText('Procesando contenido')).toBeVisible()
    expect(screen.getByRole('button', { name: /Analizando noticia/ })).toBeDisabled()
    rejectRequest(new ApiFailure('connection', 'No se pudo conectar con el servicio.'))
    expect(await screen.findByText('No se pudo completar el análisis')).toBeVisible()
    expect(screen.getByText('No se pudo conectar con el servicio.')).toBeVisible()
    expect(await screen.findByText('Servicio no disponible')).toBeVisible()
  })

  it('muestra un error del backend y conserva el texto para reintentar', async () => {
    const user = userEvent.setup()
    render(<App dependencies={createDependencies(async () => {
      throw new ApiFailure('backend', 'El servicio no pudo completar el análisis.')
    })} />)
    await user.type(screen.getByLabelText(/Contenido de la noticia/), 'Texto conservado')
    await user.click(screen.getByRole('button', { name: 'Analizar noticia' }))
    expect(await screen.findByText('El servicio no pudo completar el análisis.')).toBeVisible()
    expect(screen.getByLabelText(/Contenido de la noticia/)).toHaveValue('Texto conservado')
    await user.click(screen.getByRole('button', { name: /Limpiar formulario/ }))
    expect(screen.getByLabelText(/Contenido de la noticia/)).toHaveValue('')
  })
})
