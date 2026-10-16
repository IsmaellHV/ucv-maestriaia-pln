import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiClient } from '../api/ApiClient'
import { FastApiPredictionRepository } from './FastApiPredictionRepository'

afterEach(() => vi.unstubAllGlobals())

describe('FastApiPredictionRepository', () => {
  it('envía el contrato real y adapta probability_fake al dominio', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      prediction: 'Fake', label: 1, score: 0.8734,
      probability_fake: 0.8734, model_version: 'beto-512',
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)
    const repository = new FastApiPredictionRepository(new ApiClient('/api'))

    await expect(repository.predict({ headline: 'Titular', text: 'Contenido' })).resolves.toEqual({
      classification: 'Fake', label: 1, fakeProbability: 0.8734,
      trueProbability: 0.12660000000000005, modelVersion: 'beto-512',
    })
    expect(fetchMock).toHaveBeenCalledWith('/api/predict', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ headline: 'Titular', text: 'Contenido' }),
    }))
  })

  it('rechaza etiquetas incoherentes y respuestas incompletas', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      prediction: 'True', label: 1, probability_fake: 0.8, model_version: 'beto-512',
    }), { status: 200 })))
    const repository = new FastApiPredictionRepository(new ApiClient('/api'))
    await expect(repository.predict({ headline: '', text: 'Contenido' })).rejects.toThrow('inesperado')
  })
})
