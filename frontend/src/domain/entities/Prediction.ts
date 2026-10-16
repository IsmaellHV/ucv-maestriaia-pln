export type Classification = 'True' | 'Fake'

export interface Prediction {
  classification: Classification
  label: 0 | 1
  fakeProbability: number
  trueProbability: number
  modelVersion: string
}
