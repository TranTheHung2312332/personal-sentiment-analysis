import type { PredictRequest, PredictResponse } from '../types/sentiment'

const TIMEOUT_MS = 3000 // 3s – dùng để test

export async function predict(
  payload: PredictRequest
): Promise<PredictResponse> {
  const controller = new AbortController()

  const timeoutId = setTimeout(() => {
    controller.abort()
  }, TIMEOUT_MS)

  try {
    const res = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    })

    if (!res.ok) {
      const text = await res.text()
      throw new Error(`API error ${res.status}: ${text}`)
    }

    const data: PredictResponse = await res.json()
    return data
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error('Request timed out')
    }
    throw err
  } finally {
    clearTimeout(timeoutId)
  }
}
