import { useState } from 'react'
import { predict } from '../services/api'
import type { PreprocessOptions, PredictResponse } from '../types/sentiment'
import PreprocessSettings from './PreprocessSettings'
import { DEFAULT_PREPROCESS } from './PreprocessSettings'

interface Props {
  neutralThreshold: number
  onThresholdChange: (v: number) => void
  onResult: (r: PredictResponse) => void
}

export default function InputPanel({
  neutralThreshold,
  onThresholdChange,
  onResult
}: Props) {
  const [text, setText] = useState('')
  const [preprocess, setPreprocess] =
    useState<PreprocessOptions>({ ...DEFAULT_PREPROCESS })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async () => {
    setLoading(true)
    setError(null)

    try {
      const res = await predict({
        text,
        preprocess,
        neutral_threshold: neutralThreshold
      })
      onResult(res)
    } catch (e) {
      setError(
        e instanceof Error ? e.message : 'Failed to call API'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <textarea
        placeholder="Paste text here..."
        value={text}
        onChange={e => setText(e.target.value)}
        disabled={loading}
      />

      <button onClick={submit} disabled={loading}>
        {loading ? 'Analyzing…' : 'Analyze'}
      </button>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      <label>
        Neutral threshold: {neutralThreshold.toFixed(2)}
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={neutralThreshold}
          onChange={e => onThresholdChange(Number(e.target.value))}
          disabled={loading}
        />
      </label>

      <PreprocessSettings
        value={preprocess}
        onChange={setPreprocess}
      />
    </section>
  )
}
