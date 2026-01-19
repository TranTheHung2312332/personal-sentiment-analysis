import { useState } from 'react'
import type { PredictResponse } from './types/sentiment'

import InputPanel from './components/InputPanel'
import SentimentResult from './components/SentimentResult'
import ProbabilityBars from './components/ProbabilityBars'
import DecisionExplanation from './components/DecisionExplanation'
import CleanedText from './components/CleanedText'
import FeatureSummary from './components/FeatureSummary'

export default function App() {
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [threshold, setThreshold] = useState(0.33)

  return (
    <div className="app">
      <h1>🧠 Sentiment Analyzer – Demo</h1>

      <InputPanel
        neutralThreshold={threshold}
        onThresholdChange={setThreshold}
        onResult={setResult}
      />

      {result && (
        <>
          <SentimentResult result={result} />
          <ProbabilityBars result={result} threshold={threshold} />
          <DecisionExplanation result={result} threshold={threshold} />
          <FeatureSummary result={result} />
          <CleanedText text={result.clean_text} />
        </>
      )}
    </div>
  )
}
