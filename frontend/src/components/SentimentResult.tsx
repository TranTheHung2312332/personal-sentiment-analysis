import type { PredictResponse } from '../types/sentiment'

export default function SentimentResult({
  result
}: {
  result: PredictResponse
}) {
  return (
    <section>
        <h2>Final Sentiment</h2>
        <div className={`result-label ${result.label.toLowerCase()}`}>
            {result.label}
        </div>
    </section>
  )
}
