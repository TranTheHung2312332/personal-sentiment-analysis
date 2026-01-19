import type { PredictResponse } from '../types/sentiment'

export default function ProbabilityBars({
  result,
  threshold
}: {
  result: PredictResponse
  threshold: number
}) {
  const labels = ['Positive', 'Neutral', 'Negative']

  return (
    <section>
      <h3>Probabilities</h3>

      {result.probs.map((p, i) => {
        const percent = p * 100
        const isNeutral = labels[i] === 'Neutral'

        return (
          <div key={i} className="prob-row">
            <div className="prob-label">
              {labels[i]}: {percent.toFixed(2)}%
              {isNeutral && (
                <span>
                  {' '}
                  (threshold: {(threshold * 100).toFixed(0)}%)
                </span>
              )}
            </div>

            <div className="prob-bar">
              <div
                className="prob-fill"
                style={{ width: `${percent}%` }}
              />
            </div>
          </div>
        )
      })}
    </section>
  )
}
