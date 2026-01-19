import type { PredictResponse } from '../types/sentiment'

export default function DecisionExplanation({
  result,
  threshold
}: {
  result: PredictResponse
  threshold: number
}) {
  const [pos, neu, neg] = result.probs
  const maxIdx = result.probs.indexOf(Math.max(...result.probs))

  let final = maxIdx
  let note = 'Highest probability selected.'

  if (maxIdx === 1 && neu < threshold) {
    final = pos >= neg ? 0 : 2
    note = 'Neutral below threshold → fallback to Positive/Negative.'
  }

  return (
    <section>
      <h3>Decision explanation</h3>
      <ul>
        <li>Positive: {pos.toFixed(4)}</li>
        <li>Neutral: {neu.toFixed(4)}</li>
        <li>Negative: {neg.toFixed(4)}</li>
      </ul>
      <p>{note}</p>
      <strong>Final label: {['Positive', 'Neutral', 'Negative'][final]}</strong>
    </section>
  )
}
