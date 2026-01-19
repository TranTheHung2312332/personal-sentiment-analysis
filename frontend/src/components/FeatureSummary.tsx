import type { PredictResponse } from '../types/sentiment'

export default function FeatureSummary({
  result
}: {
  result: PredictResponse
}) {
  return (
    <section>
      <h3>Extracted features</h3>
      <ul>
        <li>Exclamation intensity: {result.ex_intensity}</li>
        <li>Emoji score: {result.emoji_score}</li>
        <li>Uppercase ratio: {result.uppercase_ratio}</li>
        <li>All uppercase: {result.all_uppercase ? 'Yes' : 'No'}</li>
      </ul>
    </section>
  )
}
