import type { PreprocessOptions } from '../types/sentiment'

interface Props {
  value: PreprocessOptions
  onChange: (v: PreprocessOptions) => void
}

export const DEFAULT_PREPROCESS: PreprocessOptions = {
  religion_norm: true,
  contraction: true,
  emoji_mapping: true,
  emoji_score: true,
  markdown: true,
  mention: true,
  url: true,
  time: true,
  date: true,
  hashtag: true,
  lowercase_norm: true,
  punctuation_norm: true,
  uppercase_ratio: true,
  ex_intensity: true,

  ex_intensity_cap: 5,
  emoji_score_scale: 1.0
}

const BOOL_FIELDS: (keyof PreprocessOptions)[] = [
  'religion_norm',
  'contraction',
  'emoji_mapping',
  'emoji_score',
  'markdown',
  'mention',
  'url',
  'time',
  'date',
  'hashtag',
  'lowercase_norm',
  'punctuation_norm',
  'uppercase_ratio',
  'ex_intensity'
]

export default function PreprocessSettings({ value, onChange }: Props) {
  const toggle = (key: keyof PreprocessOptions) => {
    onChange({
      ...value,
      [key]: !value[key]
    })
  }

  const setNumber = (key: keyof PreprocessOptions, v: string) => {
    onChange({
      ...value,
      [key]: v === '' ? DEFAULT_PREPROCESS[key] : Number(v)
    })
  }

  const reset = () => onChange({ ...DEFAULT_PREPROCESS })

  return (
    <details open>
      <summary>⚙️ Preprocess settings</summary>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
        {BOOL_FIELDS.map(k => (
          <label key={k}>
            <input
              type="checkbox"
              checked={value[k] === true}
              onChange={() => toggle(k)}
            />
            {k}
          </label>
        ))}
      </div>

      <hr />

      <label>
        Exclamation intensity cap
        <input
          type="number"
          min={0}
          value={value.ex_intensity_cap}
          onChange={e => setNumber('ex_intensity_cap', e.target.value)}
        />
      </label>

      <label>
        Emoji score scale
        <input
          type="number"
          step="any"
          value={value.emoji_score_scale}
          onChange={e => setNumber('emoji_score_scale', e.target.value)}
        />
      </label>

      <button type="button" onClick={reset}>
        Reset to default
      </button>
    </details>
  )
}
