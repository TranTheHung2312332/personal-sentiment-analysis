export interface PreprocessOptions {
    contraction?: boolean
    emoji_mapping?: boolean
    emoji_score?: boolean
    markdown?: boolean
    mention?: boolean
    url?: boolean
    time?: boolean
    date?: boolean
    hashtag?: boolean
    lowercase_norm?: boolean
    punctuation_norm?: boolean
    uppercase_ratio?: boolean
    ex_intensity?: boolean

    ex_intensity_cap?: number
    emoji_score_scale?: number
}

export interface PredictRequest {
    text: string
    preprocess?: PreprocessOptions
    neutral_threshold?: number
}

export interface PredictResponse {
    clean_text: string
    ex_intensity: number
    emoji_score: number
    all_uppercase: boolean
    uppercase_ratio: number
    probs: number[]
    label: string
}