use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use sentencepiece_model::{
    ModelType as SentencePieceModelType, SentencePieceModel, TrainerSpec, Type as SentencePieceType,
};
use serde_json::{json, Map, Value};

use crate::error::{ErrorKind, Result, TokenizerError};

pub fn model_to_tokenizer_json(bytes: &[u8]) -> Result<Vec<u8>> {
    let model = SentencePieceModel::from_slice(bytes).map_err(|error| {
        TokenizerError::new(
            ErrorKind::InvalidArgument,
            format!("failed to parse sentencepiece model: {error}"),
        )
    })?;

    let trainer = model.trainer().ok_or_else(|| {
        TokenizerError::new(
            ErrorKind::InvalidArgument,
            "sentencepiece model is missing trainer_spec",
        )
    })?;

    let model_type = sentencepiece_model_type(
        trainer
            .model_type
            .unwrap_or(SentencePieceModelType::Unigram as i32),
    )
    .ok_or_else(|| {
        TokenizerError::new(
            ErrorKind::Unimplemented,
            format!(
                "unsupported sentencepiece model type {:?}",
                trainer.model_type
            ),
        )
    })?;

    validate_supported_sentencepiece_settings(trainer)?;

    let root = match model_type {
        SentencePieceModelType::Unigram => build_unigram_tokenizer_json(&model, trainer)?,
        SentencePieceModelType::Bpe => build_bpe_tokenizer_json(&model, trainer)?,
        other => {
            return Err(TokenizerError::new(
                ErrorKind::Unimplemented,
                format!("sentencepiece model type {other:?} is not supported"),
            ))
        }
    };

    serde_json::to_vec(&root).map_err(|error| {
        TokenizerError::new(
            ErrorKind::Internal,
            format!("failed to serialize sentencepiece tokenizer json: {error}"),
        )
    })
}

fn validate_supported_sentencepiece_settings(trainer: &TrainerSpec) -> Result<()> {
    if trainer.treat_whitespace_as_suffix.unwrap_or(false) {
        return Err(TokenizerError::new(
            ErrorKind::Unimplemented,
            "sentencepiece models with treat_whitespace_as_suffix=true are not supported",
        ));
    }

    if !trainer
        .pretokenization_delimiter
        .as_deref()
        .unwrap_or("")
        .is_empty()
    {
        return Err(TokenizerError::new(
            ErrorKind::Unimplemented,
            "sentencepiece models with pretokenization_delimiter are not supported",
        ));
    }

    Ok(())
}

fn build_unigram_tokenizer_json(
    model: &SentencePieceModel,
    trainer: &TrainerSpec,
) -> Result<Value> {
    let mut root = Map::new();
    root.insert("version".to_string(), Value::String("1.0".to_string()));
    root.insert("model".to_string(), unigram_model_json(model, trainer)?);

    if let Some(normalizer) = unigram_normalizer_json(model) {
        root.insert("normalizer".to_string(), normalizer);
    }

    root.insert(
        "pre_tokenizer".to_string(),
        sentencepiece_unigram_pretokenizer_json(add_dummy_prefix(model)),
    );
    root.insert(
        "decoder".to_string(),
        sentencepiece_unigram_decoder_json(add_dummy_prefix(model)),
    );

    if let Some(post_processor) = unigram_post_processor_json(model, trainer) {
        root.insert("post_processor".to_string(), post_processor);
    }

    let added_tokens = added_tokens_json(model, trainer, false);
    if !added_tokens.is_empty() {
        root.insert("added_tokens".to_string(), Value::Array(added_tokens));
    }

    Ok(Value::Object(root))
}

fn build_bpe_tokenizer_json(model: &SentencePieceModel, trainer: &TrainerSpec) -> Result<Value> {
    let mut root = Map::new();
    root.insert("version".to_string(), Value::String("1.0".to_string()));
    root.insert("model".to_string(), bpe_model_json(model, trainer)?);

    if let Some(normalizer) = sentencepiece_bpe_normalizer_json(model) {
        root.insert("normalizer".to_string(), normalizer);
    }

    root.insert(
        "decoder".to_string(),
        sentencepiece_bpe_decoder_json(trainer),
    );

    if let Some(post_processor) = bpe_post_processor_json(model, trainer) {
        root.insert("post_processor".to_string(), post_processor);
    }

    let added_tokens = added_tokens_json(model, trainer, true);
    if !added_tokens.is_empty() {
        root.insert("added_tokens".to_string(), Value::Array(added_tokens));
    }

    Ok(Value::Object(root))
}

fn unigram_model_json(model: &SentencePieceModel, trainer: &TrainerSpec) -> Result<Value> {
    let vocab = model
        .pieces()
        .iter()
        .map(|piece| json!([piece_text(piece), piece.score.unwrap_or(0.0)]))
        .collect::<Vec<_>>();

    Ok(json!({
        "type": "Unigram",
        "vocab": vocab,
        "unk_id": trainer.unk_id.unwrap_or(0),
        "byte_fallback": trainer.byte_fallback.unwrap_or(false)
    }))
}

fn bpe_model_json(model: &SentencePieceModel, trainer: &TrainerSpec) -> Result<Value> {
    let vocab = model
        .pieces()
        .iter()
        .enumerate()
        .map(|(index, piece)| (piece_text(piece).to_string(), Value::from(index as i64)))
        .collect::<Map<String, Value>>();

    let merge_vocab = model
        .pieces()
        .iter()
        .enumerate()
        .map(|(index, piece)| (piece_text(piece).to_string(), index))
        .collect::<std::collections::HashMap<_, _>>();

    let merges = reconstruct_sentencepiece_bpe_merges(&merge_vocab);
    let unk_token = special_piece_from_id(model, trainer.unk_id.unwrap_or(0))
        .unwrap_or_else(|| "<unk>".to_string());

    Ok(json!({
        "type": "BPE",
        "dropout": Value::Null,
        "unk_token": unk_token,
        "continuing_subword_prefix": Value::Null,
        "end_of_word_suffix": Value::Null,
        "fuse_unk": false,
        "byte_fallback": trainer.byte_fallback.unwrap_or(false),
        "vocab": vocab,
        "merges": merges
    }))
}

fn unigram_normalizer_json(model: &SentencePieceModel) -> Option<Value> {
    let normalizer = model.normalizer()?;
    let charsmap = normalizer.precompiled_charsmap.as_deref().unwrap_or(&[]);
    if charsmap.is_empty() {
        None
    } else {
        Some(json!({
            "type": "Precompiled",
            "precompiled_charsmap": BASE64_STANDARD.encode(charsmap)
        }))
    }
}

fn sentencepiece_unigram_pretokenizer_json(add_prefix_space: bool) -> Value {
    json!({
        "type": "Sequence",
        "pretokenizers": [
            {"type": "WhitespaceSplit"},
            {
                "type": "Metaspace",
                "replacement": "▁",
                "str_rep": "▁",
                "add_prefix_space": add_prefix_space
            }
        ]
    })
}

fn sentencepiece_unigram_decoder_json(add_prefix_space: bool) -> Value {
    json!({
        "type": "Metaspace",
        "replacement": "▁",
        "str_rep": "▁",
        "add_prefix_space": add_prefix_space
    })
}

fn unigram_post_processor_json(model: &SentencePieceModel, trainer: &TrainerSpec) -> Option<Value> {
    let eos_id = trainer.eos_id.unwrap_or(-1);
    if eos_id < 0 {
        return None;
    }

    let eos_piece = special_piece_from_id(model, eos_id)?;

    Some(json!({
        "type": "TemplateProcessing",
        "single": [
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": eos_piece, "type_id": 0}}
        ],
        "pair": [
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": eos_piece, "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 0}},
            {"SpecialToken": {"id": eos_piece, "type_id": 0}}
        ],
        "special_tokens": {
            eos_piece.clone(): {
                "id": eos_piece,
                "ids": [eos_id],
                "tokens": [special_piece_from_id(model, eos_id)?]
            }
        }
    }))
}

fn sentencepiece_bpe_normalizer_json(model: &SentencePieceModel) -> Option<Value> {
    let mut normalizers = Vec::new();
    let normalizer = model.normalizer();

    if let Some(normalizer) = normalizer {
        let charsmap = normalizer.precompiled_charsmap.as_deref().unwrap_or(&[]);
        if !charsmap.is_empty() {
            normalizers.push(json!({
                "type": "Precompiled",
                "precompiled_charsmap": BASE64_STANDARD.encode(charsmap)
            }));
        }

        if normalizer.add_dummy_prefix.unwrap_or(true) {
            normalizers.push(json!({"type": "Prepend", "prepend": "▁"}));
        }

        if normalizer.escape_whitespaces.unwrap_or(true) {
            normalizers.push(json!({
                "type": "Replace",
                "pattern": {"String": " "},
                "content": "▁"
            }));
        }
    } else {
        normalizers.push(json!({"type": "Prepend", "prepend": "▁"}));
        normalizers.push(json!({
            "type": "Replace",
            "pattern": {"String": " "},
            "content": "▁"
        }));
    }

    match normalizers.len() {
        0 => None,
        1 => Some(normalizers.remove(0)),
        _ => Some(json!({"type": "Sequence", "normalizers": normalizers})),
    }
}

fn sentencepiece_bpe_decoder_json(trainer: &TrainerSpec) -> Value {
    let mut decoders = vec![json!({
        "type": "Replace",
        "pattern": {"String": "▁"},
        "content": " "
    })];

    if trainer.byte_fallback.unwrap_or(false) {
        decoders.push(json!({"type": "ByteFallback"}));
    }

    decoders.push(json!({"type": "Fuse"}));

    json!({
        "type": "Sequence",
        "decoders": decoders
    })
}

fn bpe_post_processor_json(model: &SentencePieceModel, trainer: &TrainerSpec) -> Option<Value> {
    let bos_id = trainer.bos_id.unwrap_or(-1);
    if bos_id < 0 {
        return None;
    }

    let bos_piece = special_piece_from_id(model, bos_id)?;

    Some(json!({
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": bos_piece, "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}}
        ],
        "pair": [
            {"SpecialToken": {"id": bos_piece, "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": bos_piece, "type_id": 1}},
            {"Sequence": {"id": "B", "type_id": 1}}
        ],
        "special_tokens": {
            bos_piece.clone(): {
                "id": bos_piece,
                "ids": [bos_id],
                "tokens": [special_piece_from_id(model, bos_id)?]
            }
        }
    }))
}

fn added_tokens_json(
    model: &SentencePieceModel,
    trainer: &TrainerSpec,
    normalized: bool,
) -> Vec<Value> {
    let mut entries = std::collections::BTreeMap::<usize, Value>::new();

    for (id, piece) in model.pieces().iter().enumerate() {
        let piece_type = piece_type(piece.r#type.unwrap_or(SentencePieceType::Normal as i32));
        if matches!(
            piece_type,
            SentencePieceType::Unknown
                | SentencePieceType::Control
                | SentencePieceType::UserDefined
        ) {
            entries.insert(
                id,
                added_token_entry(id as i32, piece_text(piece), normalized),
            );
        }
    }

    for id in [
        trainer.unk_id.unwrap_or(-1),
        trainer.bos_id.unwrap_or(-1),
        trainer.eos_id.unwrap_or(-1),
        trainer.pad_id.unwrap_or(-1),
    ] {
        if id >= 0 {
            if let Some(piece) = special_piece_from_id(model, id) {
                entries
                    .entry(id as usize)
                    .or_insert_with(|| added_token_entry(id, &piece, normalized));
            }
        }
    }

    entries.into_values().collect()
}

fn added_token_entry(id: i32, content: &str, normalized: bool) -> Value {
    json!({
        "id": id,
        "content": content,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": normalized,
        "special": true
    })
}

fn special_piece_from_id(model: &SentencePieceModel, id: i32) -> Option<String> {
    if id < 0 {
        return None;
    }

    model
        .pieces()
        .get(id as usize)
        .and_then(|piece| piece.piece.clone())
}

fn sentencepiece_model_type(value: i32) -> Option<SentencePieceModelType> {
    SentencePieceModelType::try_from(value).ok()
}

fn piece_type(value: i32) -> SentencePieceType {
    SentencePieceType::try_from(value).unwrap_or(SentencePieceType::Normal)
}

fn add_dummy_prefix(model: &SentencePieceModel) -> bool {
    model
        .normalizer()
        .and_then(|normalizer| normalizer.add_dummy_prefix)
        .unwrap_or(true)
}

fn reconstruct_sentencepiece_bpe_merges(
    vocab: &std::collections::HashMap<String, usize>,
) -> Vec<String> {
    let mut merges = Vec::<(String, String, usize)>::new();

    for (merge_piece, piece_id) in vocab {
        let mut boundaries = merge_piece
            .char_indices()
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        boundaries.push(merge_piece.len());

        let mut local = Vec::<(String, String, usize)>::new();
        for &boundary in boundaries
            .iter()
            .skip(1)
            .take(boundaries.len().saturating_sub(2))
        {
            let left = &merge_piece[..boundary];
            let right = &merge_piece[boundary..];

            if vocab.contains_key(left) && vocab.contains_key(right) {
                local.push((left.to_string(), right.to_string(), *piece_id));
            }
        }

        local.sort_by(|left, right| {
            (vocab[&left.0], vocab[&left.1]).cmp(&(vocab[&right.0], vocab[&right.1]))
        });
        merges.extend(local);
    }

    merges.sort_by(|left, right| {
        (left.2, left.0.chars().count(), left.1.chars().count()).cmp(&(
            right.2,
            right.0.chars().count(),
            right.1.chars().count(),
        ))
    });

    merges
        .into_iter()
        .map(|(left, right, _)| format!("{left} {right}"))
        .collect()
}

fn piece_text(piece: &sentencepiece_model::SentencePiece) -> &str {
    piece.piece.as_deref().unwrap_or("")
}
