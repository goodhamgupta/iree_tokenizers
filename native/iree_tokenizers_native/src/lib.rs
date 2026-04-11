mod error;
mod ffi;
mod sentencepiece;
mod stream;
mod tokenizer;

use rustler::{Env, Term};

fn on_load(_env: Env, _info: Term) -> bool {
    true
}

rustler::init!("Elixir.IREE.Tokenizers.Native", load = on_load);
