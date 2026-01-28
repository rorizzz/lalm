"""Streaming ASR decoding without JIT.

This script mirrors examples/asr/streaming_jit.py but loads a non-JIT model
checkpoint via AutoModel and wraps its encoder with the streaming adapter.
"""

import argparse
import codecs
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature
from transformers import AutoTokenizer

from auden.auto.auto_model import AutoModel
from auden.models.zipformer.utils.padding import make_pad_mask


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors."""
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        data = wave[0]
        if sample_rate != expected_sample_rate:
            logging.warning(
                "sample rate is %s, resampling to %s", sample_rate, expected_sample_rate
            )
            data = torchaudio.functional.resample(
                data, orig_freq=sample_rate, new_freq=expected_sample_rate
            )
        ans.append(data)
    return ans


def token_ids_to_words(token_ids, tokenizer, decoder) -> str:
    token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
    result = bytearray()
    for piece in token_pieces:
        if piece.startswith("<0x") and piece.endswith(">"):
            hex_val = piece[3:-1]
            result.append(int(hex_val, 16))
        else:
            result.extend(piece.encode("utf-8"))
    byte_str = bytes(result)
    text = decoder.decode(byte_str)
    return text.replace("▁", " ").strip()


def greedy_search(
    decoder: torch.nn.Module,
    joiner: torch.nn.Module,
    encoder_out: torch.Tensor,
    decoder_out: Optional[torch.Tensor] = None,
    hyp: Optional[List[int]] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[int], List[int], torch.Tensor]:
    """Streaming greedy search for a single chunk."""
    this_chunk_hyp: List[int] = []
    assert encoder_out.ndim == 2, encoder_out.shape
    context_size = decoder.context_size
    blank_id = decoder.blank_id

    if decoder_out is None:
        assert hyp is None, hyp
        hyp = [blank_id] * context_size
        decoder_input = torch.tensor(
            hyp, dtype=torch.int64, device=device
        ).unsqueeze(0)
        decoder_out = decoder(decoder_input, need_pad=False)
    else:
        assert decoder_out.ndim == 3, decoder_out.shape
        assert hyp is not None, hyp

    T = encoder_out.size(0)
    for i in range(T):
        cur_encoder_out = encoder_out[i].unsqueeze(0).unsqueeze(0)  # (1, 1, C)
        joiner_out = joiner(cur_encoder_out, decoder_out).squeeze(0).squeeze(0)
        y = joiner_out.argmax(dim=-1).item()

        if y != blank_id:
            this_chunk_hyp.append(y)
            hyp.append(y)
            decoder_input = torch.tensor(
                hyp[-context_size:], dtype=torch.int64, device=device
            ).unsqueeze(0)
            decoder_out = decoder(decoder_input, need_pad=False)

    return this_chunk_hyp, hyp, decoder_out


def create_streaming_feature_extractor(sample_rate: int) -> OnlineFeature:
    """Create a CPU streaming feature extractor."""
    opts = FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    return OnlineFbank(opts)


@torch.no_grad()
def decode_one_wav(
    model,
    tokenizer,
    wav_path: str,
    device: torch.device,
    sample_rate: int,
    chunk_sec: float,
    tail_padding_sec: float,
    chunk_size: int,
    left_context_len: int,
) -> str:
    encoder_model = model.encoder
    encoder = encoder_model.encoder
    encoder_embed = encoder_model.encoder_embed
    decoder = model.decoder
    joiner = model.joiner

    online_fbank = create_streaming_feature_extractor(sample_rate)
    wave_samples = read_sound_files([wav_path], expected_sample_rate=sample_rate)[0]

    pad_length = 7 + 2 * 3
    chunk_length = chunk_size * 2
    T = chunk_length + pad_length # T is the real chunk size, which is (2x chunk_size + 13) * 10ms. e.g. chunk_size=16 --> 450ms

    states = encoder.get_init_states(batch_size=1, device=device)
    cached_embed_left_pad = encoder_embed.get_init_states(batch_size=1, device=device)
    processed_lens = torch.zeros(1, dtype=torch.int32, device=device)
    tail_padding = torch.zeros(int(tail_padding_sec * sample_rate), dtype=torch.float32)
    wave_samples = torch.cat([wave_samples, tail_padding])

    chunk = int(chunk_sec * sample_rate) # mimicking audio chunks keep getting in. Note: this is not the actual chunk size.
    num_processed_frames = 0
    hyp = None
    decoder_out = None

    utf8_decoder = codecs.getincrementaldecoder("utf-8")()
    output_pieces: List[str] = []

    start = 0
    while start < wave_samples.numel():
        end = min(start + chunk, wave_samples.numel())
        samples = wave_samples[start:end]
        start += chunk
        online_fbank.accept_waveform(sampling_rate=sample_rate, waveform=samples)

        while online_fbank.num_frames_ready - num_processed_frames >= T:
            frames = []
            for i in range(T):
                frames.append(online_fbank.get_frame(num_processed_frames + i))
            frames = torch.cat(frames, dim=0).to(device).unsqueeze(0)
            x_lens = torch.tensor([T], dtype=torch.int32, device=device)
            x, x_lens, new_cached_embed_left_pad = encoder_embed.streaming_forward(
                x=frames,
                x_lens=x_lens,
                cached_left_pad=cached_embed_left_pad,
            )
            assert x.size(1) == chunk_size, (x.size(1), chunk_size)

            src_key_padding_mask = make_pad_mask(x_lens)
            processed_mask = torch.arange(left_context_len, device=x.device).expand(
                x.size(0), left_context_len
            )
            processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
            new_processed_lens = processed_lens + x_lens
            src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

            x = x.permute(1, 0, 2)
            encoder_out, out_lens, new_states = encoder.streaming_forward(
                x=x,
                x_lens=x_lens,
                states=states,
                src_key_padding_mask=src_key_padding_mask,
            )
            encoder_out = encoder_out.permute(1, 0, 2)
            states = new_states
            cached_embed_left_pad = new_cached_embed_left_pad
            processed_lens = new_processed_lens
            num_processed_frames += chunk_length

            this_chunk_hyp, hyp, decoder_out = greedy_search(
                decoder, joiner, encoder_out.squeeze(0), decoder_out, hyp, device=device
            )
            text = token_ids_to_words(this_chunk_hyp, tokenizer, utf8_decoder)
            if text:
                output_pieces.append(text)
                print(text, flush=True)

    # Flush any remaining bytes from the incremental decoder
    final_tail = utf8_decoder.decode(b"", final=True).strip()
    if final_tail:
        output_pieces.append(final_tail)
        print(final_tail, flush=True)

    return " ".join(output_pieces).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming decode without JIT.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="AudenAI/auden-asr-zh-stream",
        help="Model directory or huggingface model id containing config.json and weights.",
    )
    parser.add_argument(
        "--wav",
        type=str,
        nargs="+",
        required=True,
        help="Input wav file paths.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-sec", type=float, default=0.25)
    parser.add_argument("--tail-padding-sec", type=float, default=1.2)
    parser.add_argument("--chunk-size", type=int, default=16, choices=[16, 32, 64], help="Chunk size for streaming inference.")
    parser.add_argument("--left-context", type=int, default=128, choices=[64, 128, 256], help="Left context for streaming inference.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    logging.info("device: %s", device)

    model = AutoModel.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    if getattr(model, "decoder", None) is None or getattr(model, "joiner", None) is None:
        raise RuntimeError("Model does not have decoder/joiner; streaming RNNT is not available.")

    chunk_size = args.chunk_size
    left_context = args.left_context
    
    # set chunk size and left context for streaming inference
    model.encoder.encoder.chunk_size = [chunk_size]
    model.encoder.encoder.left_context_frames = [left_context]

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(Path(args.model_dir))

    for wav_path in args.wav:
        logging.info("Decoding: %s", wav_path)
        final_text = decode_one_wav(
            model=model,
            tokenizer=tokenizer,
            wav_path=wav_path,
            device=device,
            sample_rate=args.sample_rate,
            chunk_sec=args.chunk_sec,
            tail_padding_sec=args.tail_padding_sec,
            chunk_size=chunk_size,
            left_context_len=left_context,
        )
        logging.info("Final text: %s", final_text)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
