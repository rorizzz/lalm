# Data Distribution by Language and Source (TTA)

This page summarizes the training data composition for the TTA model across languages and sources.
We list open-source datasets with links, and aggregate non-open-source data by language as **In-house**.
See the TTA paper for model details: [TTA: Transcribe, Translate and Alignment for Cross-lingual Speech Representation](https://arxiv.org/abs/2511.14410).

| Language | Data Source | Type | Hours | Total Hours | Share |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Chinese (Zh)** | [WenetSpeech](https://github.com/wenet-e2e/WenetSpeech) | Open Source | 10,005 | 129,265 | 37.1% |
| | [AISHELL-2](https://www.aishelltech.com/aishell_2) | Open Source | 1,000 |
| | [AISHELL-1](https://huggingface.co/datasets/AISHELL/AISHELL-1) | Open Source | 150 |
| | [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Open Source | 237 |
| | [Yodas](https://huggingface.co/datasets/espnet/yodas) | Open Source | 222 |
| | *In-house Data* | In-house | 117,651 |
| **Code-Switch** | [TALCS](https://github.com/SpeechClub/TALCS) | Open Source | 555 | 8,924 | 2.6% |
| | *In-house Data* | In-house | 8,369 |
| **English (En)** | [Libriheavy](https://huggingface.co/datasets/pkufool/libriheavy) | Open Source | 45,751 | 107,626 | 30.9% |
| | [Multilingual LibriSpeech (MLS)](https://huggingface.co/datasets/facebook/multilingual_librispeech) | Open Source | 44,659 |
| | [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) | Open Source | 10,000 |
| | [Yodas](https://huggingface.co/datasets/espnet/yodas) | Open Source | 3,426 |
| | [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Open Source | 1,778 |
| | [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) | Open Source | 960 |
| | [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) | Open Source | 522 |
| | [TED-LIUM](https://huggingface.co/datasets/LIUM/tedlium) | Open Source | 453 |
| | [AMI Corpus](https://huggingface.co/datasets/edinburgh-cstr/ami) | Open Source | 77 |
| **Japanese (Ja)** | [ReazonSpeech](https://huggingface.co/datasets/reazon-research/reazonspeech) | Open Source | 35,389 | 40,426 | 11.6% |
| | [Yodas](https://huggingface.co/datasets/espnet/yodas) | Open Source | 499 |
| | [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Open Source | 19 |
| | *In-house Data* | In-house | 4,519 |
| **Korean (Ko)** | [KsponSpeech (AIHub)](https://huggingface.co/datasets/cheulyop/ksponspeech) | Open Source | 965 | 20,095 | 5.8% |
| | [KrespSpeech (AIHub)](https://aihub.or.kr/) | Open Source | 2,906 |
| | [KconfSpeech (AIHub)](https://aihub.or.kr/) | Open Source | 2,928 |
| | [MeetingSpeech (AIHub)](https://aihub.or.kr/) | Open Source | 4,962 |
| | [GyeongsangSpeech (AIHub)](https://aihub.or.kr/) | Open Source | 2,481 |
| | [Yodas](https://huggingface.co/datasets/espnet/yodas) | Open Source | 1,528 |
| | [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Open Source | 1 |
| | *In-house Data (Aggregated)* | In-house | 4,324 |
| **Russian (Ru)** | [Golos](https://huggingface.co/datasets/SberDevices/Golos) | Open Source | 1,221 | 15,246 | 4.4% |
| | [Public Speech & Radio](https://huggingface.co/datasets/bond005/sberdevices_golos_10h) | Open Source | 1,651 |
| | [Buriy Audiobook](https://huggingface.co/datasets/bond005/audio_books_russian) | Open Source | 874 |
| | Public Youtube Dataset | Open Source | 809 |
| | [Yodas](https://huggingface.co/datasets/espnet/yodas) | Open Source | 2,606 |
| | [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Open Source | 37 |
| | *In-house Data* | In-house | 8,048 |
| **Vietnamese (Vi)** | [GigaSpeech 2](https://huggingface.co/datasets/speechcolab/gigaspeech2) | Open Source | 6,048 | 8,390 | 2.4% |
| | [Bud500](https://huggingface.co/datasets/linhtran92/viet_bud500) | Open Source | 324 |
| | [VLSP 2020](https://vlsp.org.vn/vlsp2020) | Open Source | 101 |
| | [ViMD](https://github.com/NhutP/ViMD) | Open Source | 81 |
| | [LSVSC](https://huggingface.co/datasets/doof-ferb/LSVSC) | Open Source | 80 |
| | [Yodas](https://huggingface.co/datasets/espnet/yodas) | Open Source | 140 |
| | [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Open Source | 2 |
| | *In-house Data* | In-house | 1,614 |
| **Indonesian (Id)** | [GigaSpeech 2](https://huggingface.co/datasets/speechcolab/gigaspeech2) | Open Source | 6,352 | 8,238 | 2.4% |
| | [Yodas](https://huggingface.co/datasets/espnet/yodas) | Open Source | 442 |
| | [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Open Source | 7 |
| | *In-house Data* | In-house | 1,437 |
| **French (Fr)** | [Multilingual LibriSpeech (MLS)](https://huggingface.co/datasets/facebook/multilingual_librispeech) | Open Source | 1,076 | 4,124 | 1.2% |
| | [Yodas](https://huggingface.co/datasets/espnet/yodas) | Open Source | 1,423 |
| | [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Open Source | 831 |
| | [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) | Open Source | 205 |
| | *In-house Data* | In-house | 589 |
| **Spanish (Es)** | [Multilingual LibriSpeech (MLS)](https://huggingface.co/datasets/facebook/multilingual_librispeech) | Open Source | 917 | 4,596 | 1.3% |
| | [Yodas](https://huggingface.co/datasets/espnet/yodas) | Open Source | 2,399 |
| | [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Open Source | 502 |
| | [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) | Open Source | 151 |
| | *In-house Data* | In-house | 627 |
| **Portuguese (Pt)** | [Multilingual LibriSpeech (MLS)](https://huggingface.co/datasets/facebook/multilingual_librispeech) | Open Source | 160 | 1,602 | 0.5% |
| | [Yodas](https://huggingface.co/datasets/espnet/yodas) | Open Source | 852 |
| | [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Open Source | 25 |
| | *In-house Data* | In-house | 565 |