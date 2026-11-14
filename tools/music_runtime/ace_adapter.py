"""Trusted, local-only ACE XL-SFT one-shot runner. No accelerator imports at startup."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import functools
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import sys
import time
import uuid
import wave

CODE_REVISION = 'ca1e85fe9430179831e6bc6be790c332190a3866'
MODEL_REVISION = 'd06de46b4622f781cf07f4a013a67d591ca52819'
COMPONENT_REVISIONS = {'core': '19671f406d603126926c1b7e2adc169acbcade22',
                       'dit': MODEL_REVISION, 'planner': '0a3ec94b557aea7d508da38b31cfe7341f6ff737'}
VISIBILITY_KEYS = {'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
                   'GPU_DEVICE_ORDINAL', 'ZE_AFFINITY_MASK', 'SYCL_DEVICE_FILTER', 'ONEAPI_DEVICE_SELECTOR'}
DTYPES = {'auto', 'float32', 'float16', 'bfloat16'}
PLACEMENTS = {'auto', 'offload', 'resident'}
PLACEMENT_HEADROOM_BYTES = 4 * 1024 ** 3  # Activations, KV cache and allocator slack beside resident weights.
PLACEMENT_ITEMSIZE = {'bfloat16': 2, 'float16': 2, 'float32': 4}  # Bytes per parameter at the requested dtype.
# Bundle prefixes of the four components a run loads; sizing counts every safetensors under them.
PLACEMENT_COMPONENTS = {'planner': 'planner/', 'dit': 'dit/', 'text_encoder': 'core/Qwen3-Embedding-0.6B/', 'vae': 'core/vae/'}
SAFETENSORS_HEADER_LIMIT = 64 * 1024 * 1024
# safetensors dtype names the streaming loader accepts -> (torch dtype attribute, bytes per element). Anything
# else (F8_*, U16/U32/U64, ...) is refused at header time; the pinned bundle stores only BF16 and F32.
SAFETENSORS_DTYPES = {'F64': ('float64', 8), 'F32': ('float32', 4), 'F16': ('float16', 2), 'BF16': ('bfloat16', 2),
                      'I64': ('int64', 8), 'I32': ('int32', 4), 'I16': ('int16', 2), 'I8': ('int8', 1),
                      'U8': ('uint8', 1), 'BOOL': ('bool', 1)}
PLACEMENT_REFUSAL = 'placement_refused_before_load: '
RESIDENT_REFUSAL = 'resident_loader_refused: '
# Windows WDDM charges every device allocation to the owning process's commit charge as well (measured on this
# host, JOB_CAP_SEMANTICS_2159Z / NESTED_JOB_GPU_COMMIT_TEST_*); the launching kit does the host arithmetic.
COMMIT_NOTE = 'WDDM: device allocations also charge process commit 1:1 (informational)'
# Planner route flags a request may set (the pinned GenerationParams names). An absent flag keeps the value below: the
# planner writes audio codes from the user metadata alone and the DiT samples from them. use_cot_caption and
# use_cot_language act only inside the planner's metadata pass, which runs only with use_cot_metas
# (acestep/llm_inference.py:1367); thinking False sends the DiT no planner audio codes (acestep/inference.py:705).
PLANNER_DEFAULTS = {'thinking': True, 'use_cot_metas': False, 'use_cot_caption': False, 'use_cot_language': False}
PLAN_KEYS = ('bpm', 'caption', 'duration', 'genres', 'keyscale', 'language', 'timesignature')  # parse_lm_output's keys
PLAN_TEXT_LIMIT = 4000  # characters kept per recorded planner or DiT text value
# SOURCE_PINS_BEGIN
SOURCE_PINS = {
    "__init__.py": "4ff1234e810a6577d9036402df388eafb0fdf92f77329a5269f65d57289f162b",
    "acestep_v15_pipeline.py": "925c8deb7c0ead48a6815ddc1ada36d59fd6ef3fceba5c3a403b3ed4850b9dd4",
    "api/__init__.py": "c9269400e246e77a661070133d4cf16f28111cdfedb0d461f16fcbcec281e28b",
    "api/http/__init__.py": "bc0ebb247f3290f0fe41bdd26139a20bf72c189abe276a3781cc16acf004f7e5",
    "api/http/audio_route.py": "78a1df4eefde7f455af96a6ef64c09a807e15ba5c128b37fd8777a0ed91da78c",
    "api/http/auth.py": "01edd3efc54440faa7b09717bb587012139d1d957166f56a8f8c814b3aa7841f",
    "api/http/lora_routes.py": "40b625cda8929f56650a513fe9d8c4a7a2904f3e3b383bd8832e98c7fbd0c414",
    "api/http/model_init_service.py": "e5e1c18f29cc5ece306a64bfd4b5b4b9e586907f725740fa9379bd6b2d380932",
    "api/http/model_service_routes.py": "de8b260fe2ecbf4a081b1c778b0b8c4e5928812cea40fac27e6c148f8c6f1327",
    "api/http/query_result_route.py": "6bf0440666a7d23f91150de0d0b6bab42e68e52650e98a8664d9f85230b12284",
    "api/http/query_result_service.py": "609d4cbc718987af42be2e106550358b200f01682e5ded64670a801eab45c6fd",
    "api/http/reinitialize_route.py": "7d13aecfa538542a1c084f74353d321a39c098dba48278f68ba27c72d71f8740",
    "api/http/release_task_audio_paths.py": "c4b14c253e132dd90b5f39c905670f7a2a4a0b6b3ce99730621f96192771bf3f",
    "api/http/release_task_models.py": "d8f6d26e6ec086e8c56814a3b522342d2f3076ff3d5f472faf8cee125d183bb9",
    "api/http/release_task_param_parser.py": "092e9d5a37b7d5458a3b5ca01672920bfc192e4f316a9eef2e2704e5dbd49f13",
    "api/http/release_task_request_builder.py": "031ac48cde3c1afb83e0dd05c03bbba5d5c5339b8f2a5577ea373ab28b4f8255",
    "api/http/release_task_request_parser.py": "ff664073c4730a2690cb266029a206f1bc1793905cc5cb32d733aa0d3433b96e",
    "api/http/release_task_route.py": "7b641d1e3c39d66926aa77e928ad58ff087bd9b4f4dcd9961184e6541c914079",
    "api/http/sample_format_routes.py": "bfe8abe973fddbe394f0a1d9299c3db83c0b122e2d9488d89e71a5eda4dc26b1",
    "api/job_analysis_runtime.py": "180f80ef72e607b36ea55bd6438f859ba5069b38b51443243fcc38e804499af4",
    "api/job_blocking_generation.py": "8d4325c466bac67341449948cbc601e8c2b07d33e4ac30844b686b3acc8a3242",
    "api/job_execution_runtime.py": "b11a5a04b7dc28aa1383cd202bc5bd1afe72f7f978e2f542112ebc55118897a3",
    "api/job_generation_runtime.py": "68ac2895bee871191143a31d8b38bdb6f35ad552f8f0186c0efb27285167923e",
    "api/job_generation_setup.py": "69ee03d5d0091a9176a6001f430b84fb4b9d65ddae6b72cf2ccd367b7de9b113",
    "api/job_llm_preparation.py": "db91e9784f866bbe1394c0e57634740bcbb5f38165fe7869a8ee51eb966e353a",
    "api/job_model_selection.py": "113e81770b9642d048e16628bcae4c2b3c367ae797c875dd25dff26d12a6c67e",
    "api/job_result_payload.py": "0b77bb060f9412ad7cf479a09f1a5b3ae9461072d0169bd774a0222743763a43",
    "api/job_runtime_state.py": "599e952ee5545bfb376d4f436a9c7dfd3c1bbb177c3e5f247c7c0ae3ec1ab9c6",
    "api/jobs/__init__.py": "bd6782896f21164d9ee016f9c4e53f82169b7d1fae8d40a6209d16119a3f300f",
    "api/jobs/local_cache_updates.py": "dcb2fd1cefc89259b9b601e82632967d6e9a10dcf743a69ad1af3b0d9cf6d171",
    "api/jobs/models.py": "f9ea8b715234c56dea820330e43229a5dcaacaa04feae55b73f44cd87832357e",
    "api/jobs/store.py": "95ac9a0297f6d22a8ae25ee411721d6d0a4e6e04810d02b9c21da3ec6a427444",
    "api/jobs/worker_loops.py": "ce1681065b38da5170de961bd506e672f38f6f0d262196dca3e1e7a36c4c2b5a",
    "api/lifespan_runtime.py": "33fc937c1a6e3c35421789ceccb38e07cc9991ff6276fd74b2233414efbda623",
    "api/llm_generation_inputs.py": "a9cfa4f1d812fb4275fe406f95374e1369de124748114086dab5af502f391a82",
    "api/llm_readiness.py": "f34e5d94dbd3c7aa52a0a10edb549976c312612609ee24867382a58b07e4c52b",
    "api/log_capture.py": "b7735b093de85ac122c122b2b545a936eff5025e1c0308649d2df65d236e1d12",
    "api/model_download.py": "38723f6de7ca4349f671719c650526ad7baa7babfe4814c129b3489eac9afdbb",
    "api/route_setup.py": "525d4f52038ff6acb292abe5bc47365968a27e584faf5fae96a208c1caca4cc0",
    "api/runtime_helpers.py": "0d64ce654a8bf4148d99f62a875f4a3597e3109c5b12698afd63d749859041d7",
    "api/server_cli.py": "cd29f85d95f6c6fedd57331576f502e699ab7536c8d468201de80bddb76b7412",
    "api/server_utils.py": "60daa97abd17ee68fc80e08f8761409069119502d51dd4faeedcea6bb5d75ee6",
    "api/startup_llm_init.py": "55865604657e3d0d115c8e4d9fe3e8f3bb564b457a5d42418bb6b4e926315618",
    "api/startup_model_init.py": "cfa0f91d81ceed382ac79c92972da32abb537720594df01a741537d0a46cffd6",
    "api/train_api_dataset_auto_label_async_route.py": "905d4c6ff68ac707390ed9fc40eabeeb49d9c3de1598a4feceaaa14aea151844",
    "api/train_api_dataset_auto_label_routes.py": "480bf821d6f2d04ea124e1d43ddfb9f1b04abbd7ff4ee3af77a10260ff54622d",
    "api/train_api_dataset_auto_label_status_route.py": "b9eae2b119befb09ab4bc14b62dd49feb0941d79eb3ece82be9a1194d8a4af47",
    "api/train_api_dataset_auto_label_sync_route.py": "c686a09c6ca1d0bbb58eea6bdaf1d895ee733c33984b14d167cc211330a3403e",
    "api/train_api_dataset_models.py": "5ce74eca68de1962524d384a6c95e495ee4cece1d18ee83aeb969fefaefbb435",
    "api/train_api_dataset_preprocess_routes.py": "e268da5f291bd7a5dea84c458707d5fdf6dd6eeccb354d8acfac11359991faca",
    "api/train_api_dataset_sample_routes.py": "c71352f2ae5fe944b630f1e9ee44c06d619182f785774436696d31e5a7660d97",
    "api/train_api_dataset_scan_load_routes.py": "2f86a68bc928ac224195f65c360e08ae69135d7230a8eaf8d2d537dfca47cd51",
    "api/train_api_dataset_service.py": "89468b1bbf1c7a844ed80aaf99bcd624e39f81aeeb27a6905c7c39a5a57af32d",
    "api/train_api_dataset_status_routes.py": "87a471fe1e0eae3af3e47b6bb1274a6b457966b3d9de2ad3b73b30cc41f12d09",
    "api/train_api_lokr_start_route.py": "bb50e0c2baa7be879389d340045481d00b224efb5a211ee533a47fb452b89d5a",
    "api/train_api_lora_start_route.py": "19604d1df0cb5549ceaf0c5f24a0b30dbfcaa9dd57d7fd251b9ca19f77d4d35a",
    "api/train_api_models.py": "6f50381424f4b319a7b4f3022ba6041d11b6e46442f2ae4bed963b6a8b6ea516",
    "api/train_api_runtime.py": "f98ae68d4dd7143564a52fcd7c2be4214b994924c1d45f2fd35e7494c9a8a79d",
    "api/train_api_service.py": "bbc4723e056ba863aeec2246b02f08f37a4891104b4e9e75b5ae676d67a215c1",
    "api/worker_runtime.py": "36a683a2dcd4041553848ec77ea2806282086abda00c2a2913be464b6e500d91",
    "api_server.py": "b580bbff8878362f051d8652a2c12c57473d49edae46dd35c47b8f1bd3df3adc",
    "audio_utils.py": "6d93be392e91d74b44913d8b2130da2b64224d67383326bca9ae3489990d5abf",
    "cli_args.py": "ff652773697b84b59e752e0cf47e3686f76cd3a90a5e2b25256d830f6a1d0368",
    "constants.py": "7b8d4ce49649c819d1b3be87a434be2d90768308768d4620639328f906209b22",
    "constrained_logits_processor.py": "84cf84ad894130397ba53a4cbd8666961bf578c77295b79599b288a3825faa32",
    "core/__init__.py": "c5e27c25d766766f891bbd17d0178c03942d70bdd54de67acbdc03ade42f3bab",
    "core/audio/__init__.py": "c08f5ac2ab228ea13090d2789384c644eaeae93863689b0c096a4a49a94b91ba",
    "core/generation/__init__.py": "5b6e480dca8d31337d666760ff14d0d727c080db93166bc946e49d5a6dfdcd78",
    "core/generation/handler/__init__.py": "20c328c2acec74ec530e7b351dae99f17661cdcc726223e228c7f5176a3b5155",
    "core/generation/handler/_flow_edit_dispatch_test_support.py": "ebc01d60cf68ef8081909fded960fc47ce5725a92625ac532454f237519ef307",
    "core/generation/handler/audio_codes.py": "7dec268778c6c6d493aafc8253ae1374bff43459bbd62fb3c16fcd47fefb747a",
    "core/generation/handler/batch_prep.py": "0bf0f6f40e7a74dfee5cbe1d3c9cbb6ab5359371341dd38da408f17769b735e3",
    "core/generation/handler/conditioning_batch.py": "5bc46879830c5e35a48e7ef4a18ca41b4c3f30f7887928b31cea3cf259e82958",
    "core/generation/handler/conditioning_embed.py": "45e80702219496edcef5712fa8403583c61f0df128833b9657d0d129d035a3f2",
    "core/generation/handler/conditioning_masks.py": "5d9c1fdb249931794af76db83166347be0e8a081f958338c9a7963131e87a365",
    "core/generation/handler/conditioning_target.py": "472b302837800f54972b5b5a3d6374b7855ee3eac9e05762757fcb41d5455068",
    "core/generation/handler/conditioning_text.py": "0313ef5fc5ade1baa5a806f3748e5c853961768d057045e040fcb28a33a153b6",
    "core/generation/handler/diffusion.py": "8864cfd8195c4ab0b90cfe3c6d216f7efdc3a88e174f63d1a5d1e853b3e355d9",
    "core/generation/handler/generate_music.py": "0da6727cb5dc834697b6ff5774af79ace977b259205755e526bf1841f1ad913d",
    "core/generation/handler/generate_music_decode.py": "5ebe4dfd65635a687a5692d37ab377ab959170e764fbedf54d6e35711cc40416",
    "core/generation/handler/generate_music_execute.py": "625d7e477c88dc795913a1cbb1ba8d3d3dfa512831460fb26f780dc8cb5b0b79",
    "core/generation/handler/generate_music_payload.py": "5d3a6b4a646547d59bbdab204684c4918ee86a767dda4af2579e7b8dd8bce910",
    "core/generation/handler/generate_music_request.py": "281e481be2a0a1ab0b5dbc3cf471aef8611096305fc46cee5bcc22eaeb297c8c",
    "core/generation/handler/init_service.py": "8154b3e8398b5565e3b6e27af95e8063ffd74663d6bd235e845622206d0e559f",
    "core/generation/handler/init_service_catalog.py": "48d41952ec25da854bfcb2ccdf9fa0cbf285c1f5e50cb34b9e978d6e634620d2",
    "core/generation/handler/init_service_downloads.py": "02a95f2293dc0cd82ff5046816503668f8339157ba0b18715e061f3142999f8f",
    "core/generation/handler/init_service_loader.py": "22b41692bcb73fead4c831d16eaef3dda56cc995577f2353d763445dae7362e1",
    "core/generation/handler/init_service_loader_components.py": "621fc7d24ee847835de2eb66f35451120cb92310cf5e112e4edbdf955535d6de",
    "core/generation/handler/init_service_memory_basic.py": "f717db29be3e9e86890cddaec4155dd470c52f0643f3fb56d1fcf6da6bc91f42",
    "core/generation/handler/init_service_memory_transfer.py": "9291b08d9ea3622e7c8be5d64b567c976221696fe2e8e4980b10b15ad86ab37a",
    "core/generation/handler/init_service_offload_context.py": "9e5dac9c938e2ec8ef37b9040117182abc70740e518df4d4be5c4b1b004b8ca2",
    "core/generation/handler/init_service_orchestrator.py": "df1e3ac43db9018793adae343760b43c56e7e81648cfcd057730162352c49d72",
    "core/generation/handler/init_service_setup.py": "24d98ab1407e742de095d2fe9caf55180e74714f83cdd9670a9bf8d2fa1529b3",
    "core/generation/handler/io_audio.py": "98798d2678e6e46fd437508864245f032eedf736249af3183b3dcc856642e449",
    "core/generation/handler/lora/__init__.py": "ef9097b2f3a8bb161a77fabaf12437a52a27638a87407ef3b4a1c7d4afcdef43",
    "core/generation/handler/lora/adapter_discovery.py": "a6d3398814091c5a9a7f7af7651e0f56233c2facef94f2f22a3dc80662eb35e6",
    "core/generation/handler/lora/controls.py": "328c4a50b2f9d6fd1745fad69951fe7902d777737c34b273618ff14ff8208c54",
    "core/generation/handler/lora/lifecycle.py": "653d77beac88006f2d2f2f74621d72872623b805cf4fd9d4f64356ca47b80462",
    "core/generation/handler/lora/registry_builder.py": "91792c4636d5c1bb2bc04a206d8324aaa37ac8129b852320309d3d53b0ca2610",
    "core/generation/handler/lora/registry_state.py": "bd2ef6995e8c13cec710d6e05e76e36f3ba1cf08b49d628b788f6e8cb8a32061",
    "core/generation/handler/lora/scale_apply.py": "beedb94abd6d689a10367c471889787c99a3eaa5dde81bd67e203a4fdd4c1f72",
    "core/generation/handler/lora_manager.py": "c1d185409abc3e9882d52d018e0ab8ba9b9ec2fcf0f44c9831c120c0d18d0b0b",
    "core/generation/handler/lyric_alignment_common.py": "ec35e392a2354696e9def4f98db27c415f09763e839609dd368532971ec0eb08",
    "core/generation/handler/lyric_score.py": "92ccffbe079482b9e0e7a7fc6304a408fa7d996a7f9e88c7b3f819ee5d463510",
    "core/generation/handler/lyric_timestamp.py": "f8b54ba0d0668eae45988c4682cc2ae59aa923b1f6464d215165ac9b3b36e20c",
    "core/generation/handler/memory_utils.py": "48e61bc8890206ab3f2ce85e7feeb3153c87f5b33d6cff2c56b5c5890367f5c1",
    "core/generation/handler/metadata_utils.py": "b8485fe9c3a8ffdf0ee0147df68366770abfe6b68e6cd286dd4722d9379cdc5b",
    "core/generation/handler/mlx_dit_init.py": "e3051df98c80671583107f6ccc87a5e2c9f58a0425aad53df77f5f1ec3c4377d",
    "core/generation/handler/mlx_vae_decode_native.py": "60795c3cabe0f883c468e6a864e7d2937178d7df305c7b0507971377490d3166",
    "core/generation/handler/mlx_vae_encode_native.py": "7ca77b0ee3a03468cad94651b354b279f8d1af81ab14c77c8d85f276732d2724",
    "core/generation/handler/mlx_vae_init.py": "732ae73b34832edb610f37aba8a7db9bda0b7f332c1c4c11ee16bd923589a456",
    "core/generation/handler/padding_utils.py": "16de06cdc360c23dfbec563e36da64a8f5711e409204f800fae86b78f129e0c4",
    "core/generation/handler/progress.py": "f60a7b7282d5e9c65ff189f83006038adee9dd673a337d18991c0de59c2313fe",
    "core/generation/handler/prompt_utils.py": "242835ae29cc1bfcc40aadf1c46d8156230349d6e2f05436b2c85b3491484fae",
    "core/generation/handler/repaint_step_injection.py": "0d436325f8c33019671d326df45f366ca89a349b2aa29242a7a84fd4c4d5cabc",
    "core/generation/handler/repaint_waveform_splice.py": "b6143c4ae796f77d1b43a9c9e3ceb6c49ede73c5d474be9e4b78c9122a21141a",
    "core/generation/handler/service_generate.py": "b44f85dca2320ed11ed26d3395108e0fdc9be82bf9435c5b75fb88910926874c",
    "core/generation/handler/service_generate_execute.py": "508fd70b669600ca8626c703d4213eda2c6ef46fcaee4fd9d662cb4843d83bdc",
    "core/generation/handler/service_generate_flow_edit.py": "252a0e845002e3163f5a128d9e11502ec70a619992ef2706de5fde0073260b8a",
    "core/generation/handler/service_generate_flow_edit_source.py": "21815b0896245fc4a178f100a397e32f043b62b47e0a51f68823dc8c129bf140",
    "core/generation/handler/service_generate_outputs.py": "bc165a5b95ab8fa5b9645d90a2291a57ffa2f4f93f76d9d46131e000daba424f",
    "core/generation/handler/service_generate_request.py": "dde325040f63e3a6a2e5dfa406dd2886d82f6418e63195b61d081c367545daca",
    "core/generation/handler/task_utils.py": "a5c90c6af54d1eb207cbf79db39535e115361542cb37735ed4359287bd257a66",
    "core/generation/handler/training_preset.py": "c6f3059b69ba1d3b5092c4154df0be10ce6e14645b670c7b40c850ba858dfa4a",
    "core/generation/handler/vae_decode.py": "199538c4762dfd0da9ff0895793c640ed715208774f14aed106514be9ce121f4",
    "core/generation/handler/vae_decode_chunks.py": "8b0a38acb7619a7672968756cc01562f0681f628633f9b9fc21b3fe5c5fec80d",
    "core/generation/handler/vae_decode_test_helpers.py": "2560ccc27cc19c9f4ed92ca9d9a2442426f2098f21a8ff838afd86f87391cfbd",
    "core/generation/handler/vae_encode.py": "c91896380556bddd0bd2f581db6d3c9e0cb7798e4ccb27508fdb01b7873300f7",
    "core/generation/handler/vae_encode_chunks.py": "4bb260c3585209e34e7f04b6395e04558610c3dc08f0d30bf9d3ed969346ac1d",
    "core/llm/__init__.py": "6ede4a58e14cf9a7805028e33d38d0e81827e30664878c5fa6c61aa8b3a8c01d",
    "core/lora/__init__.py": "b24583cb94d90f466fc5499d1269d658e914831fc2c229ef32ab1763b112ccb8",
    "core/lora/introspection.py": "26e5156e02099ef2d2fac79e84d126ca62293c19d0a1f96f87c16df28d0664b8",
    "core/lora/registry.py": "046f092ef8d193525da3bd9b7cf5f3cc227c7e8d321a7866fe25ced7e5b94604",
    "core/lora/scaling.py": "140bee27c67511434c64a50af90dc2c9144ab3a365818b2e2e55d5dcf5084f33",
    "core/lora/service.py": "706e881e895026b24239df3cc04245446cca83363eafe9029f151f9274df4470",
    "core/scoring/__init__.py": "0a07e1e91c4e4c4b3240262ceb8372c29bcc8a9513d8ae814864e0b9eacc98b5",
    "core/scoring/_dtw.py": "2d2252e7108f296cd26722c7b622e1c1a1ed47c71459a8ff873164c4c1ada962",
    "core/scoring/dit_alignment.py": "b5dcf958c9fb756c34f2f4dc29a9a2df5b782582e41d23cf98223aff3cfcfb84",
    "core/scoring/dit_score.py": "73700ff976549339dad3d6c500f8cbe9a8cce08d1979bbb85e3e3ed53e0605a1",
    "core/scoring/lm_score.py": "399b47d7269da2c8dee41d3316a67094f5b0dc2ec464b88ed52c6fce8f9d589e",
    "core/system/__init__.py": "7a36fe91a14d86b84e65871d8c3a84142c67abde12ad23b41691e14ffae69e6c",
    "dataset/__init__.py": "d45c12c10fc10802cf43938b1f03392b9e194e81d12b5589f7183cbe0dc6b74a",
    "dataset/builder/__init__.py": "f6b22c00a521764a9079ec81d962236a8cd916175629157a3f18e89f45f82280",
    "dataset/runtime/__init__.py": "7339620e512ea8c12c3b6e14e6a04b304ea4fd92984dd3e0554424a17de2d7ae",
    "dataset_handler.py": "a04d2f34b1ade2d66f8704ee301edc82da4f57d2b82c1aceb220a0e0535541aa",
    "debug_utils.py": "727a9340089dd60fd7c058330fad08fdba7964e21160e04b47d91d3e02a2a8a5",
    "genres_vocab.txt": "57767ff18f5361b09f0dd563763a4e23d3c6793d1aec03411367e4d9266b2868",
    "gpu_config.py": "01987e28eef138fa27191adcfb1ee46ff2bf96540eafca8887d5243599bf549a",
    "handler.py": "0a215d92aff3a85ea50be8cfa88e819175a0b8a4254f8024893c157a4dec6875",
    "inference.py": "f43827def02a1122e2de483961b04f16544e4da9108eed9e42fd529175c15df5",
    "launcher_compat.py": "6e523f98c10d6f6c7166f749f38ea758cbf33c0621e31882775742cf08cc83ff",
    "llm_backend_compat.py": "bb4a6caed78d97574b8dd22292f324b35ae423b730bd8129cf02d2c2416e42c5",
    "llm_inference.py": "afe1baf01d8a594bd06148bf9c79a0f66f84072b5158ae598340b34cc215b9c6",
    "local_cache.py": "d784800b4fa98b84b4851813a363a54a1356202c3ef9fa431acc3ed92e90a7da",
    "model_downloader.py": "34fcd1914620c11ad487070f47966803d48a290b7b05b42259cf9fd437c032d5",
    "models/__init__.py": "748eec9214b9351add46ded4b2a16b6fce25101611cbca5015dd902b5f21e2dc",
    "models/base/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/base/apg_guidance.py": "33985a589f3af4f3daff071b71cae68da1bf971070da7c27223acbea9f5f2ae7",
    "models/base/configuration_acestep_v15.py": "5f15aa79fe793bba3c00b6992dd193a269954950603d5031ecd7703b0fa69da1",
    "models/base/modeling_acestep_v15_base.py": "f3edaa0f86ce174c0c6703a8d80442673783ba96515a0fa84126a9b605b0919d",
    "models/common/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/common/_flow_edit_test_support.py": "52863123de94c38ea966eddf903726dc5c21afc1f5dbfcb88db4e8c922fa190f",
    "models/common/apg_guidance.py": "89ecdc4686e174225d6e6d304939d75892297dc3c19fbb11e04345dc3d4685db",
    "models/common/configuration_acestep_v15.py": "0d66f38eba6a3d1665c9449a1a52724d68354477bc88dcf02389640e3602a0c7",
    "models/common/dcw_correction.py": "96f7a3073919f0fc5d1ce5459e359fd03f107f623f43d5f9b0720a84c1f3db9b",
    "models/common/dcw_loader.py": "34cb5c60ed7cb46439ddeea620d6cec6947f04f0530a129f5579348aa4900308",
    "models/common/dcw_primitives.py": "b87aeecd1f6e5d5282f60dd3de71781d3acd671e632cf7ef8253be757e043b5e",
    "models/common/flow_edit.py": "00cc4c1e04826b46223d52f247b0288a3c1d6bf3d9490a5a313340b1b6bb52b2",
    "models/common/flow_edit_helpers.py": "9c92c982d63fe8a0e2503521070a8b44e8843c5647cc4243642a584db02ddd54",
    "models/common/flow_edit_pipeline.py": "3b4a9a829eca67233ab9e5c7d469d2ff889b4b3837efef8e5e118f4bd2485c8c",
    "models/mlx/__init__.py": "aea67f5a1029b0ac649a02b9ce83de560ad5f258b446b15155c28519cb4f3582",
    "models/mlx/dcw_correction_mlx.py": "289be08bdc5b64a597f615913fd9616e578612f8059bc4e8aa44dd0abea037f2",
    "models/mlx/dit_convert.py": "d481ab46d29e5168d2cd2b4d9ca60342e2714b118a4fc9fcdfe2c3fcd1c8c26f",
    "models/mlx/dit_generate.py": "6fd77dfdb42e581fef9c5842120ce41f628a9602d280be1ef1b54e7aa9aea23d",
    "models/mlx/dit_model.py": "289e9d22d1f214f3fe25fbe55993445dc648bfbfe3df270b486f0491661aed73",
    "models/mlx/vae_convert.py": "e8f0dc89dbc9c68b2fecd28db6e0ff9954985bcef5575765e27fe431e82b29ea",
    "models/mlx/vae_model.py": "6e9142be51fb66b38b2e101a205a6ce707f1d00de9f161fb540fd259d6e073f4",
    "models/sft/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/sft/apg_guidance.py": "33985a589f3af4f3daff071b71cae68da1bf971070da7c27223acbea9f5f2ae7",
    "models/sft/configuration_acestep_v15.py": "5f15aa79fe793bba3c00b6992dd193a269954950603d5031ecd7703b0fa69da1",
    "models/sft/modeling_acestep_v15_base.py": "421f520cd8d2b1d01dbbc8d236e522aff58b0e2bbf6144c4f5f3510e8a0539d4",
    "models/turbo/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/turbo/configuration_acestep_v15.py": "5f15aa79fe793bba3c00b6992dd193a269954950603d5031ecd7703b0fa69da1",
    "models/turbo/modeling_acestep_v15_turbo.py": "71b22e01b490bd9e149e4c0c5da2e1f512126f3ab3316e0ae97779af84d19512",
    "models/xl_base/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/xl_base/apg_guidance.py": "33985a589f3af4f3daff071b71cae68da1bf971070da7c27223acbea9f5f2ae7",
    "models/xl_base/configuration_acestep_v15.py": "5f15aa79fe793bba3c00b6992dd193a269954950603d5031ecd7703b0fa69da1",
    "models/xl_base/modeling_acestep_v15_xl_base.py": "e0a61a9da2c5e4d38a995417526c595d1bf6ae3820ff376ac7c419331fe63cec",
    "models/xl_sft/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/xl_sft/apg_guidance.py": "33985a589f3af4f3daff071b71cae68da1bf971070da7c27223acbea9f5f2ae7",
    "models/xl_sft/configuration_acestep_v15.py": "5f15aa79fe793bba3c00b6992dd193a269954950603d5031ecd7703b0fa69da1",
    "models/xl_sft/modeling_acestep_v15_xl_base.py": "e0a61a9da2c5e4d38a995417526c595d1bf6ae3820ff376ac7c419331fe63cec",
    "models/xl_turbo/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/xl_turbo/configuration_acestep_v15.py": "5f15aa79fe793bba3c00b6992dd193a269954950603d5031ecd7703b0fa69da1",
    "models/xl_turbo/modeling_acestep_v15_xl_turbo.py": "e470a453e8a1b783750a76daece42e1939935c3d4d41c462358ffde4b70c4501",
    "openrouter_adapter.py": "0a7d6ee52ef9f10c3ba68939d695096ca02cb0184ce8eac987ef0fd199ecd7fd",
    "openrouter_models.py": "e0e1d7c49f73c87adfe132a526fb23927bbe445537e2ee7add5c8f52ac983701",
    "text_tasks/__init__.py": "88fd5cf0f9d646324bf099bc31a79328a842a34b17d7480403fbb4dc171f69c5",
    "text_tasks/external_ai_json_parsing.py": "1debaf3a1ebcb3906577ba834f7420c1b69f725d9bc26b78b69a2d5f53e2f24e",
    "text_tasks/external_ai_protocols.py": "40c0136484f160d4addd1aef61d6eabfbba1479f24b0c9b3348a9b2ece6b7994",
    "text_tasks/external_ai_request_helpers.py": "ff82861ccfcaacbb4b18d266d53422042d31c9f730f7a6218c296841bf5e40f4",
    "text_tasks/external_ai_response_parsing.py": "4ad696e5515a2e30c7c7b9d811776aafd2ded17a84690609bd45b6cf78fea6ba",
    "text_tasks/external_ai_types.py": "621dc1ca62b3f830ba477407a8507e914eca3d30ee69f5f0a8a7b542f38f9ae8",
    "text_tasks/external_lm_captioning.py": "6eea03470286dbc8bdc6ab715479bc4781318e0a79692d84d316b5ff2a0639c6",
    "text_tasks/external_lm_captioning_fallback_locale.py": "c46107edd0c507f2fd4849a06608cfed2fcd65d83dd72da54b9718e962d794ed",
    "text_tasks/external_lm_model_cache.py": "380e81d40ce42cb59cc07175d4bc40930fbcd5e02496d66e7e164f0041110666",
    "text_tasks/external_lm_model_discovery.py": "660f62c282fbc5addc8b788038bd4252f2a37ef1d22e272667f3862f2b3a7d2b",
    "text_tasks/external_lm_providers.py": "523bf3d81fecaa2940a324bd1aae0295b042d8058d977c80a693b105f23494f3",
    "text_tasks/external_lm_runtime_store.py": "4fe066fe919a9df6306172b53819c70513449e7ee9bad4c8d1bb8952e4030f3c",
    "text_tasks/passphrase_store.py": "f971dda0f6570df1e9d80098de55e447fa8c8360a368e6a4966a87436f143898",
    "text_tasks/secure_secret_store.py": "3f9f84fdfb9672780ead07616da524a070d8640b8d4cf664cab958a1e7a99d3b"
}
# SOURCE_PINS_END


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read_json(path, limit=2 * 1024 * 1024):
    require(path.is_file() and (limit is None or path.stat().st_size <= limit), 'Missing or oversized JSON input')
    return json.loads(path.read_bytes(), object_pairs_hook=unique_pairs)


# Windows publication of progress.json/result.json: the trusted parent polls progress.json with an ordinary open(), which
# holds the destination without FILE_SHARE_DELETE for the duration of its read, so MoveFileEx(REPLACE_EXISTING) fails with
# ERROR_ACCESS_DENIED (5) or ERROR_SHARING_VIOLATION (32) for a few milliseconds. The fully written temporary file stays in
# place across a bounded retry; an exhausted budget or any other error surfaces unchanged. POSIX rename never takes this path.
REPLACE_RETRY_ATTEMPTS = 50
REPLACE_RETRY_SLEEP_SECONDS = 0.02
REPLACE_RETRY_WINERRORS = frozenset({5, 32})


def replace_retrying(temporary, path):
    """Rename temporary over path; on Windows, wait out a concurrent reader of path within a bounded budget.

    Returns the number of attempts used. Errors other than the two Windows sharing errors, and the sharing errors once the
    budget is exhausted, propagate unchanged with the temporary file left on disk for diagnosis.
    """
    for attempt in range(1, REPLACE_RETRY_ATTEMPTS + 1):
        try:
            temporary.replace(path)
            return attempt
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in REPLACE_RETRY_WINERRORS or attempt >= REPLACE_RETRY_ATTEMPTS:
                raise
            time.sleep(REPLACE_RETRY_SLEEP_SECONDS)
    raise AssertionError('unreachable')  # pragma: no cover


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    replace_retrying(temporary, path)


def report_failure(result_path, progress, payload):
    """Best-effort failure publication: never lets a publication error replace the primary generation error."""
    for publish in (lambda: atomic_json(result_path, payload), lambda: progress('failed')):
        try:
            publish()
        except OSError as publication_error:
            print(f'ace_adapter: failure report not published: {type(publication_error).__name__}: {publication_error}',
                  file=sys.stderr, flush=True)


def ordinary(path, *, directory=False):
    require(path.is_absolute(), 'Path must be absolute')
    for part in (*reversed(path.parents), path):
        mode = part.lstat()
        require(not (stat.S_ISLNK(mode.st_mode) or getattr(mode, 'st_file_attributes', 0) & 0x400) or (part != path and os_alias(part, mode)),
                'Linked paths are not accepted')  # os_alias: OS-owned POSIX ancestors such as macOS /var -> /private/var
    require(path.is_dir() if directory else path.is_file(), 'Expected ordinary ' + ('directory' if directory else 'file'))
    return path.resolve()


def member(root, name):
    require(isinstance(name, str) and bool(name), 'Invalid component member')
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.as_posix() == name and '..' not in relative.parts
            and '\\' not in name and ':' not in name, 'Unsafe component member')
    path = ordinary(root / name)
    require(path.is_relative_to(root), 'Component member escapes bundle')
    return path


def normalized_uuid(value):
    if isinstance(value, (bytes, bytearray)) and len(value) == 16:
        value = str(uuid.UUID(bytes=bytes(value)))
    value = str(value)
    if value.startswith('GPU-'):
        value = value[4:]
    require(bool(re.fullmatch(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', value)),
            'Runtime did not expose a complete physical CUDA UUID')
    return 'GPU-' + value.lower()


def validate_request(value):
    allowed = {'bundle', 'variant', 'code_revision', 'model_revision', 'offline', 'runtime_contract',
               'prompt', 'mode', 'lyrics', 'duration', 'vocal_language', 'bpm', 'key_scale', 'seed',
               'guidance_scale', 'steps', 'reference_audio', *PLANNER_DEFAULTS}
    require(isinstance(value, dict) and not set(value) - allowed, 'Unsupported request fields')
    require(value.get('offline') is True and value.get('variant') == 'xl-sft'
            and value.get('code_revision') == CODE_REVISION and value.get('model_revision') == MODEL_REVISION,
            'Only the pinned offline XL-SFT recipe is supported')
    require(not value.get('reference_audio'), 'Reference audio is not supported by this runner')
    require(value.get('mode', 'instrumental') in {'instrumental', 'song'}, 'Unsupported music mode')
    for key, limit in (('vocal_language', 32), ('key_scale', 64)):
        require(key not in value or isinstance(value[key], str) and len(value[key]) <= limit, 'Invalid ' + key)
    contract = value.get('runtime_contract')
    required = {'version', 'requested_device', 'device', 'expected_uuid', 'visibility', 'dtype', 'threads',
                'interop_threads', 'max_audio_seconds', 'max_output_bytes', 'max_samples',
                'max_steps', 'max_prompt_chars', 'max_text_chars', 'placement'}
    require(isinstance(contract, dict) and set(contract) == required and type(contract['version']) is int and contract['version'] == 1,
            'Unsupported runtime contract')
    require(contract['placement'] in PLACEMENTS, 'Unsupported placement request')
    for key in ('max_steps', 'max_prompt_chars', 'max_text_chars'):
        require(type(contract[key]) is int and contract[key] > 0, 'Invalid configured ' + key)
    require(isinstance(value.get('prompt'), str) and value['prompt'].strip()
            and len(value['prompt']) <= contract['max_prompt_chars'], 'Music prompt exceeds configured limit or is invalid')
    require('lyrics' not in value or isinstance(value['lyrics'], str)
            and len(value['lyrics']) <= contract['max_text_chars'], 'Music lyrics exceed configured limit or are invalid')
    require(not (value.get('mode', 'instrumental') == 'instrumental' and value.get('lyrics', '').strip()),
            'Instrumental requests cannot silently discard supplied lyrics')
    require(contract['device'] in {'cpu', 'cuda:0', 'auto'} and isinstance(contract['requested_device'], str),
            'Only CPU/CUDA runtime families are supported')
    require(contract['dtype'] in DTYPES, 'Unsupported dtype')
    require(type(contract['threads']) is int and contract['threads'] > 0
            and type(contract['interop_threads']) is int and contract['interop_threads'] == 1,
            'Invalid thread contract')
    require(isinstance(contract['visibility'], dict) and not set(contract['visibility']) - VISIBILITY_KEYS
            and all(isinstance(v, str) for v in contract['visibility'].values()), 'Invalid visibility contract')
    if contract['expected_uuid'] is not None:
        require(normalized_uuid(contract['expected_uuid']) == contract['expected_uuid'], 'Noncanonical expected UUID')
    for key in ('max_output_bytes', 'max_samples'):
        require(type(contract[key]) is int and contract[key] > 0, 'Invalid output bound')
    for key, lower, upper in (('duration', 10, 600), ('guidance_scale', 0, 30)):
        number = value.get(key, 30 if key == 'duration' else 7.0)
        require(type(number) in (int, float) and math.isfinite(number) and lower <= number <= upper, 'Invalid ' + key)
    maximum = contract['max_audio_seconds']
    require(type(maximum) in (int, float) and math.isfinite(maximum) and maximum > 0
            and value.get('duration', 30) <= maximum, 'Duration exceeds runtime limit')
    frames = math.ceil(value.get('duration', 30) * 48000)
    require(frames * 2 <= contract['max_samples'] and frames * 4 + 44 <= contract['max_output_bytes'],
            'Requested stereo output exceeds the runtime budget')
    require(type(value.get('steps', 50)) is int and 1 <= value.get('steps', 50) <= contract['max_steps'],
            'Sampling steps exceed configured limit or are invalid')
    if 'seed' in value:
        require(type(value['seed']) is int and 0 <= value['seed'] <= 2**32 - 1, 'Invalid seed')
    if 'bpm' in value:
        require(type(value['bpm']) is int and 30 <= value['bpm'] <= 300, 'Invalid BPM')
    for key in PLANNER_DEFAULTS:
        require(key not in value or type(value[key]) is bool, 'Invalid ' + key)
    flags = planner_flags(value)
    require(flags['use_cot_metas'] or not (flags['use_cot_caption'] or flags['use_cot_language']),
            'use_cot_caption and use_cot_language need use_cot_metas')
    require(flags['thinking'] or flags['use_cot_metas'], 'thinking=false needs use_cot_metas so the planner still runs')
    return value


def planner_flags(request):
    """The request's planner route flags, each absent one at its PLANNER_DEFAULTS value."""
    return {key: request.get(key, default) for key, default in PLANNER_DEFAULTS.items()}


def bind_environment(contract, scratch, environment=None):
    env = os.environ if environment is None else environment
    for key in VISIBILITY_KEYS:
        expected = contract['visibility'].get(key)
        require(env.get(key) == expected, 'Child visibility differs from parent contract: ' + key)
    if contract['device'] == 'cpu':
        require(env.get('CUDA_VISIBLE_DEVICES') == '' and contract['dtype'] in {'auto', 'float32'},
                'CPU requires hidden CUDA and float32')
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
               HF_HOME=str(scratch / 'hf-cache'), XDG_CACHE_HOME=str(scratch / 'cache'),
               ACESTEP_PROJECT_ROOT=str(scratch), ACESTEP_DISABLE_TQDM='1', ACESTEP_DEBUG_STATS='0',
               TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = str(contract['threads'])
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'SPACE_ID', 'ACESTEP_CHECKPOINTS_DIR',
                'ACESTEP_VAE_CHECKPOINT', 'ACESTEP_VAE_ON_CPU'):
        env.pop(key, None)


def verify_bundle(request, progress):
    root = ordinary(Path(request['bundle']), directory=True)
    manifest_path = member(root, 'titanos-music.json')
    manifest = read_json(manifest_path)
    require(manifest.get('schema_version') == 1 and manifest.get('family') == 'ace-step-1.5'
            and manifest.get('variant') == 'xl-sft' and manifest.get('code_revision') == CODE_REVISION
            and manifest.get('model_revision') == MODEL_REVISION, 'Bundle recipe differs')
    require(set(manifest.get('components', {})) == set(COMPONENT_REVISIONS), 'Selected recipe requires core, dit and 4B planner')
    entries, inventories = {}, {}
    inventory_names = set()
    for component, revision in COMPONENT_REVISIONS.items():
        row = manifest['components'][component]
        inventory_path = member(root, row['path'])
        inventory_names.add(row['path'])
        require(row['revision'] == revision and digest(inventory_path) == row['sha256'], 'Component inventory identity differs')
        inventory = read_json(inventory_path)
        require(inventory.get('revision') == revision and isinstance(inventory.get('files'), list)
                and 0 < len(inventory['files']) <= 10000, 'Invalid component inventory')
        inventories[component] = {'revision': revision, 'sha256': row['sha256'], 'path': row['path']}
        for item in inventory['files']:
            name = item['path']
            require(name.startswith(component + '/') and name not in entries, 'Component path is duplicated or misplaced')
            path = member(root, name)
            require(path.suffix.lower() not in {'.py', '.pyc', '.pyd', '.so', '.dll', '.exe', '.bin', '.ckpt'},
                    'Executable or unsafe checkpoint file is not supported')
            require(path.suffix.lower() != '.pt' or name == 'dit/silence_latent.pt', 'Only the pinned silence tensor may use .pt')
            require(type(item['bytes']) is int and path.stat().st_size == item['bytes']
                    and digest(path) == item['sha256'], 'Checkpoint data differs: ' + name)
            entries[name] = item
    actual = set()
    for folder, directories, filenames in os.walk(root, followlinks=False):
        for name in directories:
            ordinary(Path(folder) / name, directory=True)
        for name in filenames:
            path = ordinary(Path(folder) / name)
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(entries) | inventory_names | {'titanos-music.json'},
            'Unlisted or missing bundle payload could change local model loading')
    needed = {'dit/config.json', 'dit/silence_latent.pt', 'core/vae/config.json',
              'core/Qwen3-Embedding-0.6B/config.json', 'core/Qwen3-Embedding-0.6B/tokenizer_config.json',
              'planner/config.json', 'planner/tokenizer_config.json'}
    require(needed <= set(entries), 'Selected component configuration is incomplete')
    configs = {name: read_json(root / name, limit=32 * 1024 * 1024) for name in needed if name.endswith('.json')}
    require(configs['dit/config.json'].get('model_type') == 'acestep', 'Unexpected DiT architecture')
    for name, architecture in (('core/Qwen3-Embedding-0.6B/config.json', 'Qwen3Model'), ('planner/config.json', 'Qwen3ForCausalLM')):
        require(configs[name].get('model_type') == 'qwen3' and configs[name].get('architectures') == [architecture],
                'Unexpected Qwen architecture')
    for name in ('core/Qwen3-Embedding-0.6B/tokenizer_config.json', 'planner/tokenizer_config.json'):
        require(configs[name].get('tokenizer_class') in {'Qwen2Tokenizer', 'Qwen2TokenizerFast'}, 'Unsupported tokenizer class')
    vae = configs['core/vae/config.json']
    require(vae.get('_class_name') == 'AutoencoderOobleck' and vae.get('sampling_rate') == 48000
            and vae.get('audio_channels') == 2 and math.prod(vae.get('downsampling_ratios', [])) == 1920,
            'Unexpected audio VAE format')
    progress('bundle_verified')
    return root, {'manifest_sha256': digest(manifest_path), 'components': inventories, 'payload_files_verified': len(entries)}


def verify_source():
    spec = importlib.util.find_spec('acestep')
    require(spec is not None and spec.origin is not None, 'Pinned ACE package is not installed')
    package = ordinary(Path(spec.origin)).parent
    require(bool(SOURCE_PINS), 'Runner has no bound source provenance')
    for relative, expected in SOURCE_PINS.items():
        path = member(package, relative)
        require(digest(path) == expected, 'Installed ACE source differs: ' + relative)
    return {'revision': CODE_REVISION, 'package_root': str(package), 'files_verified': len(SOURCE_PINS)}


def resolve_runtime(torch, contract):
    require(not getattr(torch.version, 'hip', None), 'HIP is not qualified by this CUDA runner')
    requested = contract['device']
    device = 'cpu'
    visible = []
    bf16 = False
    if requested != 'cpu' and torch.cuda.is_available():
        count = torch.cuda.device_count()
        require(0 < count <= 128, 'Invalid visible CUDA device count')
        visible = [{'local_device': f'cuda:{index}', 'uuid': normalized_uuid(torch.cuda.get_device_properties(index).uuid)}
                   for index in range(count)]
        require(len({row['uuid'] for row in visible}) == count, 'Duplicate observed CUDA identities')
        if requested == 'cuda:0':
            require(count == 1, 'Explicit CUDA request was not narrowed to one device')
        device = 'cuda:0'
        torch.cuda.set_device(0)
        bf16 = bool(torch.cuda.is_bf16_supported())
    elif requested == 'cuda:0':
        raise ValueError('Requested CUDA device is unavailable; no CPU fallback')
    if contract['expected_uuid'] is not None:
        require(device != 'cpu' and visible[0]['uuid'] == contract['expected_uuid'], 'Observed physical UUID differs')
    dtype = contract['dtype']
    if device == 'cpu':
        require(dtype in {'auto', 'float32'}, 'CPU supports float32 only')
        dtype = 'float32'
    else:
        if dtype == 'auto':
            dtype = 'bfloat16' if bf16 else 'float32'
        require(dtype != 'bfloat16' or bf16, 'Requested bfloat16 is not supported by the selected CUDA device')
    torch.set_num_threads(contract['threads'])
    torch.set_num_interop_threads(contract['interop_threads'])
    require(torch.get_num_threads() == contract['threads'] and torch.get_num_interop_threads() == contract['interop_threads'],
            'Framework thread settings differ')
    runtime = {'contract_version': 1, 'requested_device': contract['requested_device'], 'resolved_device': device,
               'device_uuid': visible[0]['uuid'] if visible else None, 'visible_devices': visible,
               'identity_source': 'torch.cuda.get_device_properties' if visible else 'parameter.device',
               'torch_version': str(torch.__version__), 'backend': 'cuda' if visible else 'cpu',
               'bfloat16_supported': bf16, 'visibility': {k: os.environ[k] for k in contract['visibility']},
               'torch_num_threads': torch.get_num_threads(), 'torch_num_interop_threads': torch.get_num_interop_threads(),
               'parameter_placements': []}
    return runtime, dtype


def safetensors_table(path):
    """Strictly parsed safetensors header: an 8-byte little-endian header length, then the JSON tensor table.

    Only the header is read; tensor data is never touched. Returns (entries, header_length, data_bytes) where
    entries maps tensor name -> {'dtype': <safetensors name>, 'shape': [...], 'data_offsets': [begin, end]}
    with __metadata__ removed. Refused: duplicate keys, unknown dtypes, negative dimensions, a byte span that
    differs from numel x itemsize, and any layout that is not exactly the file's data region tiled once in
    offset order (out of file, overlapping, gapped, or leaving a tail). The streaming loader trusts these
    offsets for its seeks, so the check is the whole safety argument for reading tensor bytes."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, 'Truncated safetensors header: ' + path.name)
        length = int.from_bytes(prefix, 'little')
        require(0 < length <= SAFETENSORS_HEADER_LIMIT and 8 + length <= size, 'Invalid safetensors header length: ' + path.name)
        header = stream.read(length)
    require(len(header) == length, 'Truncated safetensors header: ' + path.name)
    table = json.loads(header.decode('utf-8'), object_pairs_hook=unique_pairs)
    require(isinstance(table, dict) and any(name != '__metadata__' for name in table), 'Empty safetensors table: ' + path.name)
    data_bytes = size - 8 - length
    entries = {}
    for name, entry in table.items():
        if name == '__metadata__':
            continue
        shape = entry.get('shape') if isinstance(entry, dict) else None
        offsets = entry.get('data_offsets') if isinstance(entry, dict) else None
        require(isinstance(shape, list) and all(type(value) is int and value >= 0 for value in shape)
                and isinstance(entry.get('dtype'), str), 'Invalid safetensors tensor entry: ' + path.name)
        require(entry['dtype'] in SAFETENSORS_DTYPES, 'Unsupported safetensors dtype {} in {}'.format(entry['dtype'], path.name))
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(value) is int for value in offsets)
                and 0 <= offsets[0] <= offsets[1] <= data_bytes, 'Safetensors data offsets outside the file: ' + path.name)
        require(offsets[1] - offsets[0] == math.prod(shape) * SAFETENSORS_DTYPES[entry['dtype']][1],
                'Safetensors byte span differs from shape x itemsize: ' + path.name)
        entries[name] = {'dtype': entry['dtype'], 'shape': list(shape), 'data_offsets': [offsets[0], offsets[1]]}
    cursor = 0
    for entry in sorted(entries.values(), key=lambda row: row['data_offsets']):
        require(entry['data_offsets'][0] == cursor, 'Safetensors data offsets overlap or leave a gap: ' + path.name)
        cursor = entry['data_offsets'][1]
    require(cursor == data_bytes, 'Safetensors data region is not fully described by its header: ' + path.name)
    return entries, length, data_bytes


def safetensors_numel(path):
    """Parameter count from a safetensors header. Counting elements rather than file bytes matters because
    the DiT is stored F32 and loads at the requested dtype."""
    entries, _length, _data_bytes = safetensors_table(path)
    return sum(math.prod(entry['shape']) for entry in entries.values())


def component_safetensors(root):
    """Inventoried safetensors paths per loaded component, in inventory order.

    verify_bundle has already bound every inventory entry to the file on disk. A safetensors file
    outside the four loaded components would change what the pinned recipe loads and is refused."""
    manifest = read_json(member(root, 'titanos-music.json'))
    files = {name: [] for name in PLACEMENT_COMPONENTS}
    for component in COMPONENT_REVISIONS:
        inventory = read_json(member(root, manifest['components'][component]['path']))
        for item in inventory['files']:
            name = item['path']
            if not name.lower().endswith('.safetensors'):
                continue
            target = next((key for key, prefix in PLACEMENT_COMPONENTS.items() if name.startswith(prefix)), None)
            require(target is not None, 'Safetensors payload outside the loaded components: ' + name)
            files[target].append(member(root, name))
    require(all(files.values()), 'Bundle lacks safetensors weights for a loaded component')
    return files


def bundle_parameter_counts(root):
    """Parameters per loaded component, from the headers of every inventoried safetensors file."""
    return {component: sum(safetensors_numel(path) for path in paths) for component, paths in component_safetensors(root).items()}


def stream_safetensors(torch, path, device, target_dtype, loader=None):
    """Load one safetensors file tensor by tensor straight onto the device.

    Resident mode never maps a shard: safetensors' safe_open/load_file charge the whole file to process commit
    at OPEN (measured OSError 1455 inside the Job cap), and copy-on-write file mappings (UntypedStorage.from_file)
    are counted against the Job but only fail later at page touch (both measured with small owned probes; run r3
    died natively with 0xC0000005 while the first DiT shard loaded through the earlier from_pretrained/safe_open
    loader, and which class it hit is not proven). Here the host transient is exactly one tensor: a writable bytearray of the tensor's
    bytes is filled with readinto, viewed with torch.frombuffer at the stored dtype, copied to the device at that
    dtype, and only then cast on the device (a cross-device .to with a dtype change would first materialise the
    cast copy on the host, doubling the transient), and both host objects are dropped before the next tensor.
    Integer/bool tensors keep their stored dtype. Bytes read, tensor count and the largest tensor are recorded
    in `loader` when given."""
    entries, header_length, data_bytes = safetensors_table(path)
    tensors, largest = {}, 0
    with path.open('rb') as stream:
        for name, entry in sorted(entries.items(), key=lambda item: item[1]['data_offsets']):
            stored = getattr(torch, SAFETENSORS_DTYPES[entry['dtype']][0])
            begin, end = entry['data_offsets']
            shape = tuple(entry['shape'])
            if end == begin:
                tensor = torch.empty(shape, dtype=stored)  # Empty tensor: frombuffer refuses a zero-length buffer.
            else:
                buffer = bytearray(end - begin)
                stream.seek(8 + header_length + begin)
                require(stream.readinto(buffer) == end - begin, 'Truncated safetensors data: ' + path.name)
                tensor = torch.frombuffer(buffer, dtype=stored).reshape(shape)
                del buffer  # The tensor still views the bytearray; both go when `tensor` is dropped below.
            moved = tensor.to(device=device)
            if target_dtype is not None and moved.is_floating_point() and moved.dtype != target_dtype:
                moved = moved.to(dtype=target_dtype)
            tensors[name] = moved
            largest = max(largest, end - begin)
            del tensor, moved
    if loader is not None:
        loader['files'] += 1
        loader['tensors'] += len(tensors)
        loader['bytes_read'] += data_bytes
        loader['largest_tensor_bytes'] = max(loader['largest_tensor_bytes'], largest)
    return tensors


def new_loader_evidence():
    return {'mode': 'streamed', 'host_mapping': 'none', 'files': 0, 'tensors': 0, 'bytes_read': 0,
            'largest_tensor_bytes': 0, 'components': {}}


def resident_constructor(torch, model_cls, config, directory, attention, dtype):
    """Callable that builds a transformers model eagerly the way the pinned from_pretrained (4.57.6) would.

    from_pretrained deep-copies the caller's config (modeling_utils.py:4965) and, before `cls(config)`, sets
    `config._attn_implementation` from the kwarg (:4867; __init__ then validates it with is_init_check=True at
    :2076-2078, which for sdpa only checks class support, :2573-2578, and touches no device), `config.dtype`
    (and every sub-config's) from the dtype kwarg (_get_dtype :1258-1262), `config.name_or_path` (:4966), and
    the process default floating dtype (_set_default_dtype :1287, restored after construction :5023-5024) so
    buffers computed under it (the FSQ tables) match. Construction runs under `no_init_weights()` (:4378;
    init_empty_weights is applied by the caller so buffers such as rotary inv_freq are created normally).
    `generation_config` comes from the model config in __init__ (:2094); the planner directory has no
    generation_config.json, so from_pretrained's :5034-5044 lookup would fall back to the same object."""
    import copy
    config = copy.deepcopy(config)
    config._attn_implementation = attention
    config.dtype = dtype
    for key in getattr(config, 'sub_configs', {}) or {}:
        getattr(config, key).dtype = dtype
    config.name_or_path = str(directory)

    def construct():
        from transformers.modeling_utils import no_init_weights
        previous = model_cls._set_default_dtype(dtype)
        try:
            with no_init_weights():
                return model_cls(config)
        finally:
            torch.set_default_dtype(previous)
    return construct


def build_resident(torch, component, construct, files, device, dtype, tied_keys, loader):
    """Construct a module empty and fill it from streamed shards; refuse anything that is not a complete,
    on-device, at-dtype set of weights.

    The module is built under accelerate.init_empty_weights(include_buffers=False), so parameters start on the
    meta device and buffers are created normally on the host. Every shard is streamed with stream_safetensors
    (tensors already on the device), transformers' per-key rename hook (_fix_state_dict_key_on_load; no pinned
    checkpoint key needs it) is applied when the class has one, and the state is assigned into the module
    (load_state_dict(strict=False, assign=True); torch's own rejection of the state, e.g. a shape mismatch, is
    re-raised as a refusal). Accounting: a key seen twice across shards or any unexpected key refuses; a missing
    key is tolerated only when it is one of the module's tied-weight keys AND `tie_word_embeddings` is configured;
    `tied_keys` is the recipe's expectation and must equal that derived set. Whenever tied keys are configured,
    tie_weights() runs (as from_pretrained does unconditionally, modeling_utils.py:5065) so a checkpoint that also
    ships a tied key cannot leave a second device copy, and every configured tied key must then be bound to the
    very same tensor object as another on-device parameter. Remaining host buffers move with
    model.to(device) (what from_pretrained's dispatch_model does for a single-device map), after which every
    parameter and buffer must sit on the device and every floating parameter must carry the target dtype."""
    from accelerate import init_empty_weights

    def refuse(reason):
        raise ValueError(RESIDENT_REFUSAL + component + ': ' + reason)

    with init_empty_weights(include_buffers=False):
        model = construct()
    config = getattr(model, 'config', None)
    derived = set(getattr(model, '_tied_weights_keys', None) or []) if getattr(config, 'tie_word_embeddings', False) else set()
    if derived != set(tied_keys):
        refuse('tied-weight keys {} differ from the recipe {}'.format(sorted(derived), sorted(tied_keys)))
    rename = getattr(model, '_fix_state_dict_key_on_load', None)
    state, renamed = {}, 0
    for path in files:
        for key, tensor in stream_safetensors(torch, path, device, dtype, loader).items():
            target, changed = rename(key) if rename is not None else (key, False)
            renamed += bool(changed)
            if target in state:
                refuse('duplicate checkpoint key {} across shards'.format(key))
            state[target] = tensor
    try:
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    except RuntimeError as exc:  # e.g. 'size mismatch for <key>': the module rejected the checkpoint's shapes
        del state
        refuse('state rejected by the module: ' + ' '.join(str(exc).split())[:400])
    del state
    missing, unexpected = sorted(missing), sorted(unexpected)
    if unexpected:
        refuse('unexpected checkpoint keys {}'.format(unexpected[:8]))
    if set(missing) - derived:
        refuse('missing checkpoint keys {}'.format(sorted(set(missing) - derived)[:8]))
    if derived:
        model.tie_weights()
    parameters = list(model.named_parameters(remove_duplicate=False))
    for key in sorted(derived):
        tied = dict(parameters).get(key)
        sources = [name for name, value in parameters if name != key and value is tied]
        if tied is None or not sources or str(tied.device) != device:
            refuse('tied key {} is not bound to another on-device parameter'.format(key))
    if any(str(value.device) == 'meta' for _name, value in parameters):
        refuse('parameters left uninitialised after loading')
    model.to(device)
    for name, value in model.named_parameters(remove_duplicate=False):
        if str(value.device) != device or value.is_floating_point() and value.dtype != dtype:
            refuse('parameter {} is on {} as {}'.format(name, value.device, value.dtype))
    buffers = Counter()
    for name, value in model.named_buffers(remove_duplicate=False):
        if str(value.device) != device:
            refuse('buffer {} is on {}'.format(name, value.device))
        buffers[str(value.dtype).removeprefix('torch.')] += 1
    loader['components'][component] = {'files': len(files), 'parameters': len(parameters), 'tied_keys': sorted(derived),
                                       'renamed_keys': renamed, 'buffers': dict(buffers)}
    return model.eval()


def resolve_placement(torch, runtime, contract, bundle, dtype_name):
    """Decide where the weights live for this run BEFORE any weight is loaded, and record the basis.

    host: CPU run, every component stays on the host. offload: today's behaviour, each component
    moves to the device for its stage and back. resident: every component loads straight onto the
    device and stays there, with no CPU-offload fallback anywhere, before or during generation.
    Sizing comes only from the verified bundle's safetensors headers at the resolved dtype; no
    caller-supplied estimate or basis string is accepted. Nothing is loaded here."""
    requested = contract['placement']
    require(requested in PLACEMENTS, 'Unsupported placement request')
    require(dtype_name in PLACEMENT_ITEMSIZE, 'Unsupported resolved dtype')
    device = runtime['resolved_device']
    counts = bundle_parameter_counts(bundle)
    itemsize = PLACEMENT_ITEMSIZE[dtype_name]
    total = sum(counts.values()) * itemsize
    # The planner is released before the DiT loads, so the largest simultaneously held set is
    # max(planner, dit) + text encoder + vae. The admission bill stays the whole bundle (conservative).
    peak = (max(counts['planner'], counts['dit']) + counts['text_encoder'] + counts['vae']) * itemsize
    required = total + PLACEMENT_HEADROOM_BYTES
    evidence = {'requested': requested, 'resolved': None, 'device_free_bytes': None, 'required_bytes': required,
                'total_resident_bytes': total, 'peak_set_bytes': peak, 'headroom_bytes': PLACEMENT_HEADROOM_BYTES,
                'accelerate_version': None, 'basis': ''}
    sizing = 'required {} = {} resident bytes ({} parameters x {} bytes {}, peak set {}) + {} headroom'.format(
        required, total, sum(counts.values()), itemsize, dtype_name, peak, PLACEMENT_HEADROOM_BYTES)

    def refuse(reason):
        # The measured numbers stay in the retained receipt; nothing has been loaded yet.
        runtime['placement_refused'] = {**evidence, 'basis': reason}
        raise ValueError(PLACEMENT_REFUSAL + reason)

    if device == 'cpu':
        if requested == 'resident':
            refuse('resident placement requires a CUDA device')
        resolved = 'host'
        basis = 'resolved device is cpu: all components stay on the host; placement request {} does not apply; {}'.format(requested, sizing)
    else:
        free = int(torch.cuda.mem_get_info(0)[0])
        require(free >= 0, 'Invalid device free-memory report')
        evidence['device_free_bytes'] = free
        if requested == 'offload':
            resolved = 'offload'
            basis = 'offload requested: components move to the device per stage and back to the host; free {}; {}'.format(free, sizing)
        else:
            # Lazy: only the resident loader needs accelerate (init_empty_weights), and it is imported after torch (main).
            try:
                import accelerate
            except ImportError:
                accelerate = None
            else:
                evidence['accelerate_version'] = str(getattr(accelerate, '__version__', 'unknown'))
            if accelerate is None:
                obstacle = 'accelerate is not importable, so meta-device construction for resident loading is unavailable'
            elif dtype_name == 'bfloat16' and runtime.get('bfloat16_supported') is not True:
                obstacle = 'resolved dtype bfloat16 is not supported by the selected CUDA device'
            elif free < required:
                obstacle = 'device free {} < {}'.format(free, sizing)
            else:
                obstacle = None
            if obstacle is None:
                resolved = 'resident'
                basis = '{}: free {} >= {}; accelerate {}; every component loads onto the device and stays'.format(
                    requested, free, sizing, evidence['accelerate_version'])
                if sys.platform == 'win32':
                    basis += '; ' + COMMIT_NOTE
            elif requested == 'resident':
                refuse('resident placement: ' + obstacle)
            else:
                resolved = 'offload'
                basis = 'auto: {}; components move per stage'.format(obstacle)
    require(0 < len(basis) <= 400, 'Placement basis exceeds bound')
    evidence.update(resolved=resolved, basis=basis)
    runtime['placement'] = evidence
    return evidence


class PlacementObserver:
    def __init__(self, runtime, dtype):
        self.runtime, self.dtype = runtime, dtype
        self.handles, self.executed = {}, set()

    def record(self, component, stage, model, execution=False):
        groups = Counter((str(p.device), str(p.dtype).removeprefix('torch.')) for p in model.parameters())
        require(bool(groups), 'Component exposes no parameters: ' + component)
        if execution:
            require(set(groups) == {(self.runtime['resolved_device'], self.dtype)}, 'Actual component placement/dtype differs: ' + component)
            self.executed.add(component)
        for (device, dtype), count in groups.items():
            row = {'component': component, 'stage': stage, 'device': device, 'dtype': dtype, 'parameter_tensors': count}
            if row not in self.runtime['parameter_placements']:
                self.runtime['parameter_placements'].append(row)
        require(len(self.runtime['parameter_placements']) <= 128, 'Placement evidence exceeds bound')

    def attach(self, component, model, execution_module=None):
        def observed(_module, _inputs):
            if component not in self.executed:
                self.record(component, 'execute', model, execution=True)
        self.handles[component] = (execution_module if execution_module is not None else model).register_forward_pre_hook(observed)

    def detach(self, component):
        handle = self.handles.pop(component, None)
        if handle is not None:
            handle.remove()


class NoTruncationTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, **kwargs):
        if kwargs.get('truncation') and kwargs.get('max_length'):
            probe = self.tokenizer(text, add_special_tokens=kwargs.get('add_special_tokens', True), truncation=False)
            require(len(probe['input_ids']) <= kwargs['max_length'], 'Caption or lyrics exceed the trained conditioning window')
        return self.tokenizer(text, **kwargs)


def generation_parameters(request, seed, language):
    instrumental = request.get('mode', 'instrumental') == 'instrumental'
    # The pinned high-level orchestrator passes lyrics directly; its instrumental
    # dataclass flag alone never reaches either model's conditioning.
    lyrics = '[Instrumental]' if instrumental else request.get('lyrics', '')
    return dict(task_type='text2music', caption=request['prompt'], lyrics=lyrics,
        instrumental=instrumental, vocal_language=language,
        bpm=request.get('bpm'), keyscale=request.get('key_scale', ''), duration=float(request.get('duration', 30)),
        inference_steps=request.get('steps', 50), guidance_scale=request.get('guidance_scale', 7.0), seed=seed,
        use_cot_lyrics=False, **planner_flags(request),
        dcw_enabled=False, enable_normalization=False)  # DCW is Turbo-only upstream; over 50 SFT steps it buries the prompt


def component_manager(device, dtype, resident, observer, progress, state, reclaim):
    """Build the strict per-component context every ACE weight move routes through.

    offload/host: move the component to the device for its stage, back to cpu on exit ('offload' row).
    resident: the entry move is a no-op (weights already on the device); the exit never moves the
    component to cpu and records a 'retain' row instead."""
    active_contexts = Counter()

    @contextmanager
    def component_context(component, model):
        require(model is not None, 'Missing initialized component: ' + component)
        if active_contexts[component]:
            yield
            return
        active_contexts[component] += 1
        progress('conditioning_hints' if component == 'dit' and state.get('building_hints') else
                 {'planner': 'planning', 'dit': 'sampling', 'text_encoder': 'conditioning', 'vae': 'decoding'}[component])
        try:
            model.to(device=device, dtype=dtype)
            yield
        finally:
            active_contexts[component] -= 1
            if resident:
                observer.record(component, 'retain', model)
            else:
                model.to('cpu')
                observer.record(component, 'offload', model)
            reclaim()

    return component_context


def generation_receipt(request, runtime, seed, params, dtype_name, attention, language, duration, state):
    placement = runtime['placement']['resolved']
    placement_check = dict(runtime['placement'])
    if 'resident_loader' in runtime:
        # Measured by the streaming loader while it ran; the parent's placement block keeps its fixed key set.
        placement_check['loader'] = dict(runtime['resident_loader'])
    return {'seed': seed, 'steps': params.inference_steps, 'guidance_scale': params.guidance_scale,
            'mode': request.get('mode', 'instrumental'), 'vocal_language': language, 'duration': duration,
            'dtype': dtype_name, 'attention': attention, 'planner': '4B-pt', 'dcw_enabled': params.dcw_enabled,
            'batch_size': 1, 'cpu_offload': placement == 'offload', 'placement': placement,
            'placement_check': placement_check,
            'planner_released_before_dit_load': state['planner_released'],
            'precomputed_lm_hints_calls': state['precomputed_hint_calls'],
            'thinking': params.thinking, 'use_cot_metas': params.use_cot_metas, 'use_cot_caption': params.use_cot_caption,
            'use_cot_language': params.use_cot_language, 'planner_plan': state.get('plan'),
            'dit_conditioning': state.get('dit_conditioning'),
            'decode': {'latent_stride': 128, 'latent_overlap': 64, 'device_fallback': False},
            'threads': runtime['torch_num_threads'], 'interop_threads': runtime['torch_num_interop_threads']}


def bounded_value(value):
    """A planner or DiT value as recorded in the receipt: None, booleans, integers and finite floats as they are, anything
    else as text cut to PLAN_TEXT_LIMIT characters."""
    if value is None or type(value) in (bool, int) or type(value) is float and math.isfinite(value):
        return value
    return str(value)[:PLAN_TEXT_LIMIT]


def planner_plan(result):
    """The planner's parsed metadata (GenerationResult.extra_outputs['lm_metadata']) on the PLAN_KEYS, bounded; None
    when the pinned orchestrator returned none."""
    extra = getattr(result, 'extra_outputs', None)
    plan = extra.get('lm_metadata') if isinstance(extra, dict) else None
    return {key: bounded_value(plan[key]) for key in PLAN_KEYS if key in plan} if isinstance(plan, dict) else None


def dit_conditioning(kwargs):
    """What the pinned orchestrator handed the DiT (it passes keywords only): the caption the text encoder reads, the
    metadata it conditions on, and whether planner audio codes came with them."""
    codes = kwargs.get('audio_code_string') or ''
    codes = [codes] if isinstance(codes, str) else list(codes)
    return {'caption': bounded_value(kwargs.get('captions')), 'bpm': bounded_value(kwargs.get('bpm')),
            'key_scale': bounded_value(kwargs.get('key_scale')), 'time_signature': bounded_value(kwargs.get('time_signature')),
            'vocal_language': bounded_value(kwargs.get('vocal_language')),
            'audio_codes': any(isinstance(code, str) and code.strip() for code in codes)}


def generate(request, bundle, runtime, dtype_name, progress):
    # Imports are deliberately after request, visibility, bundle and source validation.
    import torch
    from diffusers import AutoencoderOobleck
    from transformers import Qwen3Config, Qwen3Model, Qwen3ForCausalLM, Qwen2TokenizerFast
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    from acestep.constants import VALID_LANGUAGES
    from acestep.models.common.configuration_acestep_v15 import AceStepConfig
    from acestep.models.xl_sft.modeling_acestep_v15_xl_base import AceStepConditionGenerationModel
    from pytorch_wavelets import DWT1DForward, DWT1DInverse  # noqa: F401; DCW must not silently no-op.

    device, dtype = runtime['resolved_device'], getattr(torch, dtype_name)
    attention = 'eager' if dtype_name == 'float16' else 'sdpa'
    observer = PlacementObserver(runtime, dtype_name)
    duration = float(request.get('duration', 30))
    language = request.get('vocal_language') or 'unknown'
    require(language in VALID_LANGUAGES, 'Vocal language is not in the pinned ACE language set')
    state = {'planner_released': False, 'precomputed_hint_calls': 0}
    emit_stage = progress
    resident = runtime['placement']['resolved'] == 'resident'
    load = {'local_files_only': True, 'use_safetensors': True, 'dtype': dtype,
            'low_cpu_mem_usage': True, 'attn_implementation': attention}
    if resident:
        # Resident weights never go through from_pretrained. The planner (Qwen3ForCausalLM), text encoder
        # (Qwen3Model) and DiT (AceStepConditionGenerationModel) are transformers PreTrainedModel subclasses
        # (transformers/models/qwen3/modeling_qwen3.py:281,336,429; acestep/models/xl_sft/
        # modeling_acestep_v15_xl_base.py:549,1612) without a from_pretrained override, so eager construction
        # under init_empty_weights + streamed assignment (build_resident) replaces the whole pinned loader: its
        # device_map path opens each shard with safe_open on the device (modeling_utils.py:689-691, :698), which
        # on Windows charges the whole shard file to process commit at open (measured OSError 1455 under a Job
        # cap); run r3, on that earlier loader under the Job cap, died natively with 0xC0000005 while the first
        # DiT shard loaded, and its cause is not proven (ROOT_R3_NATIVE_CRASH_TIMELINE_CORRECTION). No pinned
        # checkpoint key needs from_pretrained's renaming (_fix_state_dict_key_on_load :5169-5192 is applied
        # anyway; the VLM-only _checkpoint_conversion_mapping :4654-4657 does not apply to these classes), and
        # the planner's only missing key is the tied lm_head.weight (modeling_qwen3.py:430, tie_word_embeddings).
        # Checkpoint files come from the verified inventory (component_safetensors), never from a glob.
        files = component_safetensors(bundle)
        loader = runtime['resident_loader'] = new_loader_evidence()

    def reclaim():
        gc.collect()
        if device.startswith('cuda:'):
            torch.cuda.empty_cache()

    component_context = component_manager(device, dtype, resident, observer, progress, state, reclaim)

    class StrictPlanner(LLMHandler):
        @contextmanager
        def _load_model_context(self):
            require(self.llm is not None, 'Planner weights were released before planning completed')
            with component_context('planner', self.llm):
                yield

        def _load_pytorch_model(self, *_args, **_kwargs):
            raise RuntimeError('Stock planner loaders are disabled')

    planner = StrictPlanner()
    planner.device, planner.dtype = device, dtype
    # offload_to_cpu stays True in every placement. StrictPlanner replaces the stock
    # LLMHandler._load_model_context (acestep/llm_inference.py:4098-4155, whose stock body moves the LLM to
    # the device and back to cpu at :4145), so the flag no longer gates routing; its remaining pinned uses
    # are the disabled stock loader (llm_inference.py:401-410) and keeping the unused scoring model on the
    # host (llm_inference.py:4201-4214, 4239-4247). Setting it False would re-enable neither, but True keeps
    # every remaining stock path host-side. In resident mode the only weight moves are the no-op entry
    # move and the 'retain' observation in component_context.
    planner.offload_to_cpu, planner.llm_backend = True, 'pt'
    progress('loading_planner')
    planner_dir = bundle / 'planner'
    planner.llm_tokenizer = Qwen2TokenizerFast.from_pretrained(planner_dir, local_files_only=True)
    planner_config = Qwen3Config.from_pretrained(planner_dir, local_files_only=True)
    if resident:
        # The pinned planner checkpoint carries no lm_head.weight: it is tied to model.embed_tokens.weight
        # (Qwen3ForCausalLM._tied_weights_keys, modeling_qwen3.py:430) and bound by tie_weights() (modeling_utils.py:3010).
        planner.llm = build_resident(torch, 'planner', resident_constructor(torch, Qwen3ForCausalLM, planner_config, planner_dir, attention, dtype),
                                     files['planner'], device, dtype, {'lm_head.weight'} if planner_config.tie_word_embeddings else set(), loader)
    else:
        planner.llm = Qwen3ForCausalLM.from_pretrained(planner_dir, config=planner_config, **load).eval()
    planner.constrained_processor = MetadataConstrainedLogitsProcessor(
        tokenizer=planner.llm_tokenizer, enabled=True, debug=False, max_duration=math.ceil(duration))
    planner._lm_full_model_path = str(planner_dir)
    planner.llm_initialized = True
    observer.attach('planner', planner.llm)

    def release_planner():
        require('planner' in observer.executed, 'No actual planner execution was observed')
        observer.detach('planner')
        planner.llm = None
        planner._hf_model_for_scoring = None
        planner.llm_initialized = False
        state['planner_released'] = True
        reclaim()

    class StrictHandler(AceStepHandler):
        def initialize_service(self, *_args, **_kwargs):
            raise RuntimeError('Stock ACE initialization is disabled')

        def _get_vae_dtype(self, device=None):
            return dtype

        @contextmanager
        def _load_model_context(self, name):
            model = getattr(self, name, None)
            require(model is not None, 'Missing initialized component: ' + name)
            component = 'dit' if name == 'model' else name
            with component_context(component, model):
                if name == 'model':
                    self.silence_latent = self.silence_latent.to(device=device, dtype=dtype)
                yield

        def _parse_audio_code_string(self, text):
            values = [int(value) for value in re.findall(r'<\|audio_code_(\d+)\|>', text)]
            require(values and all(0 <= value <= 63999 for value in values), 'Planner emitted invalid or absent audio codes')
            return values

        def _decode_audio_codes_to_latents(self, text):
            state['building_hints'] = True
            try:
                result = super()._decode_audio_codes_to_latents(text)
            finally:
                state['building_hints'] = False
            require(result is not None and result.numel() and torch.isfinite(result).all().item(), 'Invalid planner conditioning hints')
            state['precomputed_hint_calls'] += 1
            return result

        @functools.wraps(AceStepHandler.generate_music)
        def generate_music(self, *args, **kwargs):
            require(not args, 'The pinned orchestrator passes the DiT conditioning by keyword')
            state['dit_conditioning'] = dit_conditioning(kwargs)
            release_planner()
            self.load_components()
            return super().generate_music(*args, **kwargs)

        def load_components(self):
            progress('loading_dit')
            if resident:
                # The DiT's only non-parameter tensors are non-persistent buffers (rotary inv_freq, FSQ tables) that
                # construction creates on the host and build_resident moves; no persistent buffer is expected.
                self.model = build_resident(torch, 'dit', resident_constructor(torch, AceStepConditionGenerationModel, self.config, bundle / 'dit', attention, dtype),
                                            files['dit'], device, dtype, set(), loader)
            else:
                self.model = AceStepConditionGenerationModel.from_pretrained(bundle / 'dit', config=self.config, **load).eval()
            self._sync_alignment_config()
            # Resident: the pinned silence tensor is deserialised straight onto the device; offload/host keep it on the host.
            silence = torch.load(bundle / 'dit/silence_latent.pt', map_location=device if resident else 'cpu', weights_only=True)
            require(isinstance(silence, torch.Tensor) and silence.ndim == 3 and silence.shape[0] == 1
                    and silence.shape[1] == 64 and 750 <= silence.shape[2] <= 250000
                    and torch.isfinite(silence).all().item(), 'Invalid restricted silence tensor')
            self.silence_latent = silence.transpose(1, 2).to(dtype=dtype)
            del silence
            observer.attach('dit', self.model, self.model.decoder)
            progress('loading_text_encoder')
            text_dir = bundle / 'core/Qwen3-Embedding-0.6B'
            text_config = Qwen3Config.from_pretrained(text_dir, local_files_only=True)
            if resident:
                # Qwen3Model has no output embeddings (its checkpoint keys are unprefixed: embed_tokens.weight, layers.*),
                # so tie_word_embeddings ties nothing and every key must be present.
                self.text_encoder = build_resident(torch, 'text_encoder', resident_constructor(torch, Qwen3Model, text_config, text_dir, attention, dtype),
                                                   files['text_encoder'], device, dtype, set(), loader)
            else:
                self.text_encoder = Qwen3Model.from_pretrained(text_dir, config=text_config, **load).eval()
            self.text_tokenizer = NoTruncationTokenizer(Qwen2TokenizerFast.from_pretrained(text_dir, local_files_only=True))
            observer.attach('text_encoder', self.text_encoder)
            progress('loading_vae')
            vae_dir = bundle / 'core/vae'
            if resident:
                # The pinned diffusers 0.37.1 from_pretrained stages this single-file checkpoint on the host whatever device_map
                # says (modeling_utils.py:1284 load_state_dict(...) -> model_loading_utils.py:159/:177 load_file(device='cpu')),
                # so the resident VAE keeps its meta-device construction (AutoencoderOobleck.from_config under
                # init_empty_weights) and its weights arrive through the same streaming loader as the other components.
                # AutoencoderOobleck (pinned autoencoder_oobleck.py) registers parameters only (torch.nn.utils.weight_norm
                # weight_g/weight_v, matching the checkpoint keys) and no buffers, so the complete-set rule covers every tensor.
                def construct_vae():
                    return AutoencoderOobleck.from_config(AutoencoderOobleck.load_config(vae_dir, local_files_only=True))
                self.vae = build_resident(torch, 'vae', construct_vae, files['vae'], device, dtype, set(), loader)
            else:
                # offload/host: the diffusers loader stages the VAE on the host, where these placements keep it between stages.
                self.vae = AutoencoderOobleck.from_pretrained(vae_dir, local_files_only=True,
                            use_safetensors=True, torch_dtype=dtype, low_cpu_mem_usage=True).eval()
            observer.attach('vae', self.vae, self.vae.decoder)

        def _decode_generate_music_pred_latents(self, pred_latents, progress, use_tiled_decode, time_costs):
            emit_stage('decoding')
            started = time.monotonic()
            with torch.inference_mode(), self._load_model_context('vae'):
                cpu_latents = pred_latents.detach().cpu()
                latents = pred_latents.transpose(1, 2).contiguous().to(device=device, dtype=dtype)
                # Use the audited chunk implementation directly, excluding its automatic CPU fallback wrappers.
                length = latents.shape[-1]
                stride, overlap = 128, 64
                wavs = self._tiled_decode_offload_cpu(latents, 1, length, stride, overlap, math.ceil(length / stride))
                require(torch.isfinite(wavs).all().item() and wavs.abs().sum().item() > 0, 'Invalid decoded music waveform')
                wavs = wavs.float()
                del latents
            time_costs['vae_decode_time_cost'] = time.monotonic() - started
            time_costs['total_time_cost'] += time_costs['vae_decode_time_cost']
            time_costs['offload_time_cost'] = self.current_offload_cost
            return wavs, cpu_latents, time_costs

    handler = StrictHandler()
    handler.device, handler.dtype = device, dtype
    handler.config = AceStepConfig.from_pretrained(bundle / 'dit', local_files_only=True)
    # Both flags stay True in every placement. StrictHandler replaces the stock
    # InitServiceOffloadContextMixin._load_model_context (acestep/core/generation/handler/
    # init_service_offload_context.py:12-90; offload_to_cpu gate :15, offload_dit_to_cpu gate :19), so
    # neither flag gates routing any more; every conditioning/hint/sampling/decode weight move still enters
    # component_context. offload_to_cpu=True additionally skips the stock VRAM pre-flight
    # (generate_music.py:132-137), whose estimate assumes the stock loader's last_init_params;
    # offload_dit_to_cpu is only read by the replaced context and the disabled stock loader
    # (init_service_loader.py:197-203, :215). The remaining stock weight moves are the LoRA-only decoder
    # fix-up (generate_music.py:59-104, gated at :326) and the stock decode CPU fallbacks: the low-VRAM
    # VAE-to-cpu switch (generate_music_decode.py:158-167), the MPS fallback (vae_decode.py:85-100) and the
    # OutOfMemoryError handlers that call _decode_on_cpu (vae_decode.py:103; vae_decode_chunks.py:50-53,
    # :64-70, :74-85). None runs here: _decode_generate_music_pred_latents below calls
    # _tiled_decode_offload_cpu (vae_decode_chunks.py:118-170) directly, which has no except clause, so a
    # device OOM fails the run instead of moving the VAE to the host; decode.device_fallback is therefore
    # False in every placement, and resident mode has no CPU-offload path before or during generation.
    handler.offload_to_cpu = handler.offload_dit_to_cpu = True
    handler.use_mlx_dit = handler.use_mlx_vae = False
    handler.mlx_decoder = handler.mlx_vae = None
    handler.compiled, handler.quantization, handler.batch_size = False, None, 1
    seed = request.get('seed', secrets.randbelow(2**32))
    params = GenerationParams(**generation_parameters(request, seed, language))
    config = GenerationConfig(batch_size=1, allow_lm_batch=False, use_random_seed=False, seeds=[seed],
                              lm_batch_chunk_size=1, audio_format='wav')
    try:
        with torch.inference_mode():
            result = generate_music(handler, planner, params, config, save_dir=None, progress=None)
        require(result.success and len(result.audios) == 1, 'ACE generation failed: ' + str(result.error or result.status_message))
        # thinking: the DiT decoded the planner's audio codes; thinking False: no planner audio codes reached the DiT.
        conditioning = state.get('dit_conditioning') or {}
        require(state['planner_released'] and conditioning.get('audio_codes') is bool(params.thinking)
                and (state['precomputed_hint_calls'] > 0) is bool(params.thinking), 'Planner conditioning route was not executed')
        require({'dit', 'text_encoder', 'vae', 'planner'} <= observer.executed, 'Required component execution was not observed')
        state['plan'] = planner_plan(result)
        audio = result.audios[0]
        return audio['tensor'].detach().cpu().float(), int(audio['sample_rate']), generation_receipt(
            request, runtime, seed, params, dtype_name, attention, language, duration, state)
    finally:
        for name in list(observer.handles):
            observer.detach(name)
        planner.llm = None
        handler.model = handler.vae = handler.text_encoder = handler.silence_latent = None
        reclaim()


def write_audio(audio, rate, path, request):
    import numpy as np
    data = audio.numpy() if hasattr(audio, 'numpy') else np.asarray(audio)
    contract = request['runtime_contract']
    require(rate == 48000 and data.ndim == 2 and data.shape[0] == 2 and data.shape[1] > 0,
            'Expected actual stereo 48000-Hz model output')
    require(np.isfinite(data).all() and np.any(data != 0), 'Non-finite or silent music output')
    # Remove only model duration padding, never time-stretch or duplicate samples.
    requested_frames = round(float(request.get('duration', 30)) * rate)
    original_frames = int(data.shape[1])
    require(original_frames >= requested_frames - 1920, 'Model returned materially shorter audio than requested')
    require(original_frames <= requested_frames + 48000, 'Model returned more than bounded duration padding')
    data = data[:, :requested_frames]
    frames = int(data.shape[1])
    require(frames / rate <= contract['max_audio_seconds'] and data.size <= contract['max_samples']
            and frames * 4 + 44 <= contract['max_output_bytes'], 'Generated output exceeds caller budget')
    peak = float(np.max(np.abs(data)))
    require(peak > 0, 'Silent music output')
    scale = max(1.0, peak)
    pcm = np.rint(np.clip(data / scale, -1, 1).T * 32767).astype('<i2')
    require(np.any(pcm != 0), 'Music became silent during PCM16 conversion')
    temporary = path.with_name(path.name + '.tmp')
    with wave.open(str(temporary), 'wb') as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())
    require(temporary.stat().st_size <= contract['max_output_bytes'], 'Encoded output exceeds byte budget')
    temporary.replace(path)
    with wave.open(str(path), 'rb') as stream:
        require(len(stream.readframes(stream.getnframes())) == frames * 4, 'Encoded WAV is incomplete')
    return {'sample_rate': rate, 'channels': 2, 'frames': frames, 'duration': frames / rate,
            'bytes': path.stat().st_size, 'sha256': digest(path), 'source_frames': original_frames,
            'duration_padding_frames_removed': max(0, original_frames - frames), 'peak_scale': scale,
            'rms_pcm16': float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'output', 'result', 'progress'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    request_path = ordinary(args.request.absolute())
    paths = [args.output.absolute(), args.result.absolute(), args.progress.absolute()]
    for path in paths:
        ordinary(path.parent, directory=True)
    require(len(set(paths + [request_path])) == 4 and not any(path.exists() for path in paths), 'Output paths must be fresh and distinct')
    output, result_path, progress_path = paths
    effective, evidence = {}, {'status': 'running', 'runtime': None, 'runner_sha256': digest(Path(__file__))}
    def progress(stage):
        atomic_json(progress_path, {'stage': stage, 'native_cancelled': False})
    try:
        # This is the trusted parent's private request file. Configured character
        # limits below govern text; a second byte ceiling would reject raised
        # bounds and JSON-escaped Unicode before those settings can be applied.
        request = validate_request(read_json(request_path, limit=None))
        progress('validating_bundle')
        bind_environment(request['runtime_contract'], result_path.parent)
        bundle, evidence['bundle'] = verify_bundle(request, progress)
        evidence['source'] = verify_source()
        import torch
        runtime, dtype = resolve_runtime(torch, request['runtime_contract'])
        evidence['runtime'] = runtime
        resolve_placement(torch, runtime, request['runtime_contract'], bundle, dtype)
        audio, rate, effective = generate(request, bundle, runtime, dtype, progress)
        progress('writing_audio')
        evidence['audio'] = write_audio(audio, rate, output, request)
        evidence['status'] = 'completed'
        evidence['quality_acceptance'] = 'not established by generation or PCM validation'
        atomic_json(result_path, {'effective': effective, 'evidence': evidence, 'native_format': 'pcm16-wav'})
        progress('completed')
    except BaseException as error:
        evidence.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        report_failure(result_path, progress, {'effective': effective, 'evidence': evidence, 'native_format': 'pcm16-wav'})
        raise


def os_alias(part, mode):
    """A POSIX symlink only root can change (owned by root inside a root-owned directory that group and others cannot
    write), as macOS /var and /tmp -> /private/... are. Windows links and any other link stay refused."""
    if os.name == 'nt' or not stat.S_ISLNK(mode.st_mode) or mode.st_uid != 0:
        return False
    try:
        parent = os.lstat(part.parent)
    except OSError:
        return False
    return parent.st_uid == 0 and stat.S_ISDIR(parent.st_mode) and not parent.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


if __name__ == '__main__':
    main()
