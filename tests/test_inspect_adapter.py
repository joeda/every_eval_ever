import contextlib
from pathlib import Path
import tempfile

from eval_converters.inspect.adapter import InspectAIAdapter
from eval_converters.inspect.utils import extract_model_info_from_model_path
from eval_types import (
    EvaluationLog,
    EvaluatorRelationship,
    SourceDataHf,
    SourceMetadata
)


def _load_eval(adapter, filepath, metadata_args):
    eval_path = Path(filepath)
    metadata_args = dict(metadata_args)
    metadata_args.setdefault("file_uuid", "test-file-uuid")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args['parent_eval_output_dir'] = tmpdir
        converted_eval = adapter.transform_from_file(eval_path, metadata_args=metadata_args)
    
    assert isinstance(converted_eval, EvaluationLog)
    assert isinstance(converted_eval.evaluation_results[0].source_data, SourceDataHf)

    assert isinstance(converted_eval.source_metadata, SourceMetadata)
    assert converted_eval.source_metadata.source_name == 'inspect_ai'
    assert converted_eval.source_metadata.source_type.value == 'evaluation_run'

    return converted_eval


def _extract_file_uuid_from_detailed_results(converted_eval: EvaluationLog) -> str:
    assert converted_eval.detailed_evaluation_results is not None
    stem = Path(converted_eval.detailed_evaluation_results.file_path).stem
    assert stem.endswith("_samples")
    return stem[: -len("_samples")]


def test_pubmedqa_eval():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/inspect/data_pubmedqa_gpt4o_mini.json', metadata_args)

    assert converted_eval.evaluation_timestamp == '1751553870.0'
    assert converted_eval.retrieved_timestamp is not None
    
    assert converted_eval.evaluation_results[0].source_data.dataset_name == 'pubmed_qa'
    assert converted_eval.evaluation_results[0].source_data.hf_repo == 'bigbio/pubmed_qa'
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 2

    assert converted_eval.model_info.name == 'openai/gpt-4o-mini-2024-07-18'
    assert converted_eval.model_info.id == 'openai/gpt-4o-mini-2024-07-18'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'openai'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'inspect_evals/pubmedqa - choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 1.0

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows == 2


def test_transform_without_metadata_args_uses_defaults(tmp_path, caplog):
    adapter = InspectAIAdapter()
    eval_file = (
        Path(__file__).resolve().parent
        / "data/inspect/data_pubmedqa_gpt4o_mini.json"
    )
    with contextlib.chdir(tmp_path):
        converted_eval = adapter.transform_from_file(
            eval_file.as_posix(),
            metadata_args=None,
        )

    assert isinstance(converted_eval, EvaluationLog)
    assert "Missing metadata_args['file_uuid']" in caplog.text
    assert converted_eval.source_metadata.source_organization_name == 'unknown'
    assert (
        converted_eval.source_metadata.evaluator_relationship
        == EvaluatorRelationship.third_party
    )
    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.total_rows == 2
    assert _extract_file_uuid_from_detailed_results(converted_eval) != "none"


def test_transform_directory_assigns_unique_file_uuid_per_log():
    adapter = InspectAIAdapter()
    fixture_dir = Path(__file__).resolve().parent / "data/inspect"

    with tempfile.TemporaryDirectory() as tmp_logs_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        tmp_logs_path = Path(tmp_logs_dir)
        fixture_targets = {
            "data_pubmedqa_gpt4o_mini.json": "2026-02-01T11-00-00+00-00_pubmedqa_test1.json",
            "data_arc_qwen.json": "2026-02-01T11-05-00+00-00_arc_test2.json",
        }
        for source_name, target_name in fixture_targets.items():
            source = fixture_dir / source_name
            target = tmp_logs_path / target_name
            target.write_bytes(source.read_bytes())

        converted_logs = adapter.transform_from_directory(
            tmp_logs_path,
            metadata_args={
                "source_organization_name": "TestOrg",
                "evaluator_relationship": EvaluatorRelationship.first_party,
                "parent_eval_output_dir": tmp_out_dir,
                "file_uuid": "shared-uuid",
            },
        )

    assert len(converted_logs) == 2

    uuids = {_extract_file_uuid_from_detailed_results(log) for log in converted_logs}
    assert "shared-uuid" not in uuids
    assert len(uuids) == 2


def test_transform_directory_uses_file_uuids_metadata_when_provided():
    adapter = InspectAIAdapter()
    fixture_dir = Path(__file__).resolve().parent / "data/inspect"
    expected_uuids = ["explicit-uuid-1", "explicit-uuid-2"]

    with tempfile.TemporaryDirectory() as tmp_logs_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        tmp_logs_path = Path(tmp_logs_dir)
        fixture_targets = {
            "data_pubmedqa_gpt4o_mini.json": "2026-02-01T11-00-00+00-00_pubmedqa_test1.json",
            "data_arc_qwen.json": "2026-02-01T11-05-00+00-00_arc_test2.json",
        }
        for source_name, target_name in fixture_targets.items():
            source = fixture_dir / source_name
            target = tmp_logs_path / target_name
            target.write_bytes(source.read_bytes())

        converted_logs = adapter.transform_from_directory(
            tmp_logs_path,
            metadata_args={
                "source_organization_name": "TestOrg",
                "evaluator_relationship": EvaluatorRelationship.first_party,
                "parent_eval_output_dir": tmp_out_dir,
                "file_uuids": expected_uuids,
            },
        )

    assert len(converted_logs) == 2
    uuids = {_extract_file_uuid_from_detailed_results(log) for log in converted_logs}
    assert uuids == set(expected_uuids)


def test_arc_sonnet_eval():
    adapter = InspectAIAdapter()

    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }
    converted_eval = _load_eval(adapter, 'tests/data/inspect/data_arc_sonnet.json', metadata_args)

    assert converted_eval.evaluation_timestamp == '1761000045.0'
    assert converted_eval.retrieved_timestamp is not None

    assert converted_eval.evaluation_results[0].source_data.dataset_name == 'ai2_arc'
    assert converted_eval.evaluation_results[0].source_data.hf_repo == 'allenai/ai2_arc'
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 5

    assert converted_eval.model_info.name == 'anthropic/claude-sonnet-4-20250514'
    assert converted_eval.model_info.id == 'anthropic/claude-sonnet-4-20250514'
    assert converted_eval.model_info.developer == 'anthropic'
    assert converted_eval.model_info.inference_platform == 'anthropic'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'arc_easy - choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 1.0

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows > 0


def test_arc_qwen_eval():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/inspect/data_arc_qwen.json', metadata_args)

    assert converted_eval.evaluation_timestamp == '1761001924.0'
    assert converted_eval.retrieved_timestamp is not None

    assert converted_eval.evaluation_results[0].source_data.dataset_name == 'ai2_arc'
    assert converted_eval.evaluation_results[0].source_data.hf_repo == 'allenai/ai2_arc'
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 3

    assert converted_eval.model_info.name == 'ollama/qwen2.5:0.5b'
    assert converted_eval.model_info.id == 'ollama/qwen2.5-0.5b'
    assert converted_eval.model_info.developer == 'ollama'
    assert converted_eval.model_info.inference_platform is None
    assert converted_eval.model_info.inference_engine.name == 'ollama'

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'arc_easy - choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 0.3333333333333333

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows > 0


def test_gaia_eval():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/inspect/2026-02-07T11-26-57+00-00_gaia_4V8zHbbRKpU5Yv2BMoBcjE.json', metadata_args)

    assert converted_eval.evaluation_timestamp is not None
    assert converted_eval.retrieved_timestamp is not None
    
    assert converted_eval.evaluation_results[0].source_data.dataset_name == 'GAIA'
    assert converted_eval.evaluation_results[0].source_data.hf_repo is not None
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) > 0

    assert converted_eval.model_info.name == 'openai/gpt-4.1-mini-2025-04-14'
    assert converted_eval.model_info.id == 'openai/gpt-4.1-mini-2025-04-14'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'openai'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert len(results) > 0
    assert results[0].evaluation_name == 'gaia - gaia_scorer'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score >= 0.0

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows > 0


def test_humaneval_eval():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/inspect/2026-02-24T11-23-20+00-00_humaneval_ENiBTeoXr2dbbNcDtpbVvq.json', metadata_args)
    assert converted_eval.detailed_evaluation_results is not None

def test_convert_model_path_to_standarized_model_ids():
    model_path_to_standarized_id_map = {
        "openai/gpt-4o-mini": "openai/gpt-4o-mini",
        "openai/azure/gpt-4o-mini": "openai/gpt-4o-mini",
        "anthropic/claude-sonnet-4-0": "anthropic/claude-sonnet-4-0",
        "anthropic/bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0": "anthropic/claude-3-5-sonnet@20241022",
        "anthropic/vertex/claude-3-5-sonnet-v2@20241022": "anthropic/claude-3-5-sonnet@20241022",
        "google/gemini-2.5-pro": "google/gemini-2.5-pro",
        "google/vertex/gemini-2.0-flash": "google/gemini-2.0-flash",
        "mistral/mistral-large-latest": "mistral/mistral-large-latest",
        "mistral/azure/Mistral-Large-2411": "mistral/Mistral-Large-2411",
        "openai-api/deepseek/deepseek-reasoner": "deepseek/deepseek-reasoner",
        "bedrock/meta.llama2-70b-chat-v1": "meta/llama2-70b-chat",
        "azureai/Llama-3.3-70B-Instruct": "azureai/Llama-3.3-70B-Instruct",
        "together/meta-llama/Meta-Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "groq/llama-3.1-70b-versatile": "meta-llama/llama-3.1-70b-versatile",
        "fireworks/accounts/fireworks/models/deepseek-r1-0528": "deepseek-ai/deepseek-r1-0528",
        "sambanova/DeepSeek-V1-0324": "deepseek-ai/DeepSeek-V1-0324",
        "cf/meta/llama-3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
        "perplexity/sonar": "perplexity/sonar",
        "hf/openai-community/gpt2": "openai-community/gpt2",
        "vllm/openai-community/gpt2": "openai-community/gpt2",
        "vllm/meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "sglang/meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "ollama/llama3.1": "ollama/llama3.1",
        "llama-cpp-python/llama3": "llama-cpp-python/llama3",
        "openrouter/gryphe/mythomax-l2-13b": "gryphe/mythomax-l2-13b",
        "hf-inference-providers/openai/gpt-oss-120b": "openai/gpt-oss-120b",
        "hf-inference-providers/openai/gpt-oss-120b:cerebras": "openai/gpt-oss-120b:cerebras",
    }

    for model_path, model_id in model_path_to_standarized_id_map.items():
        model_info = extract_model_info_from_model_path(model_path)
        assert model_info.id == model_id

_INSPECT_SHORTENED_EXPECTATIONS = {
    "DS-1000.json":                   ("DS-1000",                                   "accuracy",                                       0.0),
    "browse-comp.json":               ("a8e48c63e8a0202fcbde685141796329",           "inspect_evals/browse_comp_accuracy",              0.005529225908372828),
    "chembench.json":                 ("ChemBench",                                  "analytical_chemistry",                           0.2565789473684211),
    "class-eval.json":                ("ClassEval",                                  "mean",                                           0.82),
    "commonsense-qa.json":            ("commonsense_qa",                             "accuracy",                                       0.8),
    "compute-eval.json":              ("compute-eval",                               "accuracy",                                       0.6),
    "cybermetric-10000.json":         ("CyberMetric-10000",                          "accuracy",                                       1.0),
    "drop.json":                      ("drop",                                       "mean",                                           0.7846345044572627),
    "gaia.json":                      ("GAIA",                                       "accuracy",                                       0.0),
    "gpqa-diamond.json":              ("gpqa_diamond_74187e36ccadd6a06b1d98d13e064fed", "accuracy",                                    0.5),
    "gsm8k.json":                     ("gsm8k",                                      "accuracy",                                       1.0),
    "hellaswag.json":                 ("hellaswag",                                  "accuracy",                                       1.0),
    "humaneval.json":                 ("openai_humaneval",                           "accuracy",                                       0.8),
    "ifeval.json":                    ("IFEval",                                     "prompt_strict_acc",                              0.7208872458410351),
    "ifevalcode.json":                ("IfEvalCode-testset",                         "inspect_evals/overall_accuracy",                 0.043209876543209874),
    "lab-bench-cloning-scenarios.json": ("lab-bench",                               "accuracy",                                       0.15151515151515152),
    "lab-bench-dbqa.json":            ("lab-bench",                                  "accuracy",                                       0.026923076923076925),
    "lab-bench-litqa.json":           ("lab-bench",                                  "accuracy",                                       0.1457286432160804),
    "lab-bench-protocolqa.json":      ("lab-bench",                                  "accuracy",                                       0.28703703703703703),
    "lab-bench-seqqa.json":           ("lab-bench",                                  "accuracy",                                       0.3333333333333333),
    "lab-bench-suppqa.json":          ("lab-bench",                                  "accuracy",                                       0.036585365853658534),
    "lingoly-too.json":               ("LingOly-TOO",                                "inspect_evals/obfuscated_mean",                  0.06243145821552145),
    "lingoly.json":                   ("lingoly",                                    "inspect_evals/no_context_delta",                 0.0794351279788173),
    "livecodebench-pro.json":         ("livecodebench_pro",                          "accuracy",                                       0.0),
    "math.json":                      ("MATH-lighteval",                             "accuracy",                                       0.2),
    "mbpp.json":                      ("mbpp",                                       "accuracy",                                       1.0),
    "medqa.json":                     ("med_qa",                                     "accuracy",                                       0.4),
    "mind2web-sc.json":               ("mind2web_sc",                               "accuracy",                                       0.6),
    "mind2web.json":                  ("Multimodal-Mind2Web",                        "inspect_evals/element_accuracy",                 0.3896551724137931),
    "mmlu-0-shot.json":               ("mmlu",                                       "accuracy",                                       0.8),
    "mmlu-pro.json":                  ("MMLU-Pro",                                   "accuracy",                                       0.6),
    "musr.json":                      ("MuSR",                                       "accuracy",                                       0.4),
    "niah.json":                      ("niah",                                       "target_context_length_10000_accuracy",           5.0),
    "onet-m6.json":                   ("thai-onet-m6-exam",                          "accuracy",                                       0.4),
    "paws.json":                      ("paws",                                       "accuracy",                                       0.8),
    "personality-BFI.json":           ("bfi_dc398081fd7520dab7af7028dd830d84",       "Extraversion",                                   0.725),
    "personality-TRAIT.json":         ("personality_TRAIT",                          "Openness",                                       0.5955955955955956),
    "piqa.json":                      ("piqa",                                       "accuracy",                                       0.8),
    "pre-flight.json":                ("pre-flight-06",                              "accuracy",                                       0.8),
    "pubmedqa.json":                  ("PubMedQA",                                   "accuracy",                                       0.2),
    "race-h.json":                    ("race",                                       "accuracy",                                       0.8),
    "sad-facts-human-defaults.json":  ("sad_facts_human_defaults",                   "accuracy",                                       0.4),
    "sad-facts-llms.json":            ("sad_facts_llms",                             "accuracy",                                       0.8),
    "sad-influence.json":             ("sad_influence",                              "accuracy",                                       0.6),
    "sad-stages-full.json":           ("sad_stages_full",                            "accuracy",                                       0.2),
    "sad-stages-oversight.json":      ("sad_stages_oversight",                       "accuracy",                                       0.3),
    "scicode.json":                   ("problems_excl_dev",                          "inspect_evals/percentage_main_problems_solved",  0.0),
    "sec-qa-v1.json":                 ("secqa",                                      "accuracy",                                       1.0),
    "sec-qa-v2.json":                 ("secqa",                                      "accuracy",                                       1.0),
    "sevenllm-mcq-en.json":           ("sevenllm_c2388953e215061b1324c268e3c108a1", "accuracy",                                       0.0),
}


def test_many():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    fixture_dir = Path(__file__).parent / "data/inspect/inspect_shortened"
    for inspect_eval_path in sorted(fixture_dir.glob("*.json")):
        converted_eval = _load_eval(adapter, inspect_eval_path.resolve(), metadata_args)
        assert converted_eval.detailed_evaluation_results is not None

        assert converted_eval.model_info.id == 'Qwen/Qwen2.5-7B-Instruct'
        assert converted_eval.model_info.developer == 'Qwen'

        expected = _INSPECT_SHORTENED_EXPECTATIONS.get(inspect_eval_path.name)
        assert expected is not None, f"No expectations defined for {inspect_eval_path.name}"

        expected_dataset_name, expected_eval_description, expected_score = expected
        result = converted_eval.evaluation_results[0]
        assert result.source_data.dataset_name == expected_dataset_name, inspect_eval_path.name
        assert result.metric_config.evaluation_description == expected_eval_description, inspect_eval_path.name
        assert result.score_details.score == expected_score, inspect_eval_path.name
