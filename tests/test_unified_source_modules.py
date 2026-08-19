"""Regression coverage for the feature-oriented public module layout."""

from importlib import import_module


def test_pipeline_features_are_exported_from_shared_modules() -> None:
    """Catch generation APIs being split back into stage-specific modules."""

    llm = import_module("community_shorts.llm")
    models = import_module("community_shorts.models")
    prompts = import_module("community_shorts.prompts")

    assert llm.FixtureLlmClient.model_name
    assert llm.FixtureGenerationLlmClient.model_name
    assert models.RawItem is not None
    assert models.GeneratedScript is not None
    assert callable(prompts.build_messages)
    assert callable(prompts.build_analysis_input)
