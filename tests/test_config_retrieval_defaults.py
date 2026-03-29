def test_config_has_retrieval_defaults():
    from src.pipeline.config import Config
    config = Config()
    assert config.get('retrieval.use_metadata') is False
    weights = config.get('retrieval.weights')
    assert weights is not None
    # Updated to match retrieval_v1_stable in config.yaml
    assert weights['text'] == 0.5
    assert weights['metadata_query'] == 0.4
    assert weights['expanded'] == 0.1

    # Check metadata_bonus defaults (also from retrieval_v1_stable)
    metadata_bonus = config.get('retrieval.metadata_bonus')
    assert metadata_bonus is not None
    assert metadata_bonus['enabled'] is True
    assert metadata_bonus['threshold'] == 0.6
    assert metadata_bonus['boost_factor'] == 1.1
