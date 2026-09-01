def test_old_and_new_import_paths_resolve_same_objects():
    from external_agents_eval.adapters.pi import PiAdapter as OldPiAdapter
    from external_agents_eval.agents.pi import PiAdapter
    from external_agents_eval.benchmarks.swebench.manifest import (
        TaskManifest as NewTaskManifest,
    )
    from external_agents_eval.manifest import TaskManifest

    assert OldPiAdapter is PiAdapter
    assert TaskManifest is NewTaskManifest
