"""Benchmark-only public /testbed path alias for the unchanged A file tools."""
from __future__ import annotations

from external_agents_eval.agents.mini_v62 import MiniPiFileEnvironment
from external_agents_eval.agents.mini_v63 import MiniV63Agent, MiniV63RoleRunner

PATH_PROTOCOL = "featurebench-testbed-to-worktree-v1"


class FeatureBenchPiFileEnvironment(MiniPiFileEnvironment):
    def execute(self, action):
        requested = action.get("arguments", {}).get("path")
        mapped = action
        if isinstance(requested, str) and (requested == "/testbed" or requested.startswith("/testbed/")):
            mapped = {**action, "arguments": {
                **action["arguments"], "path": self.cwd + requested[len("/testbed"):],
            }}
        result = super().execute(mapped)
        if mapped is not action:
            self.events[-1][1]["featurebench_requested_path"] = requested
            self.events[-1][1]["featurebench_path_protocol"] = PATH_PROTOCOL
        return result

    def serialize(self):
        value = super().serialize()
        value["info"]["environment"]["public_path_alias"] = PATH_PROTOCOL
        return value


class FeatureBenchV63RoleRunner(MiniV63RoleRunner):
    def _make_mini_environment(self, **kwargs):
        environment = super()._make_mini_environment(**kwargs)
        if isinstance(environment, MiniPiFileEnvironment):
            return FeatureBenchPiFileEnvironment(environment.bridge, environment.cwd)
        return environment


class FeatureBenchV63Agent(MiniV63Agent):
    def _make_role_runner(self, **kwargs):
        return FeatureBenchV63RoleRunner(**kwargs)
