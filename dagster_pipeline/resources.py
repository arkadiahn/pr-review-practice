"""
Dagster resources for the Spotify ML pipeline.
Provides LakeFS client and model registry path configuration.
"""

import os
import lakefs_client
from lakefs_client.client import LakeFSClient
from dagster import ConfigurableResource


LAKEFS_ENDPOINT = os.environ.get("LAKEFS_ENDPOINT", "http://localhost:8000")
LAKEFS_ACCESS_KEY = os.environ.get("LAKEFS_ACCESS_KEY", "access_key")
LAKEFS_SECRET_KEY = os.environ.get("LAKEFS_SECRET_KEY", "secret_key")
LAKEFS_REPO = os.environ.get("LAKEFS_REPO", "spotify-pipeline")


def get_lakefs_client() -> LakeFSClient:
    """Create and return a configured LakeFS client."""
    configuration = lakefs_client.Configuration(
        host=LAKEFS_ENDPOINT,
        username=LAKEFS_ACCESS_KEY,
        password=LAKEFS_SECRET_KEY,
    )
    return LakeFSClient(configuration=configuration)


class LakeFSResource(ConfigurableResource):
    """
    Dagster resource wrapping the LakeFS client.
    Provides helpers for committing processed data to the data lake.
    """

    endpoint: str = LAKEFS_ENDPOINT
    access_key: str = LAKEFS_ACCESS_KEY
    secret_key: str = LAKEFS_SECRET_KEY
    repo: str = LAKEFS_REPO

    # BUG 4 (Integration — Wrong branch target, Easy):
    # Processed features should be committed to the "staging" branch so they
    # can be reviewed before merging to "main". Committing directly to "main"
    # bypasses the review step and can corrupt production data.
    target_branch: str = "main"

    def get_client(self) -> LakeFSClient:
        configuration = lakefs_client.Configuration(
            host=self.endpoint,
            username=self.access_key,
            password=self.secret_key,
        )
        return LakeFSClient(configuration=configuration)

    def commit_features(self, partition_key: str, message: str = "") -> dict:
        """
        Commit processed feature files for a given partition to the target branch.

        Args:
            partition_key: The partition identifier (e.g. "2024-01").
            message: Optional commit message.

        Returns:
            Commit metadata dict from LakeFS.
        """
        client = self.get_client()
        commit_message = message or f"Add processed features for partition {partition_key}"

        commit = client.commits_api.commit(
            repository=self.repo,
            branch=self.target_branch,
            commit_creation={
                "message": commit_message,
                "metadata": {"partition": partition_key, "pipeline": "spotify-features"},
            },
        )
        return commit


class ModelRegistryResource(ConfigurableResource):
    """Simple file-based model registry resource."""

    registry_path: str = os.environ.get("MODEL_REGISTRY_PATH", "/tmp/model_registry")

    def model_path(self, partition_key: str) -> str:
        return f"{self.registry_path}/model_{partition_key}.pkl"

    def metrics_path(self, partition_key: str) -> str:
        return f"{self.registry_path}/metrics_{partition_key}.json"
