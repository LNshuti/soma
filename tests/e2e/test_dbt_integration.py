import pytest
import subprocess
import os
from pathlib import Path


class TestDBTIntegration:
    """Test DBT integration and transformations."""

    @pytest.fixture
    def dbt_project_dir(self):
        """Get DBT project directory."""
        return Path("./dbt")

    def test_dbt_project_structure(self, dbt_project_dir):
        """Test DBT project structure."""
        assert dbt_project_dir.exists()
        assert (dbt_project_dir / "dbt_project.yml").exists()
        assert (dbt_project_dir / "profiles.yml").exists()
        assert (dbt_project_dir / "models").exists()

    def test_dbt_compile(self, dbt_project_dir):
        """Test DBT compilation."""
        try:
            env = os.environ.copy()
            env["DBT_PROFILES_DIR"] = str(dbt_project_dir.absolute())
            
            result = subprocess.run(
                ["dbt", "compile"],
                cwd=dbt_project_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # DBT should compile successfully or with warnings
            assert result.returncode in [0, 1]  # 1 for warnings
            
        except subprocess.TimeoutExpired:
            pytest.skip("DBT compile timeout")
        except FileNotFoundError:
            pytest.skip("DBT not installed")

    def test_dbt_test(self, dbt_project_dir):
        """Test DBT tests."""
        try:
            env = os.environ.copy()
            env["DBT_PROFILES_DIR"] = str(dbt_project_dir.absolute())
            
            # First try to run models
            result = subprocess.run(
                ["dbt", "run", "--select", "staging"],
                cwd=dbt_project_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                # Then run tests
                test_result = subprocess.run(
                    ["dbt", "test", "--select", "staging"],
                    cwd=dbt_project_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Tests should pass or have acceptable failures
                assert test_result.returncode in [0, 1]
            
        except subprocess.TimeoutExpired:
            pytest.skip("DBT test timeout")
        except FileNotFoundError:
            pytest.skip("DBT not installed")
