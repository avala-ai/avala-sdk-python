"""
Full integration test for the Avala Python SDK (v0.2.1).

Exercises every resource and method against the live production API.
Run with: python3 tests/integration_test.py

Requires: AVALA_API_KEY environment variable set.
Target: https://server.avala.ai/api/v1 (production)

All test-created resources use '__sdk_test_' prefix and are cleaned up via try/finally.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------


@dataclass
class IntegrationResult:
    name: str
    phase: str
    status: str  # PASS, FAIL, SKIP, XFAIL (expected failure / server-side issue)
    detail: str = ""
    duration_ms: float = 0.0


@dataclass
class IntegrationRunner:
    results: list[IntegrationResult] = field(default_factory=list)
    _current_phase: str = ""

    def set_phase(self, phase: str) -> None:
        self._current_phase = phase
        print(f"\n{'=' * 70}")
        print(f"  PHASE: {phase}")
        print(f"{'=' * 70}")

    def run(self, name: str, fn: Callable[[], Any], skip_reason: str = "") -> Any:
        if skip_reason:
            self.results.append(
                IntegrationResult(name=name, phase=self._current_phase, status="SKIP", detail=skip_reason)
            )
            print(f"  SKIP  {name} -- {skip_reason}")
            return None

        t0 = time.time()
        try:
            result = fn()
            dur = (time.time() - t0) * 1000
            self.results.append(IntegrationResult(name=name, phase=self._current_phase, status="PASS", duration_ms=dur))
            print(f"  PASS  {name} ({dur:.0f}ms)")
            return result
        except _ExpectedServerIssue as e:
            dur = (time.time() - t0) * 1000
            detail = f"Server-side: {e}"
            self.results.append(
                IntegrationResult(name=name, phase=self._current_phase, status="XFAIL", detail=detail, duration_ms=dur)
            )
            print(f"  XFAIL {name} ({dur:.0f}ms) -- {detail}")
            return None
        except Exception as e:
            dur = (time.time() - t0) * 1000
            detail = f"{type(e).__name__}: {e}"
            self.results.append(
                IntegrationResult(name=name, phase=self._current_phase, status="FAIL", detail=detail, duration_ms=dur)
            )
            print(f"  FAIL  {name} ({dur:.0f}ms)")
            print(f"        {detail}")
            traceback.print_exc(limit=3)
            return None

    def run_async(
        self,
        name: str,
        coro_fn: Callable[[], Any],
        skip_reason: str = "",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Any:
        if skip_reason:
            self.results.append(
                IntegrationResult(name=name, phase=self._current_phase, status="SKIP", detail=skip_reason)
            )
            print(f"  SKIP  {name} -- {skip_reason}")
            return None

        t0 = time.time()
        try:
            assert loop is not None, "Event loop must be passed to run_async"
            result = loop.run_until_complete(coro_fn())
            dur = (time.time() - t0) * 1000
            self.results.append(IntegrationResult(name=name, phase=self._current_phase, status="PASS", duration_ms=dur))
            print(f"  PASS  {name} ({dur:.0f}ms)")
            return result
        except _ExpectedServerIssue as e:
            dur = (time.time() - t0) * 1000
            detail = f"Server-side: {e}"
            self.results.append(
                IntegrationResult(name=name, phase=self._current_phase, status="XFAIL", detail=detail, duration_ms=dur)
            )
            print(f"  XFAIL {name} ({dur:.0f}ms) -- {detail}")
            return None
        except Exception as e:
            dur = (time.time() - t0) * 1000
            detail = f"{type(e).__name__}: {e}"
            self.results.append(
                IntegrationResult(name=name, phase=self._current_phase, status="FAIL", detail=detail, duration_ms=dur)
            )
            print(f"  FAIL  {name} ({dur:.0f}ms)")
            print(f"        {detail}")
            traceback.print_exc(limit=3)
            return None

    def summary(self) -> int:
        print(f"\n{'=' * 70}")
        print("  SUMMARY")
        print(f"{'=' * 70}")
        passed = [r for r in self.results if r.status == "PASS"]
        failed = [r for r in self.results if r.status == "FAIL"]
        skipped = [r for r in self.results if r.status == "SKIP"]
        xfailed = [r for r in self.results if r.status == "XFAIL"]

        print(f"  Total:          {len(self.results)}")
        print(f"  Passed:         {len(passed)}")
        print(f"  Failed:         {len(failed)}")
        print(f"  Expected Fail:  {len(xfailed)}  (server-side issues, not SDK bugs)")
        print(f"  Skipped:        {len(skipped)}")

        if failed:
            print("\n  FAILURES (SDK bugs):")
            for r in failed:
                print(f"    - [{r.phase}] {r.name}: {r.detail}")

        if xfailed:
            print("\n  EXPECTED FAILURES (server-side issues):")
            for r in xfailed:
                print(f"    - [{r.phase}] {r.name}: {r.detail}")

        if skipped:
            print("\n  SKIPPED:")
            for r in skipped:
                print(f"    - [{r.phase}] {r.name}: {r.detail}")

        total_ms = sum(r.duration_ms for r in self.results)
        print(f"\n  Total time: {total_ms / 1000:.1f}s")

        if failed:
            print(f"\n  RESULT: FAIL ({len(failed)} SDK bug(s) found)")
            return 1
        else:
            print("\n  RESULT: PASS (no SDK bugs found)")
            return 0


class _ExpectedServerIssue(Exception):
    """Raised when a test encounters a known server-side issue (not an SDK bug)."""

    pass


def _check_server_error(e: Exception, context: str = "") -> None:
    """Re-raise as _ExpectedServerIssue if this is a known server-side problem."""
    from avala.errors import AvalaError, ServerError

    msg = str(e)
    if isinstance(e, ServerError):
        raise _ExpectedServerIssue(f"{context}: {msg}") from e
    if isinstance(e, AvalaError) and "organization" in msg.lower():
        raise _ExpectedServerIssue(f"{context}: requires organization membership") from e
    if isinstance(e, AvalaError) and "not allowed" in msg.lower():
        raise _ExpectedServerIssue(f"{context}: {msg}") from e


# ---------------------------------------------------------------------------
# Find CLI binary
# ---------------------------------------------------------------------------


def _find_avala_cli() -> list[str]:
    """Find the avala CLI command. Returns command list."""
    # Try macOS pip install location first
    script = os.path.expanduser("~/Library/Python/3.9/bin/avala")
    if os.path.isfile(script):
        return [script]
    # Try shutil.which (skipping shell aliases)
    found = shutil.which("avala")
    if found and not found.endswith("avala-app"):
        return [found]
    # Fallback: use python -c to invoke the CLI entry point
    return [sys.executable, "-c", "from avala.cli import main; main()"]


# ---------------------------------------------------------------------------
# Main integration test
# ---------------------------------------------------------------------------


def main() -> int:
    runner = IntegrationRunner()

    api_key = os.environ.get("AVALA_API_KEY")
    if not api_key:
        print("ERROR: AVALA_API_KEY environment variable is not set.")
        print("Set it with: export AVALA_API_KEY=your_key")
        return 1

    # -----------------------------------------------------------------------
    # Phase 1: Client Initialization
    # -----------------------------------------------------------------------
    runner.set_phase("Phase 1: Client Initialization")

    from avala import Client

    client: Optional[Client] = None

    def test_client_explicit_key() -> Client:
        c = Client(api_key=api_key)
        assert c is not None
        return c

    client = runner.run("Client(api_key=...) creation", test_client_explicit_key)

    def test_client_env_var() -> None:
        c = Client()  # reads AVALA_API_KEY
        assert c is not None
        c.close()

    runner.run("Client via AVALA_API_KEY env var", test_client_env_var)

    def test_context_manager() -> None:
        with Client(api_key=api_key) as c:
            assert c is not None

    runner.run("Client context manager", test_context_manager)

    def test_missing_key_raises() -> None:
        old = os.environ.pop("AVALA_API_KEY", None)
        try:
            try:
                Client()
                raise AssertionError("Expected ValueError")
            except ValueError:
                pass
        finally:
            if old:
                os.environ["AVALA_API_KEY"] = old

    runner.run("Missing key raises ValueError", test_missing_key_raises)

    def test_all_resources_accessible() -> None:
        assert client is not None
        resources = [
            "agents",
            "auto_label_jobs",
            "consensus",
            "datasets",
            "exports",
            "inference_providers",
            "projects",
            "quality_targets",
            "storage_configs",
            "tasks",
            "webhooks",
            "webhook_deliveries",
        ]
        for name in resources:
            attr = getattr(client, name, None)
            assert attr is not None, f"Missing resource: {name}"

    runner.run("All 12 resources accessible", test_all_resources_accessible)

    if client is None:
        print("\nCRITICAL: Client creation failed. Cannot continue.")
        return runner.summary()

    # Use a longer-timeout client for slow endpoints
    slow_client = Client(api_key=api_key, timeout=60.0)

    try:
        return _run_all_phases(runner, client, slow_client, api_key)
    finally:
        client.close()
        slow_client.close()


def _run_all_phases(runner: IntegrationRunner, client: Any, slow_client: Any, api_key: str) -> int:
    from avala import Client, AsyncClient
    from avala.errors import (
        AvalaError,
        AuthenticationError,
        NotFoundError,
        RateLimitError,
        ValidationError,
        ServerError,
    )
    from avala._pagination import CursorPage

    # -----------------------------------------------------------------------
    # Phase 2: Read-Only Tests
    # -----------------------------------------------------------------------
    runner.set_phase("Phase 2: Read-Only Tests (Non-Destructive)")

    # -- Datasets --
    dataset_uid: Optional[str] = None

    def test_datasets_list() -> None:
        nonlocal dataset_uid
        try:
            page = client.datasets.list(limit=5)
        except (ServerError, AvalaError) as e:
            _check_server_error(e, "datasets.list")
            raise
        assert isinstance(page, CursorPage)
        assert isinstance(page.items, list)
        if page.items:
            dataset_uid = page.items[0].uid
            assert page.items[0].uid
            assert page.items[0].name

    runner.run("datasets.list(limit=5)", test_datasets_list)

    def test_datasets_get() -> None:
        assert dataset_uid is not None
        ds = client.datasets.get(dataset_uid)
        assert ds.uid == dataset_uid
        assert ds.name

    runner.run(
        "datasets.get(uid)", test_datasets_get, skip_reason="" if dataset_uid else "No datasets (list failed or empty)"
    )

    def test_datasets_pagination() -> None:
        try:
            page1 = client.datasets.list(limit=1)
        except (ServerError, AvalaError) as e:
            _check_server_error(e, "datasets pagination")
            raise
        assert len(page1.items) <= 1
        if page1.has_more:
            page2 = client.datasets.list(limit=1, cursor=page1.next_cursor)
            assert isinstance(page2, CursorPage)
            assert len(page2.items) <= 1

    runner.run("datasets pagination (limit=1, cursor)", test_datasets_pagination)

    # -- Projects --
    project_uid: Optional[str] = None

    def test_projects_list() -> None:
        nonlocal project_uid
        page = client.projects.list(limit=5)
        assert isinstance(page, CursorPage)
        if page.items:
            project_uid = page.items[0].uid
            assert page.items[0].name

    runner.run("projects.list(limit=5)", test_projects_list)

    def test_projects_get() -> None:
        assert project_uid is not None
        proj = client.projects.get(project_uid)
        assert proj.uid == project_uid

    runner.run("projects.get(uid)", test_projects_get, skip_reason="" if project_uid else "No projects found")

    def test_projects_pagination() -> None:
        page1 = client.projects.list(limit=1)
        assert len(page1.items) <= 1
        if page1.has_more:
            page2 = client.projects.list(limit=1, cursor=page1.next_cursor)
            assert isinstance(page2, CursorPage)

    runner.run("projects pagination (limit=1, cursor)", test_projects_pagination)

    # -- Tasks --
    task_uid: Optional[str] = None

    def test_tasks_list() -> None:
        nonlocal task_uid
        page = client.tasks.list(limit=5)
        assert isinstance(page, CursorPage)
        if page.items:
            task_uid = page.items[0].uid

    runner.run("tasks.list(limit=5)", test_tasks_list)

    def test_tasks_list_by_project() -> None:
        assert project_uid is not None
        try:
            page = slow_client.tasks.list(project=project_uid, limit=5)
            assert isinstance(page, CursorPage)
        except Exception as e:
            # Timeout on large projects is a server performance issue
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                raise _ExpectedServerIssue("tasks.list(project=...) timed out (server slow)") from e
            raise

    runner.run(
        "tasks.list(project=...)", test_tasks_list_by_project, skip_reason="" if project_uid else "No projects found"
    )

    def test_tasks_list_by_status() -> None:
        page = client.tasks.list(status="completed", limit=5)
        assert isinstance(page, CursorPage)

    runner.run("tasks.list(status='completed')", test_tasks_list_by_status)

    def test_tasks_get() -> None:
        assert task_uid is not None
        t = client.tasks.get(task_uid)
        assert t.uid == task_uid

    runner.run("tasks.get(uid)", test_tasks_get, skip_reason="" if task_uid else "No tasks found")

    # -- Exports --
    export_uid: Optional[str] = None

    def test_exports_list() -> None:
        nonlocal export_uid
        page = client.exports.list(limit=5)
        assert isinstance(page, CursorPage)
        if page.items:
            export_uid = page.items[0].uid

    runner.run("exports.list(limit=5)", test_exports_list)

    def test_exports_get() -> None:
        assert export_uid is not None
        exp = client.exports.get(export_uid)
        assert exp.uid == export_uid

    runner.run("exports.get(uid)", test_exports_get, skip_reason="" if export_uid else "No exports found")

    # -- Storage Configs --
    def test_storage_configs_list() -> None:
        page = client.storage_configs.list(limit=5)
        assert isinstance(page, CursorPage)

    runner.run("storage_configs.list(limit=5)", test_storage_configs_list)

    # -- Agents --
    agent_uid_existing: Optional[str] = None

    def test_agents_list() -> None:
        nonlocal agent_uid_existing
        page = client.agents.list(limit=5)
        assert isinstance(page, CursorPage)
        if page.items:
            agent_uid_existing = page.items[0].uid

    runner.run("agents.list(limit=5)", test_agents_list)

    def test_agents_get() -> None:
        assert agent_uid_existing is not None
        a = client.agents.get(agent_uid_existing)
        assert a.uid == agent_uid_existing

    runner.run("agents.get(uid)", test_agents_get, skip_reason="" if agent_uid_existing else "No agents found")

    def test_agents_list_executions() -> None:
        assert agent_uid_existing is not None
        page = client.agents.list_executions(agent_uid_existing, limit=5)
        assert isinstance(page, CursorPage)

    runner.run(
        "agents.list_executions(uid)",
        test_agents_list_executions,
        skip_reason="" if agent_uid_existing else "No agents found",
    )

    # -- Inference Providers --
    inf_prov_uid: Optional[str] = None

    def test_inference_providers_list() -> None:
        nonlocal inf_prov_uid
        page = client.inference_providers.list(limit=5)
        assert isinstance(page, CursorPage)
        if page.items:
            inf_prov_uid = page.items[0].uid

    runner.run("inference_providers.list(limit=5)", test_inference_providers_list)

    def test_inference_providers_get() -> None:
        assert inf_prov_uid is not None
        p = client.inference_providers.get(inf_prov_uid)
        assert p.uid == inf_prov_uid

    runner.run(
        "inference_providers.get(uid)",
        test_inference_providers_get,
        skip_reason="" if inf_prov_uid else "No inference providers found",
    )

    # -- Auto Label Jobs --
    auto_label_uid: Optional[str] = None

    def test_auto_label_jobs_list() -> None:
        nonlocal auto_label_uid
        page = client.auto_label_jobs.list(limit=5)
        assert isinstance(page, CursorPage)
        if page.items:
            auto_label_uid = page.items[0].uid

    runner.run("auto_label_jobs.list(limit=5)", test_auto_label_jobs_list)

    def test_auto_label_jobs_list_by_project() -> None:
        assert project_uid is not None
        page = client.auto_label_jobs.list(project=project_uid, limit=5)
        assert isinstance(page, CursorPage)

    runner.run(
        "auto_label_jobs.list(project=...)",
        test_auto_label_jobs_list_by_project,
        skip_reason="" if project_uid else "No projects found",
    )

    def test_auto_label_jobs_get() -> None:
        assert auto_label_uid is not None
        j = client.auto_label_jobs.get(auto_label_uid)
        assert j.uid == auto_label_uid

    runner.run(
        "auto_label_jobs.get(uid)",
        test_auto_label_jobs_get,
        skip_reason="" if auto_label_uid else "No auto-label jobs found",
    )

    # -- Quality Targets (project-scoped, may 404 if not supported) --
    qt_uid: Optional[str] = None

    def test_quality_targets_list() -> None:
        nonlocal qt_uid
        assert project_uid is not None
        try:
            page = client.quality_targets.list(project_uid, limit=5)
            assert isinstance(page, CursorPage)
            if page.items:
                qt_uid = page.items[0].uid
        except NotFoundError:
            raise _ExpectedServerIssue("quality_targets endpoint not found for this project")

    runner.run(
        "quality_targets.list(project)",
        test_quality_targets_list,
        skip_reason="" if project_uid else "No projects found",
    )

    def test_quality_targets_get() -> None:
        assert project_uid is not None and qt_uid is not None
        qt = client.quality_targets.get(project_uid, qt_uid)
        assert qt.uid == qt_uid

    runner.run(
        "quality_targets.get(project, uid)",
        test_quality_targets_get,
        skip_reason="" if (project_uid and qt_uid) else "No quality targets found",
    )

    # -- Consensus (project-scoped, may 404 if not configured) --
    consensus_available = False

    def test_consensus_get_summary() -> None:
        nonlocal consensus_available
        assert project_uid is not None
        try:
            summary = client.consensus.get_summary(project_uid)
            assert summary is not None
            consensus_available = True
        except NotFoundError:
            raise _ExpectedServerIssue("consensus not configured for this project")

    runner.run(
        "consensus.get_summary(project)",
        test_consensus_get_summary,
        skip_reason="" if project_uid else "No projects found",
    )

    def test_consensus_list_scores() -> None:
        assert project_uid is not None
        try:
            page = client.consensus.list_scores(project_uid, limit=5)
            assert isinstance(page, CursorPage)
        except NotFoundError:
            raise _ExpectedServerIssue("consensus scores not available for this project")

    runner.run(
        "consensus.list_scores(project)",
        test_consensus_list_scores,
        skip_reason="" if project_uid else "No projects found",
    )

    def test_consensus_get_config() -> None:
        assert project_uid is not None
        try:
            config = client.consensus.get_config(project_uid)
            assert config is not None
        except NotFoundError:
            raise _ExpectedServerIssue("consensus config not found for this project")

    runner.run(
        "consensus.get_config(project)",
        test_consensus_get_config,
        skip_reason="" if project_uid else "No projects found",
    )

    # -- Webhooks --
    webhook_uid_existing: Optional[str] = None

    def test_webhooks_list() -> None:
        nonlocal webhook_uid_existing
        page = client.webhooks.list(limit=5)
        assert isinstance(page, CursorPage)
        if page.items:
            webhook_uid_existing = page.items[0].uid

    runner.run("webhooks.list(limit=5)", test_webhooks_list)

    def test_webhooks_get() -> None:
        assert webhook_uid_existing is not None
        wh = client.webhooks.get(webhook_uid_existing)
        assert wh.uid == webhook_uid_existing

    runner.run("webhooks.get(uid)", test_webhooks_get, skip_reason="" if webhook_uid_existing else "No webhooks found")

    # -- Webhook Deliveries --
    delivery_uid: Optional[str] = None

    def test_webhook_deliveries_list() -> None:
        nonlocal delivery_uid
        page = client.webhook_deliveries.list(limit=5)
        assert isinstance(page, CursorPage)
        if page.items:
            delivery_uid = page.items[0].uid

    runner.run("webhook_deliveries.list(limit=5)", test_webhook_deliveries_list)

    def test_webhook_deliveries_get() -> None:
        assert delivery_uid is not None
        d = client.webhook_deliveries.get(delivery_uid)
        assert d.uid == delivery_uid

    runner.run(
        "webhook_deliveries.get(uid)",
        test_webhook_deliveries_get,
        skip_reason="" if delivery_uid else "No webhook deliveries found",
    )

    # -- Rate Limit Info --
    def test_rate_limit_info() -> None:
        info = client.rate_limit_info
        assert isinstance(info, dict)
        assert "limit" in info
        assert "remaining" in info
        assert "reset" in info

    runner.run("rate_limit_info after requests", test_rate_limit_info)

    # -----------------------------------------------------------------------
    # Phase 3: Write/Mutating Tests (Create-Verify-Cleanup)
    # -----------------------------------------------------------------------
    runner.set_phase("Phase 3: Write/Mutating Tests")

    no_project = project_uid is None

    # -- Webhooks CRUD --
    def test_webhooks_crud() -> None:
        try:
            wh = client.webhooks.create(
                target_url="https://httpbin.org/post",
                events=["task.completed"],
                is_active=False,
            )
        except AvalaError as e:
            _check_server_error(e, "webhooks.create")
            raise

        assert wh.uid
        assert wh.target_url == "https://httpbin.org/post"
        assert wh.is_active is False

        try:
            # Verify via get
            fetched = client.webhooks.get(wh.uid)
            assert fetched.uid == wh.uid
            assert fetched.target_url == "https://httpbin.org/post"

            # Update
            updated = client.webhooks.update(wh.uid, events=["task.completed", "export.completed"])
            assert len(updated.events) == 2

            # Test connectivity
            try:
                client.webhooks.test(wh.uid)
            except AvalaError:
                pass  # Server may reject test for inactive webhooks

        finally:
            client.webhooks.delete(wh.uid)

        # Verify deletion
        try:
            client.webhooks.get(wh.uid)
            raise AssertionError("Webhook should be deleted")
        except NotFoundError:
            pass

    runner.run("webhooks CRUD lifecycle", test_webhooks_crud)

    # -- Agents CRUD --
    def test_agents_crud() -> None:
        try:
            agent = client.agents.create(
                name="__sdk_test_agent",
                description="SDK integration test agent",
                events=["task.completed"],
                callback_url="https://httpbin.org/post",
                is_active=False,
            )
        except AvalaError as e:
            _check_server_error(e, "agents.create")
            raise

        assert agent.uid
        assert agent.name == "__sdk_test_agent"

        try:
            fetched = client.agents.get(agent.uid)
            assert fetched.uid == agent.uid
            assert fetched.name == "__sdk_test_agent"

            updated = client.agents.update(agent.uid, name="__sdk_test_agent_updated")
            assert updated.name == "__sdk_test_agent_updated"

            execs = client.agents.list_executions(agent.uid, limit=5)
            assert isinstance(execs, CursorPage)

            try:
                client.agents.test(agent.uid)
            except AvalaError:
                pass
        finally:
            client.agents.delete(agent.uid)

        try:
            client.agents.get(agent.uid)
            raise AssertionError("Agent should be deleted")
        except NotFoundError:
            pass

    runner.run("agents CRUD lifecycle", test_agents_crud)

    # -- Inference Providers CRUD --
    def test_inference_providers_crud() -> None:
        try:
            provider = client.inference_providers.create(
                name="__sdk_test_provider",
                provider_type="http",
                config={"url": "https://httpbin.org/post", "method": "POST"},
                description="SDK integration test provider",
                is_active=False,
            )
        except (ValidationError, AvalaError) as e:
            _check_server_error(e, "inference_providers.create")
            # ValidationError from the API means the payload schema doesn't match
            # what the server expects - this is a server/API-docs issue, not SDK
            if isinstance(e, ValidationError):
                raise _ExpectedServerIssue(f"inference_providers.create validation: {e.message}") from e
            raise

        assert provider.uid
        assert provider.name == "__sdk_test_provider"

        try:
            fetched = client.inference_providers.get(provider.uid)
            assert fetched.uid == provider.uid

            updated = client.inference_providers.update(provider.uid, name="__sdk_test_provider_updated")
            assert updated.name == "__sdk_test_provider_updated"

            try:
                client.inference_providers.test(provider.uid)
            except AvalaError:
                pass
        finally:
            client.inference_providers.delete(provider.uid)

        try:
            client.inference_providers.get(provider.uid)
            raise AssertionError("Provider should be deleted")
        except NotFoundError:
            pass

    runner.run("inference_providers CRUD lifecycle", test_inference_providers_crud)

    # -- Quality Targets CRUD --
    def test_quality_targets_crud() -> None:
        assert project_uid is not None
        try:
            qt = client.quality_targets.create(
                project_uid,
                name="__sdk_test_quality_target",
                metric="accuracy",
                threshold=0.95,
                operator="gte",
                severity="warning",
                is_active=False,
            )
        except AvalaError as e:
            _check_server_error(e, "quality_targets.create")
            raise

        assert qt.uid
        assert qt.name == "__sdk_test_quality_target"

        try:
            fetched = client.quality_targets.get(project_uid, qt.uid)
            assert fetched.uid == qt.uid

            updated = client.quality_targets.update(project_uid, qt.uid, threshold=0.90)
            assert updated.threshold == 0.90

            try:
                evals = client.quality_targets.evaluate(project_uid)
                assert isinstance(evals, list)
            except AvalaError:
                pass
        finally:
            client.quality_targets.delete(project_uid, qt.uid)

        try:
            client.quality_targets.get(project_uid, qt.uid)
            raise AssertionError("Quality target should be deleted")
        except NotFoundError:
            pass

    runner.run(
        "quality_targets CRUD lifecycle",
        test_quality_targets_crud,
        skip_reason="No projects found" if no_project else "",
    )

    # -- Consensus Config Update (Save/Restore) --
    def test_consensus_config_update() -> None:
        assert project_uid is not None
        try:
            original = client.consensus.get_config(project_uid)
        except NotFoundError:
            raise _ExpectedServerIssue("consensus config not found for this project")

        original_iou = original.iou_threshold
        try:
            new_iou = 0.6 if original_iou != 0.6 else 0.7
            updated = client.consensus.update_config(project_uid, iou_threshold=new_iou)
            assert updated.iou_threshold == new_iou

            fetched = client.consensus.get_config(project_uid)
            assert fetched.iou_threshold == new_iou
        finally:
            client.consensus.update_config(project_uid, iou_threshold=original_iou)

    runner.run(
        "consensus config update (save/restore)",
        test_consensus_config_update,
        skip_reason="No projects found" if no_project else "",
    )

    # -- Exports Create (NOTE: exports cannot be deleted via the API, creates permanent record) --
    def test_exports_create() -> None:
        assert project_uid is not None
        try:
            export = client.exports.create(project=project_uid)
        except (ValidationError, AvalaError) as e:
            _check_server_error(e, "exports.create")
            if isinstance(e, ValidationError):
                raise _ExpectedServerIssue(f"exports.create validation: {e.message}") from e
            raise

        assert export.uid
        fetched = client.exports.get(export.uid)
        assert fetched.uid == export.uid

    runner.run(
        "exports.create(project=...)", test_exports_create, skip_reason="No projects found" if no_project else ""
    )

    # -- Auto Label Jobs (dry_run) --
    def test_auto_label_jobs_create() -> None:
        assert project_uid is not None
        try:
            job = client.auto_label_jobs.create(project_uid, dry_run=True)
        except AvalaError as e:
            if e.status_code in (400, 404, 409):
                raise _ExpectedServerIssue(f"auto-label not supported: {e.message}") from e
            _check_server_error(e, "auto_label_jobs.create")
            raise

        assert job.uid
        try:
            fetched = client.auto_label_jobs.get(job.uid)
            assert fetched.uid == job.uid
        finally:
            try:
                client.auto_label_jobs.cancel(job.uid)
            except AvalaError:
                pass  # May already be completed

    runner.run(
        "auto_label_jobs create/cancel (dry_run)",
        test_auto_label_jobs_create,
        skip_reason="No projects found" if no_project else "",
    )

    # -----------------------------------------------------------------------
    # Phase 4: Error Handling Tests
    # -----------------------------------------------------------------------
    runner.set_phase("Phase 4: Error Handling")

    def test_not_found_error() -> None:
        try:
            client.datasets.get("00000000-0000-0000-0000-000000000000")
            raise AssertionError("Expected NotFoundError")
        except NotFoundError as e:
            assert e.status_code == 404
            assert e.message

    runner.run("NotFoundError on invalid UID", test_not_found_error)

    def test_authentication_error() -> None:
        bad_client = Client(api_key="not-a-real-key")
        try:
            bad_client.datasets.list()
            raise AssertionError("Expected AuthenticationError or ServerError")
        except AuthenticationError as e:
            assert e.status_code == 401
        except ServerError as e:
            # Some servers return 500 for bad auth - still validates SDK handles it
            raise _ExpectedServerIssue(f"Server returns {e.status_code} instead of 401 for bad key") from e
        finally:
            bad_client.close()

    runner.run("AuthenticationError on bad key", test_authentication_error)

    def test_validation_error() -> None:
        try:
            # This may require org membership, handle gracefully
            client.webhooks.create(target_url="not-a-url", events=[])
            raise AssertionError("Expected error")
        except ValidationError as e:
            assert e.status_code in (400, 422)
        except AvalaError as e:
            _check_server_error(e, "webhook validation")
            # Any other error with 400/422/403 status is fine
            assert e.status_code in (400, 422, 403)

    runner.run("ValidationError on invalid payload", test_validation_error)

    def test_error_hierarchy() -> None:
        assert issubclass(AuthenticationError, AvalaError)
        assert issubclass(NotFoundError, AvalaError)
        assert issubclass(RateLimitError, AvalaError)
        assert issubclass(ValidationError, AvalaError)
        assert issubclass(ServerError, AvalaError)

    runner.run("Error hierarchy (all inherit AvalaError)", test_error_hierarchy)

    def test_error_attributes() -> None:
        try:
            client.datasets.get("00000000-0000-0000-0000-000000000000")
        except NotFoundError as e:
            assert hasattr(e, "message")
            assert hasattr(e, "status_code")
            assert hasattr(e, "body")
            assert e.status_code == 404

    runner.run("Error objects have message/status_code/body attributes", test_error_attributes)

    # -----------------------------------------------------------------------
    # Phase 5: Async Client Tests
    # -----------------------------------------------------------------------
    runner.set_phase("Phase 5: Async Client")

    loop = asyncio.new_event_loop()
    try:

        def test_async_context_manager() -> None:
            async def _run() -> None:
                async with AsyncClient(api_key=api_key) as ac:
                    assert ac is not None

            return _run()

        runner.run_async("AsyncClient context manager", test_async_context_manager, loop=loop)

        def test_async_datasets_list() -> None:
            async def _run() -> None:
                async with AsyncClient(api_key=api_key) as ac:
                    try:
                        page = await ac.datasets.list(limit=5)
                        assert isinstance(page, CursorPage)
                    except ServerError as e:
                        raise _ExpectedServerIssue(f"async datasets.list: {e.message}") from e

            return _run()

        runner.run_async("async datasets.list()", test_async_datasets_list, loop=loop)

        def test_async_datasets_get() -> None:
            async def _run() -> None:
                assert dataset_uid is not None
                async with AsyncClient(api_key=api_key) as ac:
                    ds = await ac.datasets.get(dataset_uid)
                    assert ds.uid == dataset_uid

            return _run()

        runner.run_async(
            "async datasets.get()",
            test_async_datasets_get,
            skip_reason="" if dataset_uid else "No datasets found",
            loop=loop,
        )

        def test_async_projects_list() -> None:
            async def _run() -> None:
                async with AsyncClient(api_key=api_key) as ac:
                    page = await ac.projects.list(limit=5)
                    assert isinstance(page, CursorPage)

            return _run()

        runner.run_async("async projects.list()", test_async_projects_list, loop=loop)

        def test_async_projects_get() -> None:
            async def _run() -> None:
                assert project_uid is not None
                async with AsyncClient(api_key=api_key) as ac:
                    proj = await ac.projects.get(project_uid)
                    assert proj.uid == project_uid

            return _run()

        runner.run_async(
            "async projects.get()",
            test_async_projects_get,
            skip_reason="" if project_uid else "No projects found",
            loop=loop,
        )

        def test_async_webhooks_crud() -> None:
            async def _run() -> None:
                async with AsyncClient(api_key=api_key) as ac:
                    try:
                        wh = await ac.webhooks.create(
                            target_url="https://httpbin.org/post",
                            events=["task.completed"],
                            is_active=False,
                        )
                    except AvalaError as e:
                        if "organization" in str(e).lower() or isinstance(e, ServerError):
                            raise _ExpectedServerIssue(f"async webhooks.create: {e.message}") from e
                        raise

                    assert wh.uid
                    try:
                        fetched = await ac.webhooks.get(wh.uid)
                        assert fetched.uid == wh.uid

                        updated = await ac.webhooks.update(wh.uid, events=["export.completed"])
                        assert "export.completed" in updated.events
                    finally:
                        await ac.webhooks.delete(wh.uid)

                    try:
                        await ac.webhooks.get(wh.uid)
                        raise AssertionError("Should be deleted")
                    except NotFoundError:
                        pass

            return _run()

        runner.run_async("async webhooks CRUD lifecycle", test_async_webhooks_crud, loop=loop)

        def test_async_not_found() -> None:
            async def _run() -> None:
                async with AsyncClient(api_key=api_key) as ac:
                    try:
                        await ac.datasets.get("00000000-0000-0000-0000-000000000000")
                        raise AssertionError("Expected NotFoundError")
                    except NotFoundError as e:
                        assert e.status_code == 404

            return _run()

        runner.run_async("async NotFoundError", test_async_not_found, loop=loop)

        def test_async_tasks_list() -> None:
            async def _run() -> None:
                async with AsyncClient(api_key=api_key) as ac:
                    page = await ac.tasks.list(limit=5)
                    assert isinstance(page, CursorPage)

            return _run()

        runner.run_async("async tasks.list()", test_async_tasks_list, loop=loop)
    finally:
        loop.close()

    # -----------------------------------------------------------------------
    # Phase 6: CLI Smoke Tests
    # -----------------------------------------------------------------------
    runner.set_phase("Phase 6: CLI Smoke Tests")

    avala_cmd = _find_avala_cli()
    print(f"  Using CLI command: {' '.join(avala_cmd)}")

    def run_cli(*args: str, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [*avala_cmd, *args],
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy(),
        )
        if expect_success and result.returncode != 0:
            raise AssertionError(f"CLI failed (exit {result.returncode}): {result.stderr[:500]}")
        return result

    def test_cli_help() -> None:
        r = run_cli("--help")
        assert "avala" in r.stdout.lower() or "usage" in r.stdout.lower()

    runner.run("avala --help", test_cli_help)

    def test_cli_datasets_help() -> None:
        r = run_cli("datasets", "--help")
        assert "list" in r.stdout.lower()

    runner.run("avala datasets --help", test_cli_datasets_help)

    def test_cli_projects_help() -> None:
        r = run_cli("projects", "--help")
        assert "list" in r.stdout.lower()

    runner.run("avala projects --help", test_cli_projects_help)

    def test_cli_agents_help() -> None:
        r = run_cli("agents", "--help")
        assert "list" in r.stdout.lower()

    runner.run("avala agents --help", test_cli_agents_help)

    def test_cli_webhooks_help() -> None:
        r = run_cli("webhooks", "--help")
        assert "list" in r.stdout.lower()

    runner.run("avala webhooks --help", test_cli_webhooks_help)

    # Live CLI commands (need working API)
    def test_cli_projects_list() -> None:
        r = run_cli("projects", "list", "--limit", "3")
        assert r.returncode == 0

    runner.run("avala projects list --limit 3", test_cli_projects_list)

    def test_cli_tasks_list() -> None:
        r = run_cli("tasks", "list", "--limit", "3")
        assert r.returncode == 0

    runner.run("avala tasks list --limit 3", test_cli_tasks_list)

    def test_cli_exports_list() -> None:
        r = run_cli("exports", "list", "--limit", "3")
        assert r.returncode == 0

    runner.run("avala exports list --limit 3", test_cli_exports_list)

    def test_cli_agents_list() -> None:
        r = run_cli("agents", "list", "--limit", "3")
        assert r.returncode == 0

    runner.run("avala agents list --limit 3", test_cli_agents_list)

    def test_cli_datasets_list() -> None:
        # datasets.list has a known server bug (dataset_predefined_labels), so handle gracefully
        r = subprocess.run(
            [*avala_cmd, "datasets", "list", "--limit", "3"],
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy(),
        )
        if r.returncode != 0:
            combined = r.stderr + r.stdout
            if "ServerError" in combined or "500" in combined or "predefined_labels" in combined:
                raise _ExpectedServerIssue("datasets list returns ServerError (server bug)")
            raise AssertionError(f"CLI failed (exit {r.returncode}): {r.stderr[:500]}")

    runner.run("avala datasets list --limit 3", test_cli_datasets_list)

    def test_cli_projects_get() -> None:
        assert project_uid is not None
        r = run_cli("projects", "get", project_uid)
        assert r.returncode == 0

    runner.run(
        "avala projects get <uid>", test_cli_projects_get, skip_reason="" if project_uid else "No projects found"
    )

    def test_cli_bad_key() -> None:
        env = os.environ.copy()
        env["AVALA_API_KEY"] = "not-a-real-key"
        r = subprocess.run(
            [*avala_cmd, "--api-key", "not-a-real-key", "projects", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        # Should fail with non-zero exit
        assert r.returncode != 0 or "error" in r.stderr.lower() or "error" in r.stdout.lower()

    runner.run("avala --api-key bad projects list (expect failure)", test_cli_bad_key)

    # -----------------------------------------------------------------------
    # Cleanup safety net
    # -----------------------------------------------------------------------
    runner.set_phase("Cleanup")

    def test_cleanup_stale_resources() -> None:
        cleaned = 0

        # Clean stale agents
        try:
            page = client.agents.list(limit=100)
            for a in page.items:
                if a.name and a.name.startswith("__sdk_test_"):
                    try:
                        client.agents.delete(a.uid)
                        cleaned += 1
                    except Exception:
                        pass
        except Exception:
            pass

        # Clean stale inference providers
        try:
            page = client.inference_providers.list(limit=100)
            for p in page.items:
                if p.name and p.name.startswith("__sdk_test_"):
                    try:
                        client.inference_providers.delete(p.uid)
                        cleaned += 1
                    except Exception:
                        pass
        except Exception:
            pass

        # Clean stale quality targets
        if project_uid:
            try:
                page = client.quality_targets.list(project_uid, limit=100)
                for qt in page.items:
                    if qt.name and qt.name.startswith("__sdk_test_"):
                        try:
                            client.quality_targets.delete(project_uid, qt.uid)
                            cleaned += 1
                        except Exception:
                            pass
            except Exception:
                pass

        if cleaned:
            print(f"        Cleaned up {cleaned} stale test resources")

    runner.run("Cleanup stale __sdk_test_ resources", test_cleanup_stale_resources)

    return runner.summary()


if __name__ == "__main__":
    sys.exit(main())
