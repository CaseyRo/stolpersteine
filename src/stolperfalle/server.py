"""FastMCP server for Stolperfalle — experiential knowledge capture."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fastmcp import Context, FastMCP
from mcp.types import Icon, ToolAnnotations
from starlette.responses import JSONResponse

from stolperfalle.auth import create_auth, create_bearer_only_auth
from stolperfalle.config import settings
from stolperfalle.models import (
    KUResponse,
    QueryResult,
    ReflectResult,
    StatusReport,
)

SERVER_INSTRUCTIONS = """\
Stolperfalle is an experiential knowledge base for AI coding agents. It stores
"Knowledge Units" (KUs) — generalizable pitfalls, workarounds, and
tool-recommendations discovered while working — so a *different* agent on a
*different* project can recall them later. The internal model is a superset of
Mozilla AI's `cq` schema (see the `cq://` resources for the strict schema and
the Stolperfalle extension registry).

Lifecycle: propose -> confirm -> flag, with query and reflect alongside.

- query(text, ...): READ. Always call this FIRST when you hit an error, an
  unfamiliar API, or a tech stack — recall before you rediscover. Returns
  ranked KUs (hybrid FTS5 + vector search).
- propose(...): WRITE. Capture a NEW learning. Requires summary/detail/action,
  a non-empty domains[] tag list, and a kind. Duplicate-detected automatically.
- confirm(ku_id): WRITE. You hit a KU's situation and its advice held — bump
  its confidence. NOT idempotent (each call increments confirmations).
- flag(ku_id, reason, ...): WRITE. A KU is wrong/stale/superseded/duplicate.
  `superseded` and `duplicate` ARCHIVE the KU (destructive); `incorrect`/
  `dangerous` mark it disputed; `stale` marks it stale.
- reflect(session_summary): READ. End-of-session helper — extracts ranked
  candidate KUs (pre-filled context_* + severity) you can pass straight to
  propose(). Slow when an LLM endpoint is configured; reports progress.
- status(debug=False): READ. Store health (counts, confidence, staleness).

Vocabulary:
- kind: pitfall | workaround | tool-recommendation. (`tool-gap-signal` is
  emergent-only; `gap-signal` is deprecated.)
- severity: low | medium | high | critical (ranking tiebreaker + decay floor).
- flag reason: stale | incorrect | superseded | dangerous | duplicate.

Prefer query() before any non-trivial task; prefer reflect()->propose() to
capture durable learnings at session end.
"""


def _build_auth():
    """Build auth provider for HTTP transport.

    Two modes:

    - **Behind CF MCP Portal (default)**: bearer-token-only. No OIDC config
      needed — the Portal handles upstream identity; we just validate the
      static bearer token that Claude Code / hooks present in the header.
    - **Standalone with OIDC**: set `CF_ACCESS_CLIENT_ID` +
      `CF_ACCESS_CLIENT_SECRET` to enable Cloudflare Access for SaaS OIDC
      flow alongside bearer auth (for browser OAuth clients).

    Stdio transport has no auth layer.
    """
    if settings.transport != "http":
        return None
    api_key = settings.ensure_api_key()

    # Bearer-only mode (the current CDiT fleet pattern — MCP Portal in front).
    if not settings.cf_access_client_secret.get_secret_value():
        logger = logging.getLogger(__name__)
        logger.info("CF Access OIDC disabled; bearer-token auth only")
        return create_bearer_only_auth(api_key=api_key, base_url=settings.base_url)

    return create_auth(
        api_key=api_key,
        base_url=settings.base_url,
        cf_access_config_url=settings.cf_access_config_url,
        cf_access_client_id=settings.cf_access_client_id,
        cf_access_client_secret=settings.cf_access_client_secret.get_secret_value(),
    )


mcp = FastMCP(
    "mcp-stolperfalle",
    instructions=SERVER_INSTRUCTIONS,
    auth=_build_auth(),
    icons=[
        Icon(
            src="data:image/png;base64,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",
            mimeType="image/png",
            sizes=["96x96"],
        ),
    ],
)


async def _ctx_info(ctx: Context | None, message: str) -> None:
    """Best-effort `ctx.info`. Never let progress/logging break a tool call.

    `ctx.info`/`ctx.report_progress` require an established MCP session; when a
    tool is invoked outside one (e.g. in-process calls), they raise. Logging is
    advisory, so swallow those failures rather than fail the underlying work.
    """
    if ctx is None:
        return
    try:
        await ctx.info(message)
    except Exception:
        logging.getLogger(__name__).debug("ctx.info unavailable", exc_info=True)


async def _ctx_progress(ctx: Context | None, progress: float, total: float) -> None:
    """Best-effort `ctx.report_progress` (see `_ctx_info`)."""
    if ctx is None:
        return
    try:
        await ctx.report_progress(progress=progress, total=total)
    except Exception:
        logging.getLogger(__name__).debug("ctx.report_progress unavailable", exc_info=True)


@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    """Unauthenticated liveness + schema_version probe.

    Returns 200 with a small JSON payload when the server can read its own
    store and report a schema version. Used by the Docker HEALTHCHECK and
    by any upstream (Komodo / Cloudflare / Tailscale) that wants a
    no-auth GET endpoint to verify the container is serving.

    Intentionally does NOT reveal `proposer_did` or KU content — just
    shape health (can we reach SQLite + run migrations).
    """
    try:
        from stolperfalle import migrations as m
        from stolperfalle.store import store
        db = store._get_db()
        m.current_version(db)  # exercise the DB read + migration check
        # Liveness only — no internal state (schema_version/transport) in the
        # unauthenticated body; those stay behind status(debug=True).
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse(
            {"status": "degraded", "error": type(e).__name__},
            status_code=503,
        )


async def _hook_authorize(request):
    """Shared auth + JSON-parse for /hook/* endpoints.

    Returns either `(payload_dict, None)` on success or `(None, JSONResponse)`
    with the appropriate 4xx/5xx error to return directly. Centralizes the
    bearer comparison and transport/key guards so all hook routes behave
    identically on the auth path.
    """
    import hmac

    if settings.transport != "http":
        return None, JSONResponse({"error": "http transport required"}, status_code=503)

    api_key = settings.mcp_stolperfalle_api_key.get_secret_value()
    if not api_key:
        return None, JSONResponse(
            {"error": "server has no API key configured"}, status_code=503
        )

    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return None, JSONResponse({"error": "bearer token required"}, status_code=401)
    provided = auth[len("Bearer "):].strip()
    if not hmac.compare_digest(provided, api_key):
        return None, JSONResponse({"error": "invalid bearer token"}, status_code=401)

    try:
        payload = await request.json()
    except Exception:
        return None, JSONResponse({"error": "invalid JSON body"}, status_code=400)

    if not isinstance(payload, dict):
        return None, JSONResponse(
            {"error": "JSON body must be an object"}, status_code=400
        )

    return payload, None


@mcp.custom_route("/hook/query", methods=["POST"])
async def hook_query(request):
    """Stdlib-friendly REST endpoint for Claude Code hook handlers.

    Hook handlers are plain Python scripts run by Claude Code; they shouldn't
    need to speak full MCP over streamable-HTTP. This endpoint is a thin
    JSON-in / JSON-out wrapper around `query()` with bearer-token auth.

    Request: `POST /hook/query`
    Headers: `Authorization: Bearer <MCP_STOLPERFALLE_API_KEY>`
    Body: `{"text": "...", "limit": 1, "confidence_min": 0.5}`
    Response: `{"results": [...], "count": N}` (same as MCP tool)
    """
    payload, err = await _hook_authorize(request)
    if err is not None:
        return err

    text = payload.get("text", "")
    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)
    # Untrusted JSON — a non-numeric limit or non-list domain used to raise an
    # uncaught ValueError/TypeError that Starlette surfaced as an opaque 500.
    try:
        limit = int(payload.get("limit", 1))
        confidence_min = float(payload.get("confidence_min", 0.5))
    except (TypeError, ValueError):
        return JSONResponse(
            {"error": "limit must be an integer and confidence_min a number"},
            status_code=400,
        )
    domain = payload.get("domain")
    if domain is not None and not isinstance(domain, list):
        return JSONResponse(
            {"error": "domain must be a list of strings"}, status_code=400
        )

    from stolperfalle.store import store
    result = await store.query(
        text=text, domain=domain, confidence_min=confidence_min, limit=limit
    )
    return JSONResponse(result)


@mcp.custom_route("/hook/reflect", methods=["POST"])
async def hook_reflect(request):
    """REST wrapper around `reflect()` for zero-dep hook subprocesses.

    Lets on_stop.py derive a session summary locally and POST it directly to
    the origin, bypassing both the MCP Portal and the Anthropic connector
    relay (whose WAF has been observed blocking reflect-sized payloads).

    Request: `POST /hook/reflect`
    Headers: `Authorization: Bearer <MCP_STOLPERFALLE_API_KEY>`
    Body: `{"session_summary": "..."}`
    Response: ranked candidate-KU dict (same shape as MCP `reflect` tool)
    """
    payload, err = await _hook_authorize(request)
    if err is not None:
        return err

    session_summary = payload.get("session_summary", "")
    if not isinstance(session_summary, str) or not session_summary.strip():
        return JSONResponse(
            {"error": "validation", "message": "session_summary (non-empty string) required"},
            status_code=400,
        )

    try:
        from stolperfalle.reflect import reflect_with_dedup
        from stolperfalle.store import store
        result = await reflect_with_dedup(session_summary, store=store)
    except Exception as e:
        logging.getLogger(__name__).exception("hook_reflect failed")
        return JSONResponse(
            {"error": "internal", "message": type(e).__name__}, status_code=500
        )
    return JSONResponse(result)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Query knowledge units",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def query(
    text: str,
    domain: list[str] | None = None,
    confidence_min: float = 0.3,
    limit: int = 10,
) -> QueryResult:
    """Search knowledge units by natural language, error signatures, or technology tags.

    Uses hybrid search combining FTS5 keyword relevance and vector cosine similarity.
    Severity acts as a tiebreaker at equal rank (critical > high > medium > low).

    Args:
        text: Natural-language query. Error messages, symptoms, tech stacks — all fine.
        domain: Optional tag list to filter by, e.g. ["swift", "xcode"]. Intersection
            with KU's domains[] field. Default None = no filter.
        confidence_min: Exclude KUs below this confidence score. Default 0.3. Raise
            to 0.5+ when you want only well-confirmed advice.
        limit: Max results (default 10).

    Returns:
        {"results": [KU dicts...], "count": N}. Each KU carries full v1 shape
        including context, evidence.severity, provenance, and owner_org.
    """
    from stolperfalle.store import store
    result = await store.query(
        text=text, domain=domain, confidence_min=confidence_min, limit=limit
    )
    return QueryResult.model_validate(result)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Propose a knowledge unit",
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
async def propose(
    summary: str,
    detail: str,
    action: str,
    domains: list[str],
    kind: str,
    context_languages: list[str] | None = None,
    context_frameworks: list[str] | None = None,
    context_environment: str | None = None,
    context_pattern: str | None = None,
    severity: str = "medium",
    ctx: Context | None = None,
) -> KUResponse:
    """Propose a new Knowledge Unit from a discovered insight.

    Args:
        summary: One-line description, max 280 chars. ("What happened.")
        detail: Why it matters, full context. Markdown allowed.
        action: Imperative — what to do when encountering this. Plain text; angle
            brackets are stripped from KU actions before they're injected into
            agent context, so inline HTML examples belong in `detail`, not here.
        domains: REQUIRED, non-empty. Technology tags, e.g. ["swift", "xcode"].
            Called `domains` on the wire (upstream CQ name).
        kind: One of `pitfall`, `workaround`, `tool-recommendation`.
            `gap-signal` is deprecated — tool gaps are detected automatically
            from query-miss patterns.
        context_languages: Optional, e.g. ["swift"]. Language tags.
        context_frameworks: Optional, e.g. ["swiftui", "combine"].
        context_environment: Optional, e.g. "xcode-16" — version scope
            (Stolperfalle extension; proposed upstream).
        context_pattern: Optional, e.g. "concurrency" — pattern tag.
        severity: Optional `low | medium | high | critical`, default `medium`.
            Default escalates decay floor to 0.2 for `critical` so critical
            KUs never fully decay (Stolperfalle extension; proposed upstream).

    Example:
        propose(
          summary="Xcode 16 requires explicit Swift 6 concurrency opt-in",
          detail="When targeting Swift 6 language mode, all sendable violations...",
          action="Add -strict-concurrency=complete to build settings.",
          domains=["swift", "xcode"],
          kind="pitfall",
          context_languages=["swift"],
          context_environment="xcode-16",
          context_pattern="concurrency",
          severity="high",
        )
    """
    from stolperfalle.store import store
    await _ctx_info(ctx, f"Proposing {kind} KU across domains={domains}")
    result = await store.propose(
        summary=summary,
        detail=detail,
        action=action,
        domains=domains,
        kind=kind,
        context_languages=context_languages,
        context_frameworks=context_frameworks,
        context_environment=context_environment,
        context_pattern=context_pattern,
        severity=severity,
    )
    if result.get("duplicate_of"):
        await _ctx_info(ctx, f"Duplicate of existing KU {result['duplicate_of']}; not re-created")
    else:
        await _ctx_info(ctx, f"Created KU {result['ku']['id']}")
    return KUResponse.model_validate(result)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Confirm a knowledge unit",
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
async def confirm(ku_id: str, ctx: Context | None = None) -> KUResponse:
    """Confirm an existing Knowledge Unit — increments confirmations and confidence.

    NOT idempotent: calling twice increments confirmations twice and changes the
    confidence score. Diversity-weighted: a confirmation from a new owner_org boosts
    more than one from an already-contributing org.

    Args:
        ku_id: The KU's id (format `ku_[0-9a-f]{32}`). If not found, raises an
            InvalidParams error with guidance to call query() first.
    """
    from stolperfalle.store import store
    result = await store.confirm(ku_id=ku_id)
    await _ctx_info(
        ctx, f"Confirmed {ku_id}: confidence now {result['ku']['evidence']['confidence']}"
    )
    return KUResponse.model_validate(result)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Flag a knowledge unit",
        readOnlyHint=False,
        idempotentHint=False,
        destructiveHint=True,
        openWorldHint=False,
    )
)
async def flag(
    ku_id: str,
    reason: str,
    detail: str = "",
    superseded_by: str | None = None,
    ctx: Context | None = None,
) -> KUResponse:
    """Flag a Knowledge Unit as stale, incorrect, superseded, dangerous, or duplicate.

    NOT idempotent: transitions state and updates the KU's flags/superseded_by.
    Potentially DESTRUCTIVE: `superseded` and `duplicate` archive the KU (it
    drops out of query results); `incorrect`/`dangerous` mark it disputed and
    cap its confidence; `stale` marks it stale.

    Args:
        ku_id: The KU's id.
        reason: One of `stale | incorrect | superseded | dangerous | duplicate`.
            `superseded` and `duplicate` require `superseded_by`.
            `dangerous` is a Stolperfalle extension (maps to `incorrect` on the wire).
        detail: Optional explanation — server-side only, never shown to querying agents.
        superseded_by: KU id of the replacing KU. Required when reason is `superseded`
            or `duplicate`.
    """
    from stolperfalle.store import store
    result = await store.flag(
        ku_id=ku_id, reason=reason, detail=detail, superseded_by=superseded_by
    )
    await _ctx_info(
        ctx, f"Flagged {ku_id} as {reason}; status now {result['ku']['status']}"
    )
    return KUResponse.model_validate(result)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Reflect on a session for candidate KUs",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,  # may call a configured external LLM endpoint
    )
)
async def reflect(session_summary: str, ctx: Context | None = None) -> ReflectResult:
    """Extract generalizable learnings from a session summary.

    Returns ranked candidate KUs scored by generalizability, each pre-filled
    with context_* and severity so you can pass them straight to propose()
    without re-reading docs.

    Hook channel vs tool channel: auto-injected hook hints (from PostToolUse,
    UserPromptSubmit) are rate-limited, sanitized nudges — they carry summary
    and action only. For the full KU (evidence, provenance, context,
    related, owner_org), call query() directly.
    """
    from stolperfalle.reflect import reflect_with_dedup
    from stolperfalle.store import store
    await _ctx_info(ctx, "Extracting candidate KUs from session summary")
    await _ctx_progress(ctx, 0, 1)
    result = await reflect_with_dedup(session_summary, store=store)
    await _ctx_progress(ctx, 1, 1)
    await _ctx_info(
        ctx,
        f"Extracted {len(result.get('candidates', []))} candidate(s) "
        f"via {result.get('method', 'none')}",
    )
    return ReflectResult.model_validate(result)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Report store health",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def status(debug: bool = False) -> StatusReport:
    """Report store health.

    Default (token-frugal): total, by_status, confidence_distribution, staleness,
    tool_gap_signals (grandfathered vs emergent counts).

    With debug=True: also includes schema_version, proposer_did, applied
    migrations, by_owner_org breakdown, recent_emergent KU ids, query_misses
    window size. Use for operator troubleshooting, not routine agent calls.
    """
    from stolperfalle.store import store
    result = await store.status(debug=debug)
    return StatusReport.model_validate(result)


# --- Resources: reference/context data agents can pull without a tool call ---


@mcp.resource(
    "stolperfalle://vocabulary",
    name="KU vocabulary",
    description="Valid kind / severity / status / flag-reason values for KUs, "
    "with selection guidance — ground propose()/flag() inputs against this.",
    mime_type="application/json",
)
def vocabulary_resource() -> dict:
    """Closed vocabularies the tools accept, derived from the live enums.

    Lets an agent pick a valid `kind`, `severity`, or flag `reason` before
    calling propose()/flag() instead of discovering them via a failed call.
    """
    from stolperfalle.models import KUKind, KUSeverity, KUStatus

    return {
        "kind": {
            "proposable": ["pitfall", "workaround", "tool-recommendation"],
            "emergent_only": ["tool-gap-signal"],
            "deprecated": ["gap-signal"],
            "guidance": {
                "pitfall": "A surprising/undocumented behavior that bites you.",
                "workaround": "A concrete fix or detour around a problem.",
                "tool-recommendation": "Prefer tool/library/approach X for Y.",
            },
            "all_values": [k.value for k in KUKind],
        },
        "severity": {
            "values": [s.value for s in KUSeverity],
            "default": "medium",
            "note": "Ranking tiebreaker + decay floor; 'critical' KUs never "
            "fully decay.",
        },
        "status": {
            "values": [s.value for s in KUStatus],
            "note": "Lifecycle state machine (Stolperfalle extension). Set by "
            "the server, not the caller.",
        },
        "flag_reason": {
            "values": ["stale", "incorrect", "superseded", "dangerous", "duplicate"],
            "archives": ["superseded", "duplicate"],
            "disputes": ["incorrect", "dangerous"],
            "requires_superseded_by": ["superseded", "duplicate"],
        },
    }


@mcp.resource(
    "cq://extensions",
    name="CQ extension registry",
    description="Registry of Stolperfalle fields that extend the upstream "
    "mozilla-ai/cq schema (first-class in rich output; carried as "
    "stolperfalle:* keys in the extensions slot in strict).",
    mime_type="text/markdown",
)
def cq_extensions_resource() -> str:
    """The `docs/cq-extensions.md` registry, or a built-in summary if absent.

    Distinguishes upstream core fields from Stolperfalle extensions so a
    consumer knows which `stolperfalle:*` keys to expect in the
    `extensions` slot of `to_cq_json_strict()` output.
    """
    doc = (
        Path(__file__).parent.parent.parent / "docs" / "cq-extensions.md"
    )
    try:
        return doc.read_text()
    except OSError:
        return (
            "# Stolperfalle CQ extensions\n\n"
            "Extension fields — first-class in rich output, emitted as "
            "`stolperfalle:*` keys in the `extensions` slot by "
            "`to_cq_json_strict()`:\n"
            "- `evidence.severity`, `evidence.contributing_orgs`\n"
            "- `context.environment`\n"
            "- top-level `kind`, `status`, `staleness_policy`, `related[]`, "
            "`owner_org`\n"
            "- `provenance.emergent`\n"
            "(`provenance.proposer_did` is emitted as upstream core "
            "`created_by`.)\n\n"
            "See docs/cq-extensions.md in the repo for the canonical registry."
        )


@mcp.resource(
    "cq://schema/knowledge-unit",
    name="CQ knowledge-unit schema",
    description="The vendored upstream mozilla-ai/cq JSON Schema that "
    "to_cq_json_strict() output validates against.",
    mime_type="application/json",
)
def cq_schema_resource() -> dict:
    """The vendored strict CQ schema, or a note if it isn't bundled."""
    import json

    schema_path = (
        Path(__file__).parent.parent.parent
        / "tests" / "fixtures" / "cq" / "knowledge_unit.json"
    )
    try:
        return json.loads(schema_path.read_text())
    except Exception as e:  # schema file not packaged in this deployment
        return {
            "error": "schema_unavailable",
            "detail": type(e).__name__,
            "note": "The vendored CQ schema is not bundled in this deployment. "
            "See the cq:// extensions resource for the field registry.",
        }


@mcp.resource(
    "stolperfalle://status",
    name="Store status snapshot",
    description="Read-only token-frugal store health snapshot (counts, "
    "confidence distribution, staleness) — same shape as status().",
    mime_type="application/json",
)
async def status_resource() -> dict:
    """Live store health as a resource, so agents can pull it as context
    without spending a tool call. Frugal (non-debug) shape only.
    """
    from stolperfalle.store import store

    return await store.status(debug=False)


# --- Prompts: guided multi-step workflows ---


@mcp.prompt(
    name="capture-learning",
    description="Guide an agent from a session summary through reflect() "
    "candidates to clean propose() calls for durable learnings.",
)
def capture_learning_prompt(session_summary: str = "") -> str:
    """Reflect -> propose capture workflow.

    Args:
        session_summary: What happened this session (errors hit, fixes found,
            tools that helped). Leave empty to have the agent summarize first.
    """
    summary_block = (
        f"Session summary to mine:\n\n{session_summary}\n"
        if session_summary.strip()
        else "First, write a 1-2 paragraph summary of what happened this "
        "session (errors hit, root causes, fixes, tools that helped).\n"
    )
    return (
        "You are capturing durable, transferable knowledge into the "
        "Stolperfalle knowledge base.\n\n"
        f"{summary_block}\n"
        "Steps:\n"
        "1. Call `reflect(session_summary=...)` to get ranked candidate KUs "
        "(each carries summary/detail/action, domains, kind, severity, "
        "context_*).\n"
        "2. Keep only candidates that would help a DIFFERENT agent on a "
        "DIFFERENT project — drop anything project-specific. Prefer "
        "generalizability_score >= 0.5.\n"
        "3. For each keeper, first `query(text=<summary>)` to avoid duplicating "
        "an existing KU. If a near-match exists and its advice held, "
        "`confirm(ku_id=...)` instead of proposing.\n"
        "4. Otherwise `propose(...)` using the candidate's pre-filled fields. "
        "Keep `action` imperative and free of angle brackets.\n"
        "5. Report which KUs you proposed/confirmed and which you skipped and "
        "why.\n\n"
        "Valid kinds: pitfall | workaround | tool-recommendation. "
        "Severity: low | medium | high | critical."
    )


@mcp.prompt(
    name="recall-before-task",
    description="Recall relevant prior pitfalls/workarounds before starting a "
    "task, and confirm/flag KUs based on what actually happened.",
)
def recall_before_task_prompt(task: str = "", tech: str = "") -> str:
    """Query-first workflow for the start of a task.

    Args:
        task: What you are about to do (e.g. "wire up Cloudflare Access OIDC").
        tech: Optional comma-separated tech tags to narrow recall
            (e.g. "cloudflare, oidc, fastmcp").
    """
    domains = [t.strip() for t in tech.split(",") if t.strip()]
    domain_hint = (
        f" Pass domain={domains} to narrow results." if domains else ""
    )
    task_line = task.strip() or "the task you are about to start"
    return (
        "Before doing the work, recall what past sessions already learned.\n\n"
        f"1. Call `query(text=\"{task_line}\")`.{domain_hint}\n"
        "2. Read the returned KUs (mind evidence.severity and confidence). "
        "Apply any that fit BEFORE you start.\n"
        "3. As you work: if a KU's advice held, `confirm(ku_id=...)`. If a KU "
        "is wrong or outdated, `flag(ku_id=..., reason=...)` "
        "(stale|incorrect|superseded|dangerous|duplicate).\n"
        "4. If you hit something none of the KUs covered, note it for an "
        "end-of-task `reflect()` -> `propose()` capture pass.\n"
    )


# --- CLI subcommands ---

def _cmd_migrate(args: argparse.Namespace) -> int:
    """Apply pending migrations to `CQ_LOCAL_DB_PATH` without starting the server."""
    from stolperfalle import migrations
    from stolperfalle.store import connect

    db_path = args.db_path or settings.cq_local_db_path

    # Scope check: refuse arbitrary paths outside configured DB parent dir.
    configured_parent = Path(settings.cq_local_db_path).resolve().parent
    if args.db_path:
        requested = Path(args.db_path).resolve()
        if configured_parent not in requested.parents and requested.parent != configured_parent:
            print(
                f"error: --db-path must be within {configured_parent}; got {requested.parent}",
                file=sys.stderr,
            )
            return 2

    conn = connect(db_path)
    from_version = migrations.current_version(conn)
    try:
        result = migrations.run(conn, db_path=db_path)
    except Exception as e:
        print(f"Migration failed: {e}", file=sys.stderr)
        return 1
    if not result.applied:
        print(f"already at version {result.to_version}")
        return 0
    print(f"Applied {len(result.applied)} migrations ({from_version} → {result.to_version}):")
    for m in result.applied:
        print(f"  - {m}")
    if result.snapshots:
        print(f"Pre-migration snapshots: {result.snapshots}")
    return 0


def _cmd_prune_backups(args: argparse.Namespace) -> int:
    """List or delete .bak-pre-v* files in the DB directory."""
    db_dir = Path(settings.cq_local_db_path).resolve().parent
    pattern = f"{Path(settings.cq_local_db_path).name}.bak-pre-v*"
    backups = sorted(db_dir.glob(pattern))
    if not backups:
        print("No pre-migration backup files found.")
        return 0

    if not args.confirm:
        print(f"Would delete {len(backups)} backup(s) (dry run — pass --confirm to delete):")
        for b in backups:
            size = b.stat().st_size
            print(f"  - {b} ({size:,} bytes)")
        return 0

    for b in backups:
        b.unlink()
        print(f"Deleted {b}")
    return 0


def _cmd_detect_emergent(args: argparse.Namespace) -> int:
    """Run emergent-signal detection manually."""
    if settings.stolperfalle_emergent_disabled or settings.emergent_detect_every_n == 0:
        print("emergent detection is disabled; "
              "set STOLPERFALLE_EMERGENT_DISABLED=false to run")
        return 0
    try:
        from stolperfalle.emergent import detect_emergent
        from stolperfalle.store import store
    except ImportError:
        print("emergent module not yet implemented", file=sys.stderr)
        return 2
    emitted = detect_emergent(store)
    print(f"Emitted {len(emitted)} emergent tool-gap-signal KU(s).")
    for ku_id in emitted:
        print(f"  - {ku_id}")
    return 0


def main() -> None:
    """Entry point for mcp-stolperfalle."""
    parser = argparse.ArgumentParser(prog="mcp-stolperfalle")
    sub = parser.add_subparsers(dest="cmd")

    p_migrate = sub.add_parser("migrate", help="Apply pending DB migrations and exit.")
    p_migrate.add_argument("--db-path", default=None,
                           help=f"Override DB path (default: {settings.cq_local_db_path})")
    p_migrate.set_defaults(func=_cmd_migrate)

    p_prune = sub.add_parser("prune-backups", help="List/delete pre-migration .bak files.")
    p_prune.add_argument("--confirm", action="store_true",
                         help="Actually delete (default is dry-run).")
    p_prune.set_defaults(func=_cmd_prune_backups)

    p_detect = sub.add_parser("detect-emergent",
                              help="Run emergent-signal aggregation manually.")
    p_detect.set_defaults(func=_cmd_detect_emergent)

    args = parser.parse_args()

    if getattr(args, "func", None):
        sys.exit(args.func(args))

    # Default: start the MCP server.
    # Pre-warm the store so migrations run before the first tool call.
    from stolperfalle.store import store
    store._get_db()

    if settings.transport == "http":
        mcp.run(
            transport="streamable-http",
            stateless_http=True,
            host=settings.host,
            port=settings.port,
            allowed_hosts=[
                h.strip() for h in settings.allowed_hosts.split(",") if h.strip()
            ],
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
