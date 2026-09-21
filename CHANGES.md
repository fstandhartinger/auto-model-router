# Changes

## 0.4.0 — delegation safety repair (21 Sep 2026)

Repairs the delegate MCP tools and installer added in 0.3.0 (commit 7a7bc31),
following an internal review. An independent (non-Anthropic) review of this
release is still outstanding.

**Behaviour changes you may notice**

- **Launched routes no longer inherit the launcher's environment.** They get an
  allowlist (`PATH`, `HOME`, locale, `TERM`, `TMPDIR`, XDG directories) plus
  what the route names with `env_pass`, `env_from` or `env`, and what the new
  `launcher.env_allow` adds for every route. A route that relied on an
  inherited API key needs `env_pass: [THAT_KEY]`. `launcher.inherit_env: true`
  restores the old behaviour. Applies to `route-run` and to delegated workers.
- **Worker tiers stay inside the policy.** `--tier cheap|strong` and the
  delegate tool's `tier` now choose only among routes the policy allows (tools,
  context window, quota, `--no-plans`). Before, `route-run --tier cheap` or
  `--tier strong` could launch a plan whose quota was closed or a route without
  tools, and recorded that as an operator `--route` override. Tier choices are
  now recorded as tier choices, in one routing pass (the classifier no longer
  runs twice).
- **Timeouts end the whole job.** The launcher runs the agent in its own process
  group and the delegate server runs the launcher in its own session. A timeout,
  `Ctrl-C` or `SIGTERM`/`SIGHUP` stops every process in them, and background
  processes left behind by a finished agent are stopped too
  (`auto_router/procs.py`).
- **Parallel workers get their own copies.** `delegate(parallel>1)` and
  `delegate_many` with more than one brief run each worker in a disposable copy
  of `cwd` and return a diff plus the copy's path; nothing is applied to `cwd`.
- **`cwd` is bounded.** Delegated work must stay inside
  `AUTO_ROUTER_DELEGATE_ROOT` or the directory the server started in; `/`,
  `$HOME` and its parents are refused as roots. `route-run` honours
  `launcher.cwd_root`.
- **Arguments are validated.** Malformed tool calls return a structured error
  and the server keeps running (it used to exit on `timeout_s: "soon"`); a
  string `tasks` is refused instead of being split into characters; ranges and
  the 32-brief limit are enforced server-side. Briefs follow `--`, so `--list`
  or `--help` as a brief is a task, not a launcher option.
- **Installer.** `install-delegation.sh` requires a full commit SHA and never
  installs a branch head; it no longer `pull`s, force-links over an existing
  command or replaces a differing install. `install-delegation.py` no longer
  deletes an existing skill directory, overwrites a differing MCP entry or
  removes and re-adds CLI entries without `--force` (with backups), writes JSON
  atomically, refuses JSONC, records `AUTO_ROUTER_CONFIG` only for an existing
  file you named, and writes the Cursor rule only with `--project`.

**Documentation**

- README: the opening paragraph no longer says the gateway runs in front of
  Codex, opencode or Cursor; the delegate section no longer claims actual cost is
  reported (it is always `null`).
- The delegation evaluation is labelled a derived re-tabulation of the 19 Sep
  runs, names the plan model (Claude Sonnet 5), says the orchestrator delegated
  in two of three tasks, and discloses that the Codex worker model is recorded
  as Qwen3.8 27B in one artifact and Kimi K3 in another.
- EXPERIMENTS: the delegation section had been inserted inside §15.4 under a
  duplicate "16" heading; it is now §18 and §15.4 is whole again.
