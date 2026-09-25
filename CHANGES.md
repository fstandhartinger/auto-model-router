# Changes

## Unreleased — copy symlinks and installer control-flow tests (22 Sep 2026)

Follow-up to 0.4.0 (commit b7bda45). Not independently reviewed; no live
worker or real installation was run.

- **A worker can no longer write through a symlink out of its copy.** 0.4.0
  copied symlinks as they were, so a worker editing a linked file in its
  disposable copy wrote to the link's target: another file in `cwd` or anywhere
  outside it. Per-worker copies now keep only relative links that stay inside
  the copy (as written and after following every hop); absolute links, `..`
  escapes and chains that end outside are left out and reported in the new
  `copy_skipped` field of the result. Entries are opened with `O_NOFOLLOW`
  relative to their parent directory, links are vetted on the finished private
  copy before a worker starts, and FIFOs, sockets and devices are skipped.
  Worker copies are now made from the vetted base copy rather than from `cwd`
  again, and copied files and directories are owner-writable.
- **Diffs of worker copies** compare a link by its target instead of the file
  it reaches, and no longer try to read a FIFO a worker created (the read would
  block the server).
- **Links to directories in worker copies are no longer invisible.** The diff
  walked the copy with `os.walk`, which files a symlink to a directory among
  the directories and never enters it, so such a link was left out of the diff
  altogether: a worker could add `home -> /home/you` or retarget an in-tree
  directory link to leave its copy, and the result reported no change; a copy
  whose only change was such a link was deleted as unchanged. Links to
  directories are now listed and compared by target like any other link, and
  each result's `changes.links_leaving_copy` names the added or retargeted
  links that lead out of the copy. (Found by an offline self-audit on 22 Sep;
  reproduced with a fake worker; no live worker was run.)
- **The diff of a worker copy no longer reads a huge file into memory.** Only
  `cwd` was capped (200 MB / 50,000 files) before copying; the diff then read
  every file a worker changed whole and built the full patch before cutting it
  to 60,000 characters. A worker writing a 64 MB text file made the server peak
  at 244.5 MiB of traced memory for a 60,000-character reply. Now a changed
  file over 1 MiB (`DIFF_FILE_LIMIT_BYTES`) is listed in the new
  `changes.not_diffed` field without being read past its first 1 MiB + 1 byte,
  and once the patch is over its cap the remaining files are listed there
  unread (same case: 1.0 MiB peak). The worker's copy itself, the number of
  listed paths, and files under ignored names (`.git`, `node_modules`, ...)
  remain uncapped or unreported, as before. (Self-audit 22 Sep; fake local
  worker only; no live worker was run.)
- **Stopping the server during parallel work** no longer starts queued briefs
  or leaves the copies behind. Before this change, on `SIGTERM` or `SIGHUP` the
  running workers were stopped, but the thread pool then started every brief
  still queued (`parallel` below the number of briefs), each one running to
  completion or its timeout. After that, the base and worker copies were left in
  `$TMPDIR/auto-router-delegate-*` with no reply naming them. Now the signal
  handler bars queued briefs before it stops the running ones. An interrupted
  `run_many` (the stop signal, `Ctrl-C` or a crash) stops every remaining worker
  and only then deletes the copies. It keeps them if a worker cannot be stopped
  within 30 s. The exit status is still 128 + the signal number. Found by an
  offline self-audit on 22 Sep and reproduced by signalling a real server that
  was running fake workers. No live worker was run.
- **An interrupted run with briefs still queued no longer waits 30 s and
  keeps its copies.** The interruption cancels the queued briefs. A cancelled
  brief never runs, but the wait for stopped workers (`concurrent.futures.wait`)
  counts it as done only once a pool thread has seen it, and none does. On
  `Ctrl-C` (`SIGINT`) this happened every time; on `SIGTERM`/`SIGHUP`, only
  when the queue was cancelled before a pool thread picked the brief up. The
  server then waited the full 30 s, judged a worker unstoppable, and left
  every copy in `$TMPDIR/auto-router-delegate-*`, although all workers had been
  stopped. Cancelled briefs are no longer waited for. The real-server signal
  test now covers `SIGINT` and `SIGHUP` as well as `SIGTERM`. Found by an offline
  self-audit on 22 Sep and reproduced with fake workers; no live worker was run.
- **A stop signal can no longer hang the delegate server or `route-run`.**
  The `SIGTERM`/`SIGHUP` handler ends running jobs through
  `procs.terminate_all()`, which takes the lock guarding the list of running
  jobs. Python runs the handler on the main thread, and the main thread takes
  that same lock itself: when it starts or finishes a job (a single-worker
  `delegate` call, the agent `route-run` starts) and when it stops jobs on the
  way out. A signal landing inside one of those short sections made the handler
  wait for a lock its own thread held, so the process hung until `SIGKILL`, and
  a second stop signal changed nothing. The lock is now reentrant. What stays
  open: a signal that arrives after a job's process has started but before it is
  entered in that list still misses it, and that job is left running. (Self-audit
  24 Sep; shown with the handler triggered from inside each locked section in a
  child process and fake `sleep` jobs; no live worker was run.)
- **A second `Ctrl-C` no longer cuts the cleanup after the first one short.**
  Neither the delegate server nor `route-run` handled `SIGINT`, so a second
  `Ctrl-C` raised a new `KeyboardInterrupt` wherever the cleanup of the first
  was, typically in the grace period between `SIGTERM` and `SIGKILL`. The
  server then skipped deleting the worker copies (`$TMPDIR/auto-router-delegate-*`),
  and a single-worker `delegate` call or `route-run` left an agent that ignores
  `SIGTERM` running unsupervised. A `SIGTERM`/`SIGHUP` after a `Ctrl-C` did the
  same to the server's copies. Both now handle `SIGINT`: the first `Ctrl-C` is
  still a `KeyboardInterrupt` (exit status unchanged), later ones are ignored
  until that cleanup is over, and in the server a later `SIGTERM`/`SIGHUP` is
  ignored too. A `Ctrl-C` the process was started to ignore stays ignored.
  Unchanged: `SIGKILL` still ends either at once and leaves what was not yet
  cleaned up; in `route-run`, a `SIGTERM`/`SIGHUP` after a `Ctrl-C` still
  interrupts the cleanup (its own handler then stops the agent). (Self-audit
  25 Sep; shown in child processes with fake workers that ignore `SIGTERM` and
  answer it with the second signal; no live worker was run.)
- **Worker copies a dead server left behind are reclaimed, once nothing can
  write to them.** A server ended by `SIGKILL`, or one that kept its copies
  because a worker could not be stopped within 30 s, left
  `$TMPDIR/auto-router-delegate-*` for good: nothing later noticed, reported or
  removed them. (Shown with a real server running two fake workers that write
  into their copies: after `SIGKILL` the copies and both workers remained, and a
  fresh server start changed nothing and said nothing.) Each run's scratch
  directory now holds an owner record (pid, process start time, boot id, pid
  namespace), locked by the server while it runs and removed when a result
  hands the copies over. At startup the server deletes such a directory only if
  its record names that path and comes from this boot and pid namespace, the
  owner is gone (no process with that pid and start time, lock free), and no
  process visible in `/proc` holds anything inside it, nor has any process
  whose references cannot be read started since that server did. Anything else
  (untagged, handed over, another boot or container, still in use, ambiguous)
  is kept and listed on stderr with the reason. The record is removed last,
  so an interrupted deletion is resumed by the next start. Not seen: a process
  holding nothing inside the copy that later writes into it by path (a
  `setsid()` descendant that changed directory). Unchanged: workers of a
  `SIGKILL`ed server keep running, unsupervised. (Self-audit 25 Sep; fake
  workers in Bubblewrap only; no live worker was run.)
- **A refused install no longer leaves part of the install behind.** The
  installer copied the skill before it checked the MCP entry or settings file,
  so an install refused because Claude Code or Codex already had an
  `auto-router-delegate` entry, or because opencode's settings were JSONC or
  held a different entry, still left the skill directory in place, with exit
  status 1. For Cursor with `--project`, `~/.cursor/mcp.json` was written
  before a differing project rule stopped the install. Every refusal is now
  checked before anything is written. A command that fails while the install
  is writing can still leave a partial install. Re-running an identical Claude
  Code or Codex install still stops at the existing entry (use `--force`): the
  installer does not parse those CLIs' output to tell whether the entry is
  current. Found by an offline self-audit on 22 Sep and reproduced against
  fake CLIs in a disposable HOME; no real installation was run.
- **Installer tests.** `tests/test_install_delegation_e2e.py` runs
  `install-delegation.sh` and `install-delegation.py` end to end in a
  disposable HOME against fake `git`, `python3 -m venv`, `pip` and agent CLIs
  that record their calls: normal installs for all four tools, and refusals for
  a dirty checkout, a commit that differs from the pin, a foreign command link,
  a missing config, JSONC settings and an existing MCP entry (replaced only with
  `--force`). This proves control flow only; compatibility with the real tools
  is untested. Adding these tests did not change the installer; the fix
  above did.

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
