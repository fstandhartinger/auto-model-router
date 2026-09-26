# Using the auto router *with* a Claude or ChatGPT plan: what vendor documentation says

Anthropic's gateway, compatibility, and legal pages and Codex's authentication and config
pages were rechecked on **26 September 2026**. Older quotes retain their individual retrieval
dates. This is a source-based product note, not legal advice; check current terms and your
organisation's policy for your own account.

## Current summary — 26 September 2026

| Design | What the published documentation establishes | What this router does |
|---|---|---|
| **A. Claude Code gateway pass-through** – `ANTHROPIC_BASE_URL` points to a local router, without a gateway credential | Anthropic documents that the saved subscription login remains active and plan limits and billing apply. It says non-Claude routing through gateways is unsupported. Its legal page restricts developers from collecting, storing, or intermediating Claude.ai credentials or session tokens. These statements do not settle how the restriction applies to each user's local setup. | The gateway is a separate opt-in. By default, subscription-authenticated traffic passes through to Anthropic unchanged. Use only with your own login on your own machine, and check your agreement and organisation policy. |
| **B. Claude Code switch mode** – the official client runs on the user's plan or on a separate provider credential | The vendor documents both client sign-in modes; in plan mode the unmodified client connects directly to Anthropic. | Available through `auto-router switch`; recommended when users do not want the router to receive subscription-authenticated traffic. |
| **C. `route-run`** – choose which official CLI runs a task | Running the vendor's own CLI uses its ordinary account limits and terms. | Available for Claude Code and Codex; the official CLI performs the plan-backed work. |
| **D. MCP delegation** – the official plan client calls a local tool for a sub-task | MCP is a documented integration mechanism. Users remain responsible for their account and data policies. | Available as an opt-in local tool; it does not extract a plan login. |
| **Codex gateway configuration** | Codex documents custom providers and ChatGPT authentication for proxy use. This router has not implemented or live-tested that gateway path. | The supported plan paths use the official Codex CLI through `route-run` or MCP delegation. |

Codex / ChatGPT plan: Codex documents custom providers and ChatGPT authentication for proxy
use. This repository has not implemented or live-tested a ChatGPT-authenticated Codex gateway,
and this note makes no broader legal conclusion. Its plan paths run the user's own official
Codex CLI through `route-run` or MCP delegation.

---

## 1. Anthropic

### 1.1 The plan login may sit behind a gateway (design A, forwarding half)

> `ANTHROPIC_BASE_URL` is the variable that points Claude Code at the gateway. Setting only that variable, without a gateway credential, doesn't replace the subscription. Requests still route through the gateway, but a saved claude.ai login remains the active credential, so its usage limits and billing apply. Gateways that pass this traffic on to Anthropic must forward the OAuth capability in `anthropic-beta`; see the request headers reference.

— <https://code.claude.com/docs/en/llm-gateway> § *Subscriptions and gateways* (retrieved 26 Sep 2026)

> `anthropic-beta` … When the developer authenticates with a claude.ai login, which is possible when `ANTHROPIC_BASE_URL` is set without a gateway credential variable, this header also carries an OAuth capability that the upstream requires, and stripping it fails those requests with `401`

— <https://code.claude.com/docs/en/llm-gateway-protocol#request-headers> (retrieved 26 Sep 2026)

So a local router that forwards a plan-authenticated request unchanged to `api.anthropic.com`
is a configuration Anthropic describes and tells gateway authors how to implement.

### 1.2 Answering a plan session with another model (design A, routing half)

> Any gateway that exposes a supported API format works. Anthropic doesn't endorse, maintain, or audit third-party gateway products, and doesn't support routing Claude Code to non-Claude models through any gateway.

— <https://code.claude.com/docs/en/llm-gateway> (retrieved 26 Sep 2026)

Anthropic's support statement does not settle the legal status of every local use. This
repository keeps non-Claude routing behind a separate opt-in, explains the vendor's support
and credential language, and recommends switch mode or MCP delegation.

### 1.3 The line that matters for a *published* tool

> **Developers** building products or services that interact with Claude's capabilities, including those using the Agent SDK, should use API key authentication through Claude Console or a supported cloud provider. Anthropic does not permit third-party developers to offer Claude.ai login into their own applications, or to route requests through Free, Pro, or Max plan credentials on behalf of their users. Moreover, developers may not collect, store, or intermediate Claude.ai credentials or session tokens — sign-in to a Claude account must complete through Anthropic's own flow.

> This does not restrict how customers provision and manage their own API keys or third-party inference provider credentials … Nor does it prevent an end user from signing in to the unmodified Claude Code binary with their own Claude subscription, including where a platform hosts Claude Code as described under *Can customers offer Claude Code in their products?* above.

> Anthropic reserves the right to take measures to enforce these restrictions and may do so without prior notice.

— <https://code.claude.com/docs/en/legal-and-compliance> § *Authentication and credential use* (retrieved 26 Sep 2026)

How this applies:

- **(a) personal use vs a product for others.** A developer running a gateway on their own machine,
  for their own login, is the gateway configuration Anthropic documents in §1.1. The same
  legal page restricts developers from intermediating Claude.ai credentials. Because a public
  router can be installed by others, this repository does not claim that local pass-through
  is legally cleared: it stays opt-in, is labelled for a user's own login on their own machine,
  and the page recommends design B.
- **(b) forwarding unchanged vs answering with another model.** Forwarding: documented in the
  gateway setup page. Answering: unsupported by Anthropic. Neither statement is a complete
  legal analysis (§1.2).
- **(c) switching between the two supported modes (design B).** Each half is documented on
  its own: with a gateway credential "the credential replaces the subscription login for that
  session, and the subscription's usage limits don't apply"
  (<https://code.claude.com/docs/en/llm-gateway>); without one, the unmodified client signed
  in with its own subscription is expressly not restricted (quote above). In B the plan side
  has **no router in its path at all** — the router never sees, forwards or stores the plan
  login — so the "intermediate credentials" sentence does not reach it. Moving the
  conversation uses two further documented features: `claude --resume <session>` and the
  `UserPromptSubmit` hook, of which the docs say `"block"` "prevents the prompt from being
  processed and erases it from context" (<https://code.claude.com/docs/en/hooks> § *UserPromptSubmit decision control*, retrieved 19 Sep 2026).

### 1.4 Automation and "ordinary use"

> Except when you are accessing our Services via an Anthropic API Key or where we otherwise explicitly permit it, to access the Services through automated or non-human means, whether through a bot, script, or otherwise.

— <https://www.anthropic.com/legal/consumer-terms> §3 (effective 8 Oct 2025; retrieved 19 Sep 2026)

> Claude Code usage is subject to the Anthropic Usage Policy. Advertised usage limits for Pro and Max plans assume ordinary, individual usage of Claude Code and the Agent SDK.

— <https://code.claude.com/docs/en/legal-and-compliance> § *Acceptable use* (retrieved 26 Sep 2026)

Claude Code — including its headless `-p` mode, hooks and `--resume` — is the explicitly
permitted way to use a plan by program. Every design above keeps the plan inside Claude
Code. None of them *adds* plan usage; A, B and D move work off the plan. "Ordinary,
individual usage" is still the ceiling: one person's own work, not a server farm.

> You may not share your Account login information, Anthropic API key, or Account credentials with anyone else or make your Account available to anyone else.

— <https://www.anthropic.com/legal/consumer-terms> §2 (retrieved 19 Sep 2026)

## 2. OpenAI (Codex / ChatGPT plan)

> **OpenAI authentication**: Set `requires_openai_auth = true` to use OpenAI authentication. You can then sign in with ChatGPT or an API key. This is useful when you access OpenAI models through an LLM proxy server. When `requires_openai_auth = true`, Codex ignores `env_key`.

> When you sign in with an API key, Codex uses standard API pricing instead of included ChatGPT plan credits.

— <https://learn.chatgpt.com/docs/auth> (the page `developers.openai.com/codex/auth` redirects to; retrieved 26 Sep 2026)

> You can also target a specific session ID with `codex exec resume <SESSION_ID>`.

— <https://learn.chatgpt.com/docs/non-interactive-mode> (retrieved 19 Sep 2026)

> `mcp_servers.<id>.default_tools_approval_mode` — auto | prompt | writes | approve — Default approval behavior for MCP tools on this server unless a per-tool override exists.

— <https://learn.chatgpt.com/docs/config-file/config-reference> (retrieved 26 Sep 2026)

OpenAI **Terms of Use** (effective 1 January 2026), read on 19 Sep 2026 in a normal browser
session (the page answers scripted requests with HTTP 403; it was opened once as an
ordinary page view, not worked around):

> You may not share your account credentials or make your account available to anyone else and are responsible for all activities that occur under your account.

> Automatically or programmatically extract data or Output (defined below).

> Interfere with or disrupt our Services, including circumvent any rate limits or restrictions or bypass any protective measures or safety mitigations we put on our Services.

— <https://openai.com/policies/row-terms-of-use/> (retrieved 19 Sep 2026)

How this applies: Codex is OpenAI's own client for programmatic use of the plan, so running
it (headless or not) is not "programmatically extracting Output". A proxy with ChatGPT
sign-in is documented. Answering some Codex requests with another model circumvents no limit
— it uses less of the plan. Credentials must not be shared, and the plan must not serve
anyone else. No OpenAI statement comparable to Anthropic's "doesn't support non-Claude
models" was found.

## 3. What the router therefore ships

| Mode | In the router | Default |
|---|---|---|
| A. intercepting proxy | `AUTO_ROUTER_SUBSCRIPTION_MODE=route_others` | **off**; separate opt-in; states Anthropic's support position and credential restrictions |
| B. switch mode | `python -m auto_router.switch` (new) | the recommended way to combine a plan with the router |
| C. job launcher | `route-run` / `python -m auto_router.launcher` (existing) | unchanged |
| D. delegate tool | `python -m auto_router.delegate` (new MCP server) | opt-in per client |

Hard rules kept in code (`auto_router/plan_auth.py`, `switch.py`): a plan credential goes to
`api.anthropic.com` or nowhere; in switch mode the plan side runs with **no**
`ANTHROPIC_BASE_URL` and no credential variable; the gateway never picks a plan route for a
request that does not carry the plan's own login.

## 4. Codex: the proxy path is not implemented here

A Codex "cheap mode" needs an OpenAI **Responses API** endpoint (`wire_api = "responses"` is
"the only supported value"). The router has none. The local LiteLLM Responses bridge was
tried on 19 Sep: Codex got a malformed stream ("OutputTextDelta without active item") with
one model and a reconnect loop with another. This router does not expose that Codex gateway;
`route-run` and MCP delegation call the official `codex exec` client and are covered by tests.

## 5. Why "unsupported" is not an empty word (measured 19 Sep 2026)

Two breaks found in one morning, both invisible until a real Claude Code session ran:

1. Claude Code 2.1.278 sends `role: "system"` messages inside `messages`; the free host
   rejected the translated request ("System message must be at the beginning").
2. A free model returned tool-call ids like `functions.Write:0`; the next turn that went to
   the plan failed with `400 … tool_use.id: String should match pattern '^[a-zA-Z0-9_-]+$'`
   — for the whole rest of the conversation.

Both are fixed in the router, with tests. The next one will arrive with a Claude Code release.

---

## Earlier reading (18 Sep 2026)

Historical source excerpts follow. The legal and product-status conclusions from 18 September
are superseded by the current 26 September summary above. In particular, the earlier use of
"allowed" and "not forbidden" was too categorical and must not be relied on.

### Using a flat-rate coding plan through a router — what the vendors actually allow

Read on 18 September 2026 from the vendors' own current documentation. Quotes are
verbatim; every one is followed by its link. Nothing here is legal advice, and a
documentation page can change the day after it is quoted — re-read the four
Anthropic pages and the two OpenAI ones before relying on this.

The short version:

| | allowed, and documented | not allowed |
|---|---|---|
| **Launch the vendor's own client** (`claude -p --model …`, `codex exec -m …`) and choose *which* client runs a job | yes, this is ordinary use of the client | — |
| **Put a gateway in front of Claude Code** with `ANTHROPIC_BASE_URL` and no gateway credential, forwarding to Anthropic | yes — Anthropic documents this case and says the plan's "usage limits and billing apply" | — |
| **Put a proxy in front of the Codex CLI** with `requires_openai_auth = true` | yes — OpenAI documents it as "useful when you access OpenAI models through an LLM proxy server" | — |
| **Send a turn to a cheaper provider** from behind that gateway, on your own API key | works, and the plan credential is never sent there | Anthropic "doesn't support" it — you are on your own when it breaks |
| **Take the plan's OAuth token out of the official client** and use it from your own code, another harness, or an SDK | — | explicitly not permitted |
| **Serve other people's traffic** from your personal plan | — | explicitly not permitted |

---

## 1. Anthropic: a gateway in front of Claude Code is a documented configuration

This is the passage that decides the whole question. From **Other LLM gateways**,
section *Subscriptions and gateways*:

> While a [gateway credential variable](https://code.claude.com/docs/en/llm-gateway-connect#set-the-credential-variable) or `apiKeyHelper` is active, a developer's claude.ai subscription isn't used: the credential replaces the subscription login for that session, and the subscription's usage limits don't apply. That traffic is billed per token to whoever owns the credential the gateway forwards, such as your organization's Anthropic Console account, or your Amazon Bedrock, Google Cloud's Agent Platform, or Microsoft Foundry account when the gateway routes there.
>
> `ANTHROPIC_BASE_URL` is the variable that points Claude Code at the gateway. **Setting only that variable, without a gateway credential, doesn't replace the subscription. Requests still route through the gateway, but a saved claude.ai login remains the active credential, so its usage limits and billing apply.** Gateways that pass this traffic on to Anthropic must forward the OAuth capability in `anthropic-beta`; see the request headers reference.

— <https://code.claude.com/docs/en/llm-gateway> (emphasis added)

The compatibility guide says the same thing from the gateway operator's side, in the
`anthropic-beta` row of the request-header table:

> Comma-separated capability values for the request. Forward the header verbatim; don't allowlist individual values, because the set changes with Claude Code releases. **When the developer authenticates with a claude.ai login, which is possible when `ANTHROPIC_BASE_URL` is set without a gateway credential variable, this header also carries an OAuth capability that the upstream requires, and stripping it fails those requests with `401`.**

— <https://code.claude.com/docs/en/llm-gateway-protocol#request-headers>

So: Claude Code, signed in with its own claude.ai login, talking to a gateway that
forwards its requests to Anthropic, is a configuration the gateway documentation
describes and tells gateway authors how to implement. The plan pays. The legal page's
separate restrictions on developer handling of subscription credentials still apply;
see the current summary above.

Two limits stated on the same page:

> Any gateway that exposes a supported API format works. **Anthropic doesn't endorse, maintain, or audit third-party gateway products, and doesn't support routing Claude Code to non-Claude models through any gateway.**

— <https://code.claude.com/docs/en/llm-gateway>

"Doesn't support" is a support statement, not a prohibition: sending a turn to a
cheaper provider from behind the gateway is something you may do and must then
maintain yourself. It is off by default in this repository
(`AUTO_ROUTER_SUBSCRIPTION_MODE`).

> The tradeoff is that the gateway becomes infrastructure your organization operates. Claude Code adds capabilities with each release, and a gateway that doesn't forward them breaks the corresponding features, so the gateway product needs to be kept updated as Claude Code evolves.

— <https://code.claude.com/docs/en/llm-gateway>

## 2. Anthropic: what is *not* allowed

From **Legal and compliance**, section *Authentication and credential use*:

> * **OAuth authentication** is intended exclusively for purchasers of Claude Free, Pro, Max, Team, and Enterprise subscription plans and is designed to support ordinary use of Claude Code and other native Anthropic applications.
> * **Developers** building products or services that interact with Claude's capabilities, including those using the Agent SDK, should use API key authentication through Claude Console or a supported cloud provider. **Anthropic does not permit third-party developers to offer Claude.ai login into their own applications, or to route requests through Free, Pro, or Max plan credentials on behalf of their users. Moreover, developers may not collect, store, or intermediate Claude.ai credentials or session tokens — sign-in to a Claude account must complete through Anthropic's own flow.**

> This does not restrict how customers provision and manage their own API keys or third-party inference provider credentials … **Nor does it prevent an end user from signing in to the unmodified Claude Code binary with their own Claude subscription**, including where a platform hosts Claude Code as described under *Can customers offer Claude Code in their products?* above.
>
> Anthropic reserves the right to take measures to enforce these restrictions and may do so without prior notice.

— <https://code.claude.com/docs/en/legal-and-compliance>

And from the section above it, for anyone shipping Claude Code inside a product:

> * **The Claude Code binary must not be modified.** Claude Code must be installed and run as published by Anthropic, and customers may not remove, disable, or restrict any authentication method built into it …
> * **Customers may not pay for, resell, or intermediate Claude usage on their end users' behalf.** Each end user must authenticate with their own Anthropic API key, Claude subscription plan credentials, or 3P inference provider credential …

— <https://code.claude.com/docs/en/legal-and-compliance>

Same page, on limits:

> Claude Code usage is subject to the Anthropic Usage Policy. **Advertised usage limits for Pro and Max plans assume ordinary, individual usage of Claude Code and the Agent SDK.**

— <https://code.claude.com/docs/en/legal-and-compliance>

The Consumer Terms add the general rule about automation, which is why "run the
official client" and "call the API with a plan token" are different things:

> Except when you are accessing our Services via an Anthropic API Key or where we otherwise explicitly permit it, to access the Services through automated or non-human means, whether through a bot, script, or otherwise. (§3.7)
>
> You may not share your Account login information, Anthropic API key, or Account credentials with anyone else or make your Account available to anyone else. (§2)

— <https://www.anthropic.com/legal/consumer-terms>

**What this repository does with that.** The gateway forwards a subscription request
unchanged to `api.anthropic.com` and nowhere else; the refusal is mechanical
(`auto_router/plan_auth.py`), not a comment. The login is never read from disk,
never stored, never logged, and never sent to any other host. Claude Code is run
unmodified, signed in through Anthropic's own flow, by the person who owns the plan,
for their own work. Turns the router sends to another provider are authenticated
with that provider's own key and carry no Anthropic credential at all.

**What it deliberately does not do:** serve anyone else's traffic from a personal
plan, offer Claude.ai login to third parties, or lift the token out of the client.
The header is forwarded in flight between two processes on one machine; it is not
collected, not stored, and not re-used.

**Rewriting the model is off by default.** Nothing in the documentation forbids a
gateway changing the `model` field, but a plan grants particular models, and a
gateway that silently upgrades every request would be asking the plan for something
the developer could not have selected themselves. `AUTO_ROUTER_REWRITE_MODEL` is
off, and on subscription traffic it is additionally bounded by an explicit list of
models the operator states their own plan includes. Choosing the model with
`claude --model …` when the job starts is the clean way to do this, and it is what
the launcher does.

## 3. OpenAI: the same shape, with the same split

The Codex CLI has an equivalent documented proxy path. From **Authentication**,
section *Alternative model providers*:

> When you define a custom model provider in your configuration file, you can choose one of these authentication methods:
>
> **OpenAI authentication**: Set `requires_openai_auth = true` to use OpenAI authentication. You can then sign in with ChatGPT or an API key. **This is useful when you access OpenAI models through an LLM proxy server.** When `requires_openai_auth = true`, Codex ignores `env_key`.
>
> **Environment variable authentication**: Set `env_key = "<ENV_VARIABLE_NAME>"` to use a provider-specific API key from the local environment variable named `<ENV_VARIABLE_NAME>`.

— <https://learn.chatgpt.com/docs/auth> (the page `developers.openai.com/codex/auth` now redirects to)

The configuration reference confirms the fields and the one wire protocol:

> `model_providers.<id>.base_url` — API base URL for the model provider.
> `model_providers.<id>.requires_openai_auth` — The provider uses OpenAI authentication (defaults to false).
> `model_providers.<id>.wire_api` — Protocol used by the provider. `responses` is the only supported value, and it is the default when omitted.

— <https://learn.chatgpt.com/docs/config-file/config-reference>

Which credential pays is stated plainly:

> When you sign in with an API key, Codex uses standard API pricing instead of included ChatGPT plan credits.
>
> Use API key authentication for programmatic Codex CLI workflows, such as CI/CD jobs.

— <https://learn.chatgpt.com/docs/auth>

And the plan will not serve arbitrary models: Codex rejects a non-OpenAI model under
ChatGPT-plan authentication with

> The … model is not supported when using Codex with a ChatGPT account.

— reported against `openai/codex` (<https://github.com/openai/codex/issues/45467>);
a client-side observation, not a policy document.

OpenAI's Terms of Use could not be quoted from the primary source for this file:
`openai.com/policies/...` answers automated requests with `HTTP 403`, and working
around a bot wall is out of bounds here. The two clauses that matter are the
familiar ones — no sharing of account credentials, and no automated extraction of
output except through the API — and they are consistent with the documentation
above: use the plan through Codex, use an API key for everything else. Read them
in a browser at <https://openai.com/policies/terms-of-use/> before relying on this
paragraph.

**What this repository does with that.** It launches `codex exec` as OpenAI
publishes it, with the plan's own sign-in, and clears `OPENAI_API_KEY` for the child
so a stray key cannot silently move the run onto per-token billing. It does **not**
implement a Responses-API gateway in front of Codex: that path is documented and
allowed, but it needs a full `responses` surface, and nothing here would have been
tested against it. The ChatGPT plan is therefore configured as a *launch-only*
route — it can never answer another client's HTTP request, and the router will not
pretend otherwise.

## 4. Correction to the earlier conclusion

An earlier round of this work concluded that a local proxy in front of Claude Code
was a grey area and recommended against it. That was wrong, and the quotes in
section 1 are why: Anthropic documents the configuration, states that the
subscription's limits and billing apply, and tells gateways which header to forward.
The part that earlier conclusion got right is the part that is still true — the
token must stay inside the official client's own request path, and a plan may not
serve anyone else.

## 5. Pacing is a terms question, not only a politeness one

"Advertised usage limits for Pro and Max plans assume ordinary, individual usage of
Claude Code and the Agent SDK." A router that fills a personal plan with background
jobs around the clock is not obviously ordinary individual usage, whatever the
mechanism. That is one reason the subscription tier here is paced rather than
maximised: it is priced by a shadow price that rises as the week fills, it stops at
a hard limit well below the plan's own, and it closes entirely when the usage
measurement is missing or stale (`auto_router/quota.py`). The measurement itself
comes from a command *you* provide and reports percentages only — the router never
reads a vendor token to ask how full a plan is.
