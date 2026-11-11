"""Model conversations for players and the Storyteller (Claude and OpenAI).

Each seat is one append-only conversation: a frozen system prompt (shared by all
seats, so it caches across them), then alternating user turns (what happened since
the seat last acted, plus the current prompt) and the model's full replies,
including its reasoning. Nothing earlier is ever edited or re-sent in a different
form, so the model keeps its own reasoning across the whole game and every request
reads everything before the new turn from cache.

Every model gets the same per-turn output ceiling (`max_tokens`, thinking included).
A turn that hits the ceiling is retried once with the least thinking the model allows (none for adaptive
Claude models that allow it, else low effort; a 1K budget for older ones; low effort for OpenAI); a turn with no usable reply after four
attempts stops the game (SeatFailed), which then resumes from its last checkpoint.
"""
from __future__ import annotations
import asyncio
import http.client
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import anthropic
import ssl

try:  # Beaker tracing is a dev tool: current_trace() records nothing outside a Beaker capture
    from beaker.tracing import current_trace
except ImportError:
    current_trace = None
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

MODEL = 'claude-opus-5-5'


class _NoCall:
    """Stands in for a Beaker trace call when no capture is running."""

    def output(self, *args, **kwargs):
        pass

    def usage(self, *args, **kwargs):
        pass


@asynccontextmanager
async def traced_call(operation, provider, model, input_messages):
    """One model call as a Beaker trace span (nothing is recorded outside a Beaker capture)."""
    if current_trace is None:
        yield _NoCall()
    else:
        async with current_trace().model_call(operation=operation, provider=provider, model=model,
                                              input_messages=input_messages) as call:
            yield call


def _price(inp, out, read=None, write_5m=None, write_1h=None):
    """USD per million tokens. Claude cache writes are 1.25x (5 min) / 2x (1 hour) input, reads 0.1x."""
    return {'input': inp, 'read': read if read is not None else inp / 10, 'output': out,
            'write_5m': write_5m if write_5m is not None else inp * 1.25,
            'write_1h': write_1h if write_1h is not None else inp * 2}


def _openai(inp, out, explicit_cache=True):
    """OpenAI prices: cached input 10%; models with explicit cache breakpoints bill writes at 1.25x."""
    return _price(inp, out, read=inp / 10, write_5m=inp * 1.25 if explicit_cache else inp,
                  write_1h=inp * 1.25 if explicit_cache else inp)


PRICES = {  # USD per million tokens, list prices as of Sept 2026; Flex is half price
    'claude-opus-5-5': _price(4, 20, read=0.20),
    'claude-opus-5': _price(5, 25),
    'claude-fable-5-2': _price(8, 40, read=0.20),  # checked 2026-11-09
    'claude-fable-5-1': _price(10, 50, read=0.25),
    'claude-fable-5': _price(10, 50),
    'claude-opus-4-8': _price(5, 25),
    'claude-opus-4-7': _price(5, 25),
    'claude-opus-4-6': _price(5, 25),
    'claude-opus-4-5-20251101': _price(5, 25),
    'claude-sonnet-5-5': _price(2, 10),
    'claude-sonnet-5': _price(2, 10),
    'claude-sonnet-4-5-20250929': _price(3, 15),
    'claude-haiku-4-5': _price(1, 5),
    'gpt-6-astra': _price(10, 50, read=1, write_5m=12.5, write_1h=12.5),
    'gpt-6-luna': _price(2, 10, read=0.1, write_5m=2.5, write_1h=2.5),  # ASSUMED: copied from gpt-6.1-sol, not checked
    'gpt-6.1-sol': _price(2, 10, read=0.1, write_5m=2.5, write_1h=2.5),
    'gpt-6-sol': _price(2, 10, read=0.2, write_5m=2.5, write_1h=2.5),
    'gpt-5.6-sol': _price(4, 20, read=0.4, write_5m=5, write_1h=5),
    'gpt-5.5': _openai(5, 30, explicit_cache=False),
    'gpt-5.4': _openai(2.5, 15, explicit_cache=False),
    'gpt-5.2': _openai(1.75, 14, explicit_cache=False),
    'gpt-5.1': _openai(1.25, 10, explicit_cache=False),
    'gpt-5': _openai(1.25, 10, explicit_cache=False),
    'o1': _price(15, 60, read=7.5, write_5m=15, write_1h=15),
    'gpt-4o': _price(2.5, 10, read=1.25, write_5m=2.5, write_1h=2.5),
    # served only through OpenRouter (Amazon Bedrock); OpenRouter's reported cost is used when present
    'openrouter:anthropic/claude-opus-4.1': _price(15, 75),
    'openrouter:anthropic/claude-sonnet-4': _price(3, 15),
    'o3': _openai(2, 8, explicit_cache=False),
    # OpenRouter list prices checked 2026-11-11; where it reports a call's cost, that is used instead
    'gpt-6.1-terra': _price(1, 5, read=0.05, write_5m=1.25, write_1h=1.25),  # ASSUMED: cache write at 1.25x input
    'openrouter:z-ai/glm-5.3-flash': _price(0.0399, 0.0784, read=0.03, write_5m=0.0399, write_1h=0.0399),
    'openrouter:google/gemini-3.8-flash': _price(0.825, 4.125, read=0.0825, write_5m=0.825, write_1h=0.825),
}

# How each model is driven, where it differs from the current default (adaptive thinking with an
# effort level for Claude; for OpenAI, reasoning carried across all turns, concise summaries and
# explicit cache breakpoints). `memory` is what the model sees of its own past turns: 'reasoning'
# (its earlier thinking too) or 'replies' (only what it said). Measured with a two-turn check
# (pick a secret in your thinking, reveal it next turn), Sept 28 2026; see docs/protocol-history.md.
# `can_disable_thinking`: False for models that reject `thinking: disabled` (a retry after the per-turn
# ceiling then runs at low effort instead; a model found to reject it at run time is marked here too).
MODELS = {
    'claude-opus-5-5': {'can_disable_thinking': False},
    'claude-fable-5-2': {'can_disable_thinking': False},
    'claude-fable-5-1': {'can_disable_thinking': False},
    'claude-fable-5': {'can_disable_thinking': False},
    'claude-sonnet-5-5': {'can_disable_thinking': False},
    'claude-sonnet-4-5-20250929': {'thinking': 'budget', 'effort': False, 'memory': 'replies', 'window': 200_000},
    'claude-opus-4-5-20251101': {'thinking': 'budget', 'window': 200_000},
    'gpt-5.5': {'explicit_cache': False},
    'gpt-5.4': {'context': False, 'explicit_cache': False, 'memory': 'replies', 'window': 400_000},
    'gpt-5.2': {'context': False, 'explicit_cache': False, 'memory': 'replies', 'window': 400_000},
    'gpt-5.1': {'context': False, 'explicit_cache': False, 'memory': 'replies', 'window': 400_000},
    'o3': {'context': False, 'explicit_cache': False, 'summary': 'auto', 'memory': 'replies', 'window': 200_000},
    'o1': {'context': False, 'explicit_cache': False, 'summary': 'auto', 'memory': 'replies', 'window': 200_000},
    'gpt-5': {'context': False, 'explicit_cache': False, 'memory': 'replies', 'window': 400_000},  # checked Oct 1: replies only
    'openrouter:anthropic/claude-opus-4.1': {'thinking': 'budget', 'memory': 'replies', 'window': 200_000},
    'openrouter:anthropic/claude-sonnet-4': {'thinking': 'budget', 'memory': 'replies', 'window': 200_000},
    # open-weight and other models over OpenRouter chat completions: reasoning effort, replies only, no cache markers
    'openrouter:z-ai/glm-5.3-flash': {'thinking': 'effort', 'explicit_cache': False, 'memory': 'replies', 'window': 1_000_000},
    'openrouter:google/gemini-3.8-flash': {'thinking': 'effort', 'explicit_cache': False, 'memory': 'replies', 'window': 1_000_000},
    'gpt-4o': {'reasoning': False, 'context': False, 'explicit_cache': False, 'memory': 'replies',
               'window': 128_000, 'max_output': 16_384},  # no reasoning; its output limit is 16K
}


def spec(model):
    return {'thinking': 'adaptive', 'effort': True, 'context': True, 'explicit_cache': True, 'summary': 'concise',
            'memory': 'reasoning', 'reasoning': True, 'window': None, 'max_output': None, 'can_disable_thinking': True,
            **MODELS.get(model, {})}


def cost(model, usage, factor=1.0):
    """`usage` is the normalised dict every conversation records (Anthropic field names)."""
    p = PRICES[model]
    w1h = usage.get('cache_creation', {}).get('ephemeral_1h_input_tokens', 0)
    w5m = usage.get('cache_creation_input_tokens', 0) - w1h
    return factor * (p['input'] * usage.get('input_tokens', 0) + p['write_5m'] * w5m + p['write_1h'] * w1h
                     + p['read'] * usage.get('cache_read_input_tokens', 0)
                     + p['output'] * usage.get('output_tokens', 0)) / 1e6


class NoLimit:
    """Stands in for a semaphore when requests are not limited (the default)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


NO_LIMIT = NoLimit()
CREDIT_WAIT = 1800  # seconds a call waits for the account's credit to be topped up


def limiter(n):
    """A request limit: n in flight at once, or none for n <= 0."""
    return asyncio.Semaphore(n) if n and n > 0 else NoLimit()


# OpenAI calls are blocking urllib requests on threads. asyncio's default pool has only
# min(32, cpus + 4) threads (8 on a 4-CPU machine), which silently capped a whole batch at
# 8 requests in flight, so they get their own large pool.
HTTP_POOL = ThreadPoolExecutor(max_workers=1024, thread_name_prefix='openai')
TLS = ssl.create_default_context()  # one context for every request: loading the CA bundle per request was 60% of CPU


class BudgetExceeded(RuntimeError):
    pass


class SeatFailed(RuntimeError):
    """A seat's API call failed after all retries (e.g. no credit, auth, a bad request): the game
    stops rather than letting the seat silently pass, which would corrupt the result."""


class Ledger:
    """Append-only record of API spend, shared by every game that points at it."""

    def __init__(self, path, cap=None):
        self.path, self.cap = Path(path), cap
        self.spent = 0.0
        if self.path.exists():
            self.spent = sum(json.loads(line)['cost'] for line in self.path.read_text().splitlines() if line.strip())

    def check(self):
        if self.cap is not None and self.spent >= self.cap:
            raise BudgetExceeded(f'spend ${self.spent:.2f} has reached the cap ${self.cap:.2f}')

    def add(self, amount, **meta):
        self.spent += amount
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as f:
            f.write(json.dumps({'time': time.time(), 'cost': amount, **meta}) + '\n')


def make_client():
    headers = {}
    workspace = os.environ.get('ANTHROPIC_WORKSPACE_ID')
    if workspace:
        headers['anthropic-workspace-id'] = workspace
    # Claude Code on the web keeps ANTHROPIC_API_KEY for itself, so the key can live under another name
    key = os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('MY_ANTHROPIC_API_KEY')
    http = anthropic.DefaultAsyncHttpxClient(limits=anthropic.DEFAULT_CONNECTION_LIMITS.__class__(
        max_connections=4000, max_keepalive_connections=1000, keepalive_expiry=30))
    return anthropic.AsyncAnthropic(api_key=key, default_headers=headers, max_retries=8, timeout=900, http_client=http)


@dataclass
class Reply:
    action: dict
    thinking: str = ''


class _Base:
    def __init__(self, system, schema, *, label, ledger, semaphore, on_call=None, model=MODEL, effort='high',
                 max_tokens=8000):
        self.system, self.schema = system, schema
        self.label, self.ledger, self.semaphore, self.on_call = label, ledger, semaphore, on_call
        self.model, self.effort = model, effort
        if getattr(ledger, 'hard_budget', False) and model != 'gpt-6.1-sol':
            raise BudgetExceeded('House budget reservations currently support only GPT-6.1 Sol.')
        self.max_tokens = min(max_tokens, spec(model)['max_output'] or max_tokens)  # a model's own hard limit
        self.messages = []
        self.last_input = 0   # input tokens of the last call
        self.trimmed = 0      # turns dropped to fit the context window

    def turn_effort(self, low=False):
        """The effort for an attempt: the game's, or low when retrying a turn that hit the ceiling."""
        return 'low' if low else self.effort

    def fit(self, text):
        """When the game outgrows the model's context window, drop its oldest turns (never the first,
        which holds the seat's character and setup), so it sees as much of the recent game as fits."""
        window = spec(self.model)['window']
        if not window or not self.last_input:
            return
        over = self.last_input + len(text) // 3 + self.max_tokens - int(window * 0.95)
        while over > 0:
            users = [i for i, m in enumerate(self.messages) if isinstance(m, dict) and m.get('role') == 'user']
            if len(users) < 3:
                return
            dropped = self.messages[users[1]:users[2]]
            del self.messages[users[1]:users[2]]
            over -= len(json.dumps(dropped, default=str)) // 4
            self.trimmed += 1

    def record(self, meta, attempt, stop, start, spent, usage, thinking, out, extra=None, low=False):
        if self.on_call:
            self.on_call({'label': self.label, **(meta or {}), 'model': self.model, 'attempt': attempt, 'stop': stop,
                          'latency': round(time.monotonic() - start, 2), 'cost': spent, 'usage': usage,
                          'thinking': thinking, 'output': out, 'effort': self.turn_effort(low),
                          **({'trimmed': self.trimmed} if self.trimmed else {}),
                          **(extra or {})})
        total = sum(usage.get(k, 0) for k in ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'))
        if total:
            self.last_input = total


class ClaudeConversation(_Base):
    def __init__(self, client, system, schema, *, ttl='5m', **kw):
        super().__init__(system, schema, **kw)
        self.client = client
        self.cache = {'type': 'ephemeral', 'ttl': ttl}

    def _thinking(self, low=False):
        """Thinking and effort, by model: adaptive thinking with an effort level, or (older models) a thinking
        budget of three quarters of the per-turn ceiling, the rest left for the answer."""
        sp = spec(self.model)
        out = {'format': {'type': 'json_schema', 'schema': self.schema}}
        if sp['effort']:
            out['effort'] = self.turn_effort(low)
        if sp['thinking'] == 'budget':  # older models: a budget of 3/4 of the ceiling (the minimum when retrying)
            thinking = {'type': 'enabled', 'budget_tokens': 1024 if low else self.max_tokens * 3 // 4}
        elif low and sp['can_disable_thinking']:  # low effort doesn't bound adaptive models (v16 Opus 4.8): none
            thinking = {'type': 'disabled'}
        else:
            thinking = {'type': 'adaptive', 'display': 'summarized'}
        return {'thinking': thinking, 'output_config': out}

    async def _stream(self, messages, low=False):
        """One streamed request. When the account is out of credit (as happens while auto-reload
        catches up with a large batch), wait and retry for up to CREDIT_WAIT seconds, so the game
        pauses instead of failing."""
        waited = 0
        while True:
            try:
                async with self.client.messages.stream(
                        model=self.model, max_tokens=self.max_tokens,
                        system=[{'type': 'text', 'text': self.system, 'cache_control': self.cache}],
                        messages=messages, cache_control=self.cache,
                        **self._thinking(low)) as stream:
                    return await stream.get_final_message()
            except anthropic.APIStatusError as e:
                if low and 'thinking.type.disabled' in str(e) and spec(self.model)['can_disable_thinking']:
                    MODELS.setdefault(self.model, {})['can_disable_thinking'] = False  # retry at low effort
                    continue
                # out of credit (auto-reload lagging) or overloaded after the client's own retries: wait
                transient = 'credit balance' in str(e) or 'Overloaded' in str(e) or getattr(e, 'status_code', 0) >= 500
                if not transient or waited >= CREDIT_WAIT:
                    raise
                await asyncio.sleep(30)
                waited += 30

    async def send(self, text, meta=None, commit=True) -> Reply:
        """One turn. With commit=False (a probe) the turn is a fork: neither the question nor
        the reply is added to the conversation."""
        low = False  # True for the one retry of a turn that hit the ceiling (local: probes run alongside)
        user = {'role': 'user', 'content': text}
        if commit:
            self.fit(text)
            self.messages.append(user)
        messages = self.messages if commit else [*self.messages, user]
        refusals = truncations = 0
        for attempt in range(4):
            self.ledger.check()
            start = time.monotonic()
            async with (self.semaphore if commit else NO_LIMIT):  # probes never wait behind game turns
                async with traced_call(f'{self.label}:{(meta or {}).get("kind", "turn")}', 'anthropic', self.model,
                                       messages) as call:
                    r = await self._stream(messages, low)
                    call.output([b.model_dump(exclude_none=True) for b in r.content])
                    call.usage(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
                               cached_input_tokens=r.usage.cache_read_input_tokens)
            usage = r.usage.model_dump(exclude_none=True)
            spent = cost(self.model, usage)
            self.ledger.add(spent, label=self.label)
            thinking = '\n'.join(b.thinking for b in r.content if b.type == 'thinking' and b.thinking)
            out = ''.join(b.text for b in r.content if b.type == 'text')
            refusal = r.stop_details.model_dump() if getattr(r, 'stop_details', None) else None
            self.record(meta, attempt, r.stop_reason, start, spent, usage, thinking, out, {'refusal': refusal}, low)
            if r.stop_reason == 'refusal':
                refusals += 1
                if refusals >= 2:
                    return Reply({}, thinking)
                continue
            if r.stop_reason == 'max_tokens':
                truncations += 1
                low = True  # retry once at low effort, then the seat passes
                if truncations >= 2:
                    return Reply({}, thinking)
                continue
            try:
                action = json.loads(out)
            except json.JSONDecodeError:
                continue
            if commit:
                self.messages.append({'role': 'assistant', 'content': [b.model_dump(exclude_none=True) for b in r.content]})
            return Reply(action, thinking)
        raise SeatFailed(f'{self.label} ({self.model}): no usable reply in 4 attempts')


class OpenAIConversation(_Base):
    """OpenAI Responses API with stateless (store=false) reasoning continuity: every output
    item, including encrypted reasoning, is appended to the input for the next turn."""
    URL = 'https://api.openai.com/v1/responses'

    def __init__(self, system, schema, *, flex=False, api_key=None, **kw):
        super().__init__(system, schema, **kw)
        self.flex = flex
        self.key = api_key or os.environ.get('OPENAI_API_KEY')  # api_key: a player's own key (web.py)
        if not self.key:
            raise RuntimeError('OPENAI_API_KEY is not set')
        self.messages = [{'role': 'developer', 'content': [self._block(system)]}]

    def _block(self, text):
        block = {'type': 'input_text', 'text': text}
        if spec(self.model)['explicit_cache']:
            block['prompt_cache_breakpoint'] = {'mode': 'explicit'}
        return block

    def _reasoning(self, low=False):
        sp = spec(self.model)
        out = {'effort': self.turn_effort(low), 'summary': sp['summary']}
        if sp['context']:
            out['context'] = 'all_turns'  # carry reasoning across turns (models that support it)
        return out

    def _post(self, body):
        """One streamed request (server-sent events), returning the final response object. Streaming keeps
        bytes flowing while the model thinks: long unstreamed requests were cut off mid-game ("remote end
        closed connection") at high and max effort."""
        req = urllib.request.Request(self.URL, data=json.dumps({**body, 'stream': True}).encode(), method='POST',
                                     headers={'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900, context=TLS) as resp:  # the timeout is per read, between events
            for raw in resp:
                line = raw.decode(errors='replace').strip()
                if not line.startswith('data: '):
                    continue
                event = json.loads(line[6:])
                if event.get('type') in ('response.completed', 'response.incomplete', 'response.failed'):
                    return event['response']
                if event.get('type') == 'error':
                    raise ConnectionError(f"stream error: {json.dumps(event)[:500]}")
        raise ConnectionError('stream ended without a final response')

    def _reserved_post(self, body):
        """Count exact input (including reasoning/schema), then reserve a conservative standard-tier
        cost for EACH network attempt. Unknown outcomes retain their reservation, even after a crash.
        Prices: developers.openai.com/api/docs/models/gpt-6.1-sol (checked 2026-11-05)."""
        if not getattr(self.ledger, 'hard_budget', False):
            return self._post(body)
        if self.model != 'gpt-6.1-sol' or body.get('tools'):
            raise BudgetExceeded('No verified cost bound for this house request.')
        # A probe and the live conversation can share input items; freeze exactly what was counted.
        body = json.loads(json.dumps({**body, 'service_tier': 'default'}))
        count_body = {k: body[k] for k in ('model', 'input', 'text', 'reasoning') if k in body}
        req = urllib.request.Request(self.URL + '/input_tokens', data=json.dumps(count_body).encode(), method='POST',
                                     headers={'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30, context=TLS) as resp:
            count = json.load(resp).get('input_tokens')
        output = body['max_output_tokens']
        if type(count) is not int or not 0 <= count <= 922_000 or type(output) is not int or not 1 <= output <= 128_000:
            raise BudgetExceeded('Cannot establish a safe API request cost.')
        # No cache savings assumed: all input may be a 30-minute cache write ($2.50/M).
        # Long-context surcharge applies to all input/cache tokens and all output; include 1% headroom.
        upper = (count * 2.5 * (2 if count > 272_000 else 1)
                 + output * 10 * (1.5 if count > 272_000 else 1)) / 1e6
        upper = math.ceil(upper * 1.01 * 1e6) / 1e6
        reservation = self.ledger.reserve(upper, self.label)
        try:
            raw = self._post(body)
        except urllib.error.HTTPError as e:
            # Rejections before generation are safe to release. Timeouts, conflicts and server errors
            # may have generated a billable response; those reservations stay held for reconciliation.
            if 400 <= e.code < 500 and e.code not in (408, 409):
                self.ledger.add(0, reservation=reservation)
            raise
        usage = raw.get('usage') or {}
        if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ('input_tokens', 'output_tokens')):
            raise BudgetExceeded('API usage is missing; its maximum-cost reservation remains held.')
        detail = usage.get('input_tokens_details') or {}
        cached, written = detail.get('cached_tokens', 0), detail.get('cache_write_tokens', 0)
        if (type(cached) is not int or type(written) is not int or min(cached, written) < 0
                or cached + written > usage['input_tokens'] or usage['input_tokens'] > count
                or usage['output_tokens'] > output or raw.get('service_tier', 'default') != 'default'):
            raise BudgetExceeded('API usage exceeded the verified request bounds; its reservation remains held.')
        raw['_budget_reservation'] = reservation
        return raw

    async def _request(self, body):
        waited = 0  # seconds spent waiting for credit (auto-reload catching up), as for Claude
        attempt = 0
        while attempt < 10:
            try:
                return await asyncio.get_running_loop().run_in_executor(HTTP_POOL, self._reserved_post, body)
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors='replace')[:2000]
                credit = e.code == 402 or 'insufficient_quota' in detail or 'credit' in detail.lower()
                if credit and waited < CREDIT_WAIT:
                    await asyncio.sleep(30)
                    waited += 30
                    continue
                retryable = e.code in (408, 409, 429) or e.code >= 500  # incl. Cloudflare 52x
                if not retryable or attempt == 9 or credit or 'spend limit' in detail:
                    raise RuntimeError(f'OpenAI HTTP {e.code}: {detail}') from e
            except (TimeoutError, urllib.error.URLError, ConnectionError, http.client.HTTPException) as e:
                if attempt == 9:  # dropped connections (RemoteDisconnected) and timeouts are retried
                    raise RuntimeError(f'OpenAI transport error: {e}') from e
            await asyncio.sleep(min(2 ** attempt, 60) + random.random())
            attempt += 1

    async def send(self, text, meta=None, commit=True) -> Reply:
        low = False  # True for the one retry of a turn that hit the ceiling (local: probes run alongside)
        user = {'role': 'user', 'content': [self._block(text)]}
        if commit:
            self.fit(text)
            self.messages.append(user)
        messages = self.messages if commit else [*self.messages, user]
        truncations = 0
        for attempt in range(4):
            self.ledger.check()
            body = {'model': self.model, 'store': False, 'max_output_tokens': self.max_tokens, 'input': messages,
                    'text': {'format': {'type': 'json_schema', 'name': 'game_action', 'strict': True,
                                        'schema': self.schema}},
                    'prompt_cache_key': f'clocktower:{self.label}:{id(self) % 10 ** 8}'}
            if spec(self.model)['reasoning']:
                body['reasoning'] = self._reasoning(low)
            if spec(self.model)['explicit_cache']:
                body['prompt_cache_options'] = {'mode': 'explicit', 'ttl': '30m'}
            if self.flex:
                body['service_tier'] = 'flex'
            start = time.monotonic()
            async with (self.semaphore if commit else NO_LIMIT):
                async with traced_call(f'{self.label}:{(meta or {}).get("kind", "turn")}', 'openai', self.model,
                                       messages) as call:
                    raw = await self._request(body)
                    call.output(raw.get('output'))
                    usage_raw = raw.get('usage') or {}
                    call.usage(input_tokens=usage_raw.get('input_tokens'), output_tokens=usage_raw.get('output_tokens'),
                               cached_input_tokens=(usage_raw.get('input_tokens_details') or {}).get('cached_tokens'))
            u = raw.get('usage') or {}
            d = u.get('input_tokens_details') or {}
            read, write = d.get('cached_tokens', 0), d.get('cache_write_tokens', 0)
            usage = {'input_tokens': u.get('input_tokens', 0) - read - write, 'cache_read_input_tokens': read,
                     'cache_creation_input_tokens': write, 'output_tokens': u.get('output_tokens', 0),
                     'output_tokens_details': {'thinking_tokens': (u.get('output_tokens_details') or {}).get('reasoning_tokens', 0)}}
            priced = usage
            if u.get('input_tokens', 0) > 272_000:  # long-context pricing: input x2, output x1.5
                priced = {k: v * 2 for k, v in usage.items() if k.endswith('input_tokens')}
                priced['output_tokens'] = usage['output_tokens'] * 1.5
            spent = cost(self.model, priced, 0.5 if raw.get('service_tier') == 'flex' else 1.0)
            self.ledger.add(spent, label=self.label, **({'reservation': raw['_budget_reservation']}
                                                      if '_budget_reservation' in raw else {}))
            output = raw.get('output') or []
            thinking = '\n'.join(s.get('text', '') for item in output if item.get('type') == 'reasoning'
                                 for s in item.get('summary') or [])
            parts = [c for item in output if item.get('type') == 'message' for c in item.get('content') or []]
            out = ''.join(c.get('text', '') for c in parts if c.get('type') == 'output_text')
            refused = any(c.get('type') == 'refusal' for c in parts)
            stop = 'refusal' if refused else raw.get('status')
            if (raw.get('incomplete_details') or {}).get('reason') == 'max_output_tokens':
                stop = 'max_tokens'
            self.record(meta, attempt, stop, start, spent, usage, thinking, out, low=low)
            if stop == 'max_tokens':
                truncations += 1
                low = True  # retry once at low effort, then the seat passes
                if truncations >= 2:
                    return Reply({}, thinking)
                continue
            if stop != 'completed':
                continue
            try:
                action = json.loads(out)
            except json.JSONDecodeError:
                continue
            if commit:
                self.messages.extend(output)
            return Reply(action, thinking)
        raise SeatFailed(f'{self.label} ({self.model}): no usable reply in 4 attempts')


OPENAI_PREFIXES = ('gpt-', 'o1', 'o3', 'o4')


class OpenRouterConversation(_Base):
    """Models served through OpenRouter chat completions: older Claude models (Amazon Bedrock) with a thinking
    budget (three quarters of the ceiling) and Anthropic prompt caching, and open-weight or other models
    (spec thinking 'effort', no cache markers) with a reasoning effort. These endpoints take no
    structured-output setting, so the reply schema is stated in the system prompt and the JSON is parsed
    (a malformed reply is retried). The API drops earlier thinking from the context, so these models
    remember their replies only."""
    URL = 'https://openrouter.ai/api/v1/chat/completions'

    def __init__(self, system, schema, **kw):
        super().__init__(system, schema, **kw)
        self.key = os.environ.get('OPENROUTER_API_KEY')
        if not self.key:
            raise RuntimeError('OPENROUTER_API_KEY is not set')
        self.slug = self.model.split(':', 1)[1]
        rules = system + ('\n\nReply with one JSON object and nothing else (no code fence), matching this JSON '
                          'schema:\n' + json.dumps(schema))
        self.cache_markers = spec(self.model)['explicit_cache']
        block = {'type': 'text', 'text': rules}
        if self.cache_markers:
            block['cache_control'] = {'type': 'ephemeral'}
        self.messages = [{'role': 'system', 'content': [block]}]

    def _reserved_post(self, body):  # what OpenAIConversation._request calls; no house budget applies here
        return self._post(body)

    def _post(self, body):
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method='POST',
                                     headers={'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900, context=TLS) as resp:
            return json.load(resp)

    _request = OpenAIConversation._request

    async def send(self, text, meta=None, commit=True) -> Reply:
        low = False  # True for the one retry of a turn that hit the ceiling (local: probes run alongside)
        user = {'role': 'user', 'content': [{'type': 'text', 'text': text}]}
        if commit:
            self.fit(text)
            self.messages.append(user)
        messages = self.messages if commit else [*self.messages, user]
        truncations = 0
        for attempt in range(4):
            self.ledger.check()
            if self.cache_markers:  # one cache breakpoint on the newest user turn (plus the system prompt)
                for m in messages[1:]:
                    if m['role'] == 'user':
                        m['content'][-1].pop('cache_control', None)
                messages[-1]['content'][-1]['cache_control'] = {'type': 'ephemeral'}
            if spec(self.model)['thinking'] == 'budget':
                reasoning = {'max_tokens': 1024 if low else self.max_tokens * 3 // 4}
            else:
                reasoning = {'effort': self.turn_effort(low)}
            body = {'model': self.slug, 'messages': messages, 'max_tokens': self.max_tokens,
                    'reasoning': reasoning, 'usage': {'include': True},
                    'provider': {'require_parameters': True}}
            start = time.monotonic()
            async with (self.semaphore if commit else NO_LIMIT):
                async with traced_call(f'{self.label}:{(meta or {}).get("kind", "turn")}', 'openrouter', self.model,
                                       messages) as call:
                    raw = await self._request(body)
                    choice_raw = (raw.get('choices') or [{}])[0].get('message')
                    usage_raw = raw.get('usage') or {}
                    call.output(choice_raw)
                    call.usage(input_tokens=usage_raw.get('prompt_tokens'), output_tokens=usage_raw.get('completion_tokens'),
                               cached_input_tokens=(usage_raw.get('prompt_tokens_details') or {}).get('cached_tokens'))
            u = raw.get('usage') or {}
            read = (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
            write = (u.get('prompt_tokens_details') or {}).get('cache_write_tokens', 0)
            usage = {'input_tokens': u.get('prompt_tokens', 0) - read - write, 'cache_read_input_tokens': read,
                     'cache_creation_input_tokens': write, 'output_tokens': u.get('completion_tokens', 0),
                     'output_tokens_details': {'thinking_tokens': (u.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)}}
            spent = u.get('cost') if u.get('cost') is not None else cost(self.model, usage)
            self.ledger.add(spent, label=self.label)
            choice = (raw.get('choices') or [{}])[0]
            msg = choice.get('message') or {}
            thinking, out = msg.get('reasoning') or '', (msg.get('content') or '').strip()
            stop = {'stop': 'end_turn', 'length': 'max_tokens'}.get(choice.get('finish_reason'), choice.get('finish_reason'))
            self.record(meta, attempt, stop, start, spent, usage, thinking, out, low=low)
            if stop == 'max_tokens':
                truncations += 1
                low = True  # retry once at low effort, then the seat passes
                if truncations >= 2:
                    return Reply({}, thinking)
                continue
            try:
                action = json.loads(out[out.find('{'):out.rfind('}') + 1])
            except json.JSONDecodeError:
                continue
            if commit:
                self.messages.append({'role': 'assistant', 'content': out})
            return Reply(action, thinking)
        raise SeatFailed(f'{self.label} ({self.model}): no usable reply in 4 attempts')


def conversation(client, system, schema, *, model, **kw):
    """A Claude, OpenAI or OpenRouter conversation, by model name."""
    if model.startswith('openrouter:'):
        kw.pop('ttl', None)
        kw.pop('flex', None)
        return OpenRouterConversation(system, schema, model=model, **kw)
    if model.startswith(OPENAI_PREFIXES):
        kw.pop('ttl', None)  # a client may carry the OpenAI key to use (web.py's bring-your-own-key tables)
        return OpenAIConversation(system, schema, model=model, api_key=getattr(client, 'openai_api_key', None), **kw)
    kw.pop('flex', None)
    return ClaudeConversation(client, system, schema, model=model, **kw)
