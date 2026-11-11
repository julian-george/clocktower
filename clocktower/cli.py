"""Command line: run AI games, play as a human, simulate, or summarise (`python -m clocktower --help`).

    python -m clocktower run                          # one game, Claude Opus 5.5 in every seat, scripted Storyteller
    python -m clocktower run --model random           # random practice bots, no API calls
    python -m clocktower run --days 1                 # stop after the first day (smoke test)
    python -m clocktower run --games 6 --parallel 3   # a batch, three games at a time
    python -m clocktower play                         # you and friends vs the AIs, in the browser
    python -m clocktower sim --games 20               # random players, no API calls
    python -m clocktower report games/*               # summarise finished games
    python -m clocktower analyze games/v5/*/game-*    # win rates and belief-probe scores by model pairing
    python -m clocktower plan MODELS.json ... > P.json  # a tournament plan; `tournament P.json` runs it
    python -m clocktower tournament P.json --resume-stopped  # relaunch a plan's stopped games
    python -m clocktower ratings [--lab-trends]       # the 2026 Bradley-Terry fit and its pages
    python -m clocktower allocate ...                 # where extra games go (clocktower/allocate.py)
    python -m clocktower mixed games/v17/b1-          # mixed-team analysis
    python -m clocktower memory-check MODEL           # does a new model see its earlier reasoning? (llm.MODELS)
    python -m clocktower fetch 'v17/f3-*'             # download game logs from the Hugging Face dataset
    python -m clocktower replay games/X --view Casey  # re-read a game from one seat (or public/omniscient)
    python -m clocktower watch                        # spectator app at http://127.0.0.1:8800
    python -m clocktower site                         # the old spectator app as static files in site/legacy/
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .agents import ModelSeat, RandomSeat
from .characters import MINIONS
from .game import ALL_NAMES, Game
from .llm import MODEL, Ledger, conversation, limiter, make_client, spec
from .player import scaffold
from .prompts import player_system
from .schema import player_schema

ROOT = Path(__file__).resolve().parent.parent
GAMES = ROOT / 'games'
PROTOCOL = 'v17'  # frozen (docs/PROTOCOL.md); a change that affects play bumps it and adds to docs/protocol-history.md
RUNS = GAMES / PROTOCOL  # new games go here by default


@dataclass
class Shared:
    """One client, spend ledger and request limit for every game in this process."""
    client: object
    ledger: Ledger
    semaphore: object  # an asyncio.Semaphore, or llm.NoLimit


def shared(args):
    if args.workspace:
        os.environ.setdefault('ANTHROPIC_WORKSPACE_ID', args.workspace)
    return Shared(make_client(), Ledger(GAMES / 'ledger.jsonl', cap=args.budget), limiter(args.concurrency))


def table_size(players, seed):
    """'12' is always 12 players; a range '10-12' picks one by seed (10 + seed % 3), so a side-swapped
    pair, which shares its seed, shares its table size too."""
    lo, _, hi = str(players).partition('-')
    lo, hi = int(lo), int(hi or lo)
    if not 5 <= lo <= hi <= 15:
        raise ValueError(f'--players must be within 5-15, got {players}')
    return lo + (seed or 0) % (hi - lo + 1)


def _model(text, effort):
    """'claude-opus-5-5@low' -> ('claude-opus-5-5', 'low'); no suffix keeps `effort`."""
    model, _, e = text.partition('@')
    return model, e or effort


def seat_models(g, args, seed):
    """Which model (and effort) plays each seat, once the characters are dealt.

    A team's model may be a comma list (`--good-model A,B,C`): the team's seats, in an order shuffled
    by the seed, take the models in turn, so a mixed team splits as evenly as it can. Then each
    `--seat-model` override applies, in order: `good:MODEL:N` / `evil:MODEL:N` gives N of that team's
    seats (picked by the seed; N defaults to 1) to MODEL, and `CHARACTER:MODEL` (e.g. `Imp:gpt-6-astra`)
    gives the seat dealt that character. Any model may carry an effort, `MODEL@low`."""
    rng = random.Random(f'{seed}-seats')
    out = {}
    for team in ('good', 'evil'):
        spec_ = {'good': args.good_model, 'evil': args.evil_model}[team] or args.model
        effort = {'good': args.good_effort, 'evil': args.evil_effort}[team] or args.effort
        models = [_model(m.strip(), effort) for m in spec_.split(',')]
        seats = [p.name for p in g.players if (p.evil) == (team == 'evil')]
        rng.shuffle(seats)
        for i, name in enumerate(seats):
            out[name] = models[i % len(models)]
    for override in getattr(args, 'seat_model', None) or []:
        who, _, rest = override.partition(':')
        if who in ('good', 'evil'):
            model, _, n = rest.partition(':')
            effort = {'good': args.good_effort, 'evil': args.evil_effort}[who] or args.effort
            seats = sorted(p.name for p in g.players if p.evil == (who == 'evil'))
            for name in rng.sample(seats, int(n or 1)):
                out[name] = _model(model, effort)
        else:
            p = next((q for q in g.players if q.character == who), None)
            if p is None:
                continue  # that character isn't in this game
            effort = (args.evil_effort if p.evil else args.good_effort) or args.effort
            out[p.name] = _model(rest, effort)
    if getattr(args, 'focal_seat', None) is not None:
        out[g.players[args.focal_seat].name] = _model(args.focal_model, args.effort)
    return out


class FocalConversation:
    """The focal seat's conversation: each turn's text goes through the player harness (player/scaffold.py) first."""

    def __init__(self, conversation):
        self.conversation = conversation

    async def send(self, text, meta=None, commit=True):
        return await self.conversation.send(scaffold.wrap_turn(text, (meta or {}).get('kind')), meta, commit)

    def __getattr__(self, name):
        return getattr(self.conversation, name)


def build(args, seed, out, env: Shared | None = None, humans=None, minion=None):
    """`humans(g)`, if given, is called once the characters are dealt and returns {seat name: HumanSeat}
    for the seats people play (web.py); the model 'random' is a free practice bot (RandomSeat)."""
    characters = args.characters.split(',') if args.characters else None
    names = ALL_NAMES[:len(characters) if characters else table_size(args.players, seed)]
    probes = getattr(args, 'probes', False)
    if getattr(args, 'probe_focal_only', False):
        probes = {names[args.focal_seat]}  # the belief probe is the score: ask only the focal seat
    game = Game({}, None, names=names, seed=seed, private_beats=args.private_beats,
                public_beats=args.public_beats, max_days=args.days, waves=args.waves, characters=characters,
                minion=minion, out=out, probes=probes)
    if env is None:  # sim: random players, no API
        rng = random.Random(seed + 1)
        game.seats = {n: RandomSeat(n, random.Random(rng.random()), names) for n in names}
        return game, []
    ttl = '1h' if humans else '5m'  # a human can take longer than five minutes between turns
    convs = []
    game.meta['models'] = {}
    assigned, people = {}, {}  # seat -> (model, effort) and seat -> HumanSeat, fixed once the characters are dealt
    bots = random.Random(seed + 1)

    def seat(name, team):  # called once the characters are dealt
        if not assigned:
            assigned.update(seat_models(game.g, args, seed))
            people.update(humans(game.g) if humans else {})
        if name in people:
            game.meta['models'][name] = 'human'
            return people[name]
        model, effort = assigned[name]
        game.meta['models'][name] = model if effort == args.effort else f'{model}@{effort}'
        if model == 'random':
            return RandomSeat(name, random.Random(bots.random()), names)
        system = player_system(len(names), spec(model)['memory'], args.private_beats)
        focal = getattr(args, 'focal_seat', None) is not None and name == names[args.focal_seat]
        if focal:
            system = scaffold.build_system(system)
            game.meta['focal'] = {'seat': name, 'model': model}
        c = conversation(env.client, system, player_schema(names), label=name, ledger=env.ledger,
                         semaphore=env.semaphore, on_call=game.record_call, model=model, effort=effort,
                         max_tokens=args.max_output, ttl=ttl, flex=args.flex)
        if focal:
            c = FocalConversation(c)
        convs.append((name, c))
        return ModelSeat(c)

    game.seat_factory = seat
    config = {'protocol': PROTOCOL, 'model': args.model, 'good_model': args.good_model, 'evil_model': args.evil_model,
              'player_effort': args.effort, 'good_effort': args.good_effort or args.effort,
              'evil_effort': args.evil_effort or args.effort, 'max_output_tokens': args.max_output,
              'seat_models': getattr(args, 'seat_model', None) or [],
              'good_memory': spec(_model((args.good_model or args.model).split(',')[0], '')[0])['memory'],
              'evil_memory': spec(_model((args.evil_model or args.model).split(',')[0], '')[0])['memory'],
              'storyteller': 'scripted',
              'seed': seed,
              'private_beats': args.private_beats, 'public_beats': args.public_beats,
              'whispers': 'private', 'waves': args.waves, 'minion': minion,
              'max_days': args.days, 'human': bool(humans), 'characters': characters, 'probes': args.probes,
              'openai_tier': 'flex' if args.flex else 'default',
              'players': len(names), 'players_arg': str(args.players),
              'player_prompt_sha': hashlib.sha256(player_system(len(names), private_beats=args.private_beats).encode()).hexdigest()[:12]}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    return game, convs


def save_conversations(game, convs):
    if not game.out:
        return
    d = game.out / 'conversations'
    d.mkdir(exist_ok=True)
    for name, c in convs:
        (d / f'{name}.json').write_text(json.dumps({'system': c.system, 'messages': c.messages}, indent=1))


def summary(result, out):
    t = result['tokens']
    print(f"\n{(result['winner'] or 'no winner').upper()} — {result['reason']}  (days: {result['days']})")
    print(f"cost ${result['cost']:.2f} over {result['calls']} calls, {result['seconds'] / 60:.1f} min; tokens: "
          f"{t['cache_read'] / 1e6:.2f}M cached, {t['cache_write'] / 1e6:.2f}M written, {t['input'] / 1e3:.0f}K uncached, "
          f"{t['output'] / 1e3:.0f}K output")
    print(f'logs: {out}')


async def play_one(args, seed, out, env, humans=None, serve=None, minion=None):
    game, convs = build(args, seed, out, env, humans, minion)
    if serve:
        serve(game)
    try:
        result = await game.run()
    finally:
        save_conversations(game, convs)
    summary(result, game.out)
    return result


async def run_batch(args):
    env = shared(args)
    base = args.seed if args.seed is not None else random.randrange(2 ** 32)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    if args.games == 1 and not args.swap:
        out = Path(args.out) if args.out else RUNS / f'{stamp}-{base}'
        minion = args.minions.split(',')[0].strip() if getattr(args, 'minions', None) else None
        return [await play_one(args, base, out, env, minion=minion)]
    root = Path(args.out) if args.out else RUNS / f'batch-{stamp}'
    limit = asyncio.Semaphore(args.parallel or 10 ** 6)  # 0 = every game of the batch at once (swaps included)
    numbers = parse_only(args.only) if args.only else range(1, args.games + 1)

    swapped = argparse.Namespace(**{**vars(args), 'good_model': args.evil_model or args.model,
                                    'evil_model': args.good_model or args.model,
                                    'good_effort': args.evil_effort, 'evil_effort': args.good_effort})

    forced = [m.strip() for m in args.minions.split(',')] if getattr(args, 'minions', None) else None
    assert not forced or all(m in MINIONS for m in forced), f'--minions must name Minions: {MINIONS}'

    async def one(k, a=args, suffix=''):  # game k gets seed base + k - 1, and a guaranteed Minion: the plan's
        # choice (--minions, one per game, so a batch can balance them) or else picked by the seed
        minion = None if args.characters else (forced[(k - 1) % len(forced)] if forced
                                               else MINIONS[(base + k - 1) % len(MINIONS)])
        out = root / f'game-{k:02d}{suffix}'
        done = out / 'result.json'
        cfg = out / 'config.json'
        if cfg.exists() and json.loads(cfg.read_text()).get('protocol') != PROTOCOL:
            raise SystemExit(f'{out} holds {json.loads(cfg.read_text()).get("protocol")} games; write {PROTOCOL} games '
                             f'to a new folder (games/{PROTOCOL}/...)')
        if done.exists() and json.loads(done.read_text()).get('winner'):
            return json.loads(done.read_text())  # finished earlier: keep it (so a batch can be resumed)
        if out.exists() and not (out / 'checkpoint.pkl').exists():
            shutil.rmtree(out)  # an unfinished game with no checkpoint restarts from scratch
        # otherwise it continues from its last dusk (Game.restore)
        async with limit:
            return await play_one(a, base + k - 1, out, env, minion=minion)

    side = getattr(args, 'side', 'both')
    jobs = [one(k) for k in numbers] if side != 'swap' else []
    if args.swap and side != 'normal':  # the same seed again with the models' sides swapped, so setup luck cancels
        jobs += [one(k, swapped, '-swap') for k in numbers]
    results = await asyncio.gather(*jobs)
    from .report import main as report
    report(sorted(str(p) for p in root.iterdir() if p.is_dir()))
    return results


def parse_only(spec):
    out = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out += range(int(a), int(b or a) + 1)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('allocate', 'mixed', 'memory-check'):  # commands with their own argument parsers
        from importlib import import_module
        return import_module(f'.{argv[0].replace("-", "_")}', __package__).main(argv[1:])
    ap = argparse.ArgumentParser(prog='clocktower', description='Blood on the Clocktower (Trouble Brewing) played by '
                                 'language models: run games and experiments, rate models, watch or play.')
    sub = ap.add_subparsers(dest='command', required=True, metavar='COMMAND')
    rp = sub.add_parser('report', help='summarise finished games (win rates, days, cost)')
    rp.add_argument('paths', nargs='+')
    an = sub.add_parser('analyze', help='win rates, voting and belief-probe scores by model pairing')
    an.add_argument('paths', nargs='+')
    sv = sub.add_parser('supervise', help='watch running batches; pause them on errors, lost turns or overspend')
    sv.add_argument('roots', nargs='+', help='batch directory prefixes, e.g. games/v8/t2')
    sv.add_argument('--max-spend', type=float, help='pause if these games cost more than this (USD)')
    sv.add_argument('--review-after', type=float, help='exit (without pausing) after this many minutes')
    wp = sub.add_parser('watch', help='spectator app for the games under games/ (http://127.0.0.1:8800)')
    wp.add_argument('--port', type=int, default=8800)
    wp.add_argument('--games', default=str(GAMES))
    sp = sub.add_parser('site', help='build the spectator app as static files (site/)')
    sp.add_argument('--games', default=str(GAMES))
    sp.add_argument('--out', default=str(ROOT / 'site' / 'legacy'))  # site/src and site/dist hold the public website
    pl = sub.add_parser('plan', help='write a tournament plan (JSON) for a list of models')
    pl.add_argument('models', help='JSON file: a list of model ids, or of {id, short, ...}, in release order')
    pl.add_argument('--within', type=int, help='only pairs at most this many places apart (default: all pairs)')
    pl.add_argument('--pairs', type=int, default=1, help='side-swapped pairs per matchup')
    pl.add_argument('--seed', type=int, default=80000)
    pl.add_argument('--out', default=f'games/{PROTOCOL}/f1')
    pl.add_argument('--players', default='10-12')
    tp = sub.add_parser('tournament', help='run every matchup of a plan at once (resumable)')
    tp.add_argument('plan')
    tp.add_argument('--dry-run', action='store_true', help='print the commands instead')
    tp.add_argument('--only', help='comma-separated matchup names: run just these (same seeds as in the full plan)')
    tp.add_argument('--resume-stopped', action='store_true',
                    help='relaunch, one process each, every game that stopped (seat failed, budget, halted) and is not running')
    ra = sub.add_parser('ratings', help='the 2026 Bradley-Terry fit (docs/plans/f2-ratings.json) and its pages')
    ra.add_argument('--games', default=str(GAMES), help='folder holding v16/ and v17/ game logs (events.jsonl needed)')
    ra.add_argument('--models', default=str(ROOT / 'experiments/models-2026.json'))
    ra.add_argument('--out', default=str(ROOT / 'docs/plans/f2-ratings.json'))
    ra.add_argument('--boot', type=int, default=500, help='bootstrap resamples (over side-swapped pairs)')
    ra.add_argument('--no-page', dest='page', action='store_false', help="don't rebuild the HTML pages")
    ra.add_argument('--lab-trends', action='store_true', help='also per-lab gains (docs/charts/lab-trends.json)')
    for name, text in (('allocate', 'where extra games go, by a fixed rule (see clocktower/allocate.py)'),
                       ('mixed', 'mixed-team analysis (clocktower/mixed.py)'),
                       ('memory-check', 'does a model see its own earlier reasoning next turn? (for llm.MODELS)')):
        sub.add_parser(name, help=text)  # dispatched above, before parsing
    fp = sub.add_parser('fetch', help='download game logs from the Hugging Face dataset (e.g. v9/e1-*)')
    fp.add_argument('patterns', nargs='+')
    rv = sub.add_parser('replay', help='re-read a game from one seat, the public view or omniscient')
    rv.add_argument('path')
    rv.add_argument('--view', default='omniscient', help='a player name, public, or omniscient')
    rv.add_argument('--thinking', action='store_true', help='include thinking (omniscient or that player)')
    helps = {'run': 'play one game or a batch with AI seats (the eval)',
             'play': 'host a table in the browser: you and friends vs AI seats',
             'sim': 'random players, no API calls (an engine check)'}
    for name in ('run', 'play', 'sim'):
        p = sub.add_parser(name, help=helps[name])
        p.add_argument('--seed', type=int)
        p.add_argument('--private-beats', type=int, default=6, help='whisper-only beats each day')
        p.add_argument('--minions', help='comma-separated guaranteed Minion for game 1, 2, ... (tournament passes a balanced choice; '
                                            'default for a bare run: by seed)')
        p.add_argument('--public-beats', type=int, default=3, help='town-square beats before nominations')
        p.add_argument('--waves', type=int, default=3, help='groups per beat; later groups see earlier ones')
        p.add_argument('--days', type=int, default=15, help='stop after this many days')
        p.add_argument('--players', default='10-12', metavar='N or A-B',
                       help='table size, 5-15 (official Trouble Brewing setup for that count); a range such as '
                            '10-12 picks each game\'s size from its seed')
        p.add_argument('--characters', help='comma-separated forced setup, in seating order (tests)')
        p.add_argument('--out')
        if name in ('run', 'sim'):
            p.add_argument('--games', type=int, default=1)
        if name != 'sim':
            p.add_argument('--model', default=MODEL, help='player model')
            p.add_argument('--good-model', help='model for the good team (default --model)')
            p.add_argument('--evil-model', help='model for the evil team (default --model)')
            p.add_argument('--effort', default='high', help='player reasoning effort (same for every model)')
            p.add_argument('--good-effort', help='reasoning effort for the good team (default --effort)')
            p.add_argument('--evil-effort', help='reasoning effort for the evil team (default --effort)')
            p.add_argument('--seat-model', action='append', metavar='SPEC',
                           help='per-seat override, repeatable: good:MODEL[:N], evil:MODEL[:N] (N seats of that '
                                'team, picked by seed) or CHARACTER:MODEL (e.g. Imp:gpt-6-astra); MODEL@effort ok')
            p.add_argument('--focal-seat', type=int, metavar='INDEX',
                           help='seating index of the one seat that runs --focal-model and the player harness '
                                '(clocktower/player/); every other seat follows --model and the team options')
            p.add_argument('--focal-model', help='model for the --focal-seat')
            p.add_argument('--max-output', type=int, default=8000,
                           help='per-turn output ceiling, thinking included, identical for every model')
            p.add_argument('--budget', type=float, default=1000.0, help='cap on total spend in games/ledger.jsonl')
            p.add_argument('--concurrency', type=int, default=0,
                           help='cap on API requests in flight (0 = no cap: every game and seat runs at once; '
                                'rate limits are handled by retries)')
            p.add_argument('--flex', action='store_true',
                           help='OpenAI Flex tier (half price, but throttled under load: calls can take minutes)')
            p.add_argument('--no-probes', dest='probes', action='store_false',
                           help='skip the out-of-game belief probes (probes.jsonl)')
            p.add_argument('--probe-focal-only', action='store_true',
                           help='with --focal-seat: belief-probe only the focal seat')
            p.add_argument('--workspace', default=os.environ.get('ANTHROPIC_WORKSPACE_ID', ''),
                           help='Anthropic workspace id sent with every request (default: $ANTHROPIC_WORKSPACE_ID; '
                                'only needed if your organisation requires one)')
        if name == 'run':
            p.add_argument('--parallel', type=int, default=0, help='games at once in a batch (0 = all)')
            p.add_argument('--only', help='run just these game numbers of the batch, e.g. 13,14,16-20')
            p.add_argument('--side', choices=('both', 'normal', 'swap'), default='both',
                           help='with --swap: play only the original game or only the swapped one (one process per game)')
            p.add_argument('--swap', action='store_true',
                           help='also play every seed with the good and evil models swapped (game-NN-swap)')
        if name == 'play':
            p.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8765)))
            p.add_argument('--host', default='0.0.0.0', help='address to listen on (127.0.0.1: this computer only)')
            # for play: spend allowed this session, on top of the ledger; and no effort unless one is given,
            # so a table's AI seats use the play server's own default (web.Settings.effort; PLAY_EFFORT)
            p.set_defaults(budget=100.0, effort=None)
    args = ap.parse_args(argv)
    if getattr(args, 'focal_seat', None) is not None and not args.focal_model:
        ap.error('--focal-seat needs --focal-model')
    if args.command == 'report':
        from .report import main as report
        report(args.paths)
    elif args.command == 'supervise':
        from .supervise import main as supervise
        raise SystemExit(supervise(args.roots, args.max_spend, args.review_after))
    elif args.command == 'analyze':
        from .analyze import main as analyze
        analyze(args.paths)
    elif args.command == 'watch':
        from .watch import serve as watch
        watch(args.games, args.port)
    elif args.command == 'site':
        from .site import build as build_site
        build_site(args.games, args.out)
    elif args.command == 'plan':
        from .tournament import make_plan
        models = json.loads(Path(args.models).read_text())
        print(json.dumps(make_plan(models, args.within, args.pairs, args.seed, args.out, args.players), indent=1))
    elif args.command == 'tournament':
        from .tournament import run as tournament, stopped
        only = set(args.only.split(',')) if args.only else None
        if args.resume_stopped:
            names = stopped(args.plan)
            print(len(names), 'stopped games:', ' '.join(names), flush=True)
            if not names:
                return
            only = set(names)
        raise SystemExit(tournament(args.plan, args.dry_run, only))
    elif args.command == 'ratings':
        from .ratings import main as ratings
        ratings(args.games, args.out, args.models, args.boot, args.page, args.lab_trends)
    elif args.command == 'fetch':
        from .store import fetch
        print(fetch(args.patterns))
    elif args.command == 'replay':
        from .report import replay
        replay(args.path, args.view, args.thinking)
    elif args.command == 'run':
        asyncio.run(run_batch(args))
    elif args.command == 'play':
        from .web import main as play  # spend and caps: PLAY_* environment variables (docs/hosting.md)
        play(args)
    else:
        base = args.seed if args.seed is not None else random.randrange(2 ** 32)
        results = []
        for i in range(args.games):
            game, _ = build(args, base + i, None)
            results.append(asyncio.run(game.run()))
        wins = {w: sum(r['winner'] == w for r in results) for w in ('good', 'evil', None)}
        print(f'{args.games} random games: {wins}')
