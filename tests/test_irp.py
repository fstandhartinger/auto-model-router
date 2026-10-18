"""Server MUST conformance for IRP v0.3.0-draft (§2–§5, §8).

§6 is not claimed: this implementation offers suggest-only alongside its legacy
proxy. §5.3 requirements target clients rather than this server.
"""
import copy
import random
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_router.catalog import Catalog, ModelInfo, Prices
from auto_router.config import RouterConfig
from auto_router.irp import Problem, endpoints, rank, supported, validate
from auto_router.router import Router


@pytest.fixture
def router():
    models = [ModelInfo('small','a','test/small',Prices(1,2),capability={'general':40}),
              ModelInfo('large','b','test/large',Prices(3,8),capability={'general':70}),
              ModelInfo('private','a','private',Prices(0,0),capability={'general':55})]
    r = Router(RouterConfig({},Catalog(models),policy={'classifier':{'backend':'heuristic'}}))
    r.classifier = None
    return r


def body(candidates=None, **request):
    return {'request':{'messages':[{'role':'user','content':'Please help with this task.'}],**request},
            'routing':{'candidates':candidates or [candidate('a','test/small'),candidate('b','test/large',3)]}}


def candidate(cid, model, price=1, cache=0):
    return {'id':cid,'model':model,'pricing':{'input':price,'cache_read':price/10,'output':price*2},
            'expected_usage':{'cache_read_tokens':cache}}


def client_for(router, guard=None):
    app=FastAPI();app.include_router(endpoints(lambda:router,guard));return TestClient(app)


def test_shapes_and_only_defined_fields(router):
    models=supported(router.config)
    assert models['object']=='list'
    assert all(m['object']=='model' for m in models['data'])
    assert 'system1models.ai/private' in {m['id'] for m in models['data']}
    data=rank(router,body())
    assert data['object']=='routing.ranking' and isinstance(data['id'],str)
    assert type(data['created']) is int and set(data['router'])=={'id','version'}
    assert set(data)=={'id','object','created','router','ranked','extra'}
    assert set(data['extra'])=={'system1models.ai'}
    for e in data['ranked']:
        assert set(e)<= {'candidate_id','expected_quality','expected_cost_usd','expected_usage','reasoning_effort','extra'}
        assert 0<=e['expected_quality']<=1
        assert set(e['expected_usage'])=={'input_tokens','cache_read_tokens','output_tokens'}
        assert all(type(v) is int and v>=0 for v in e['expected_usage'].values())
        assert 'reasoning_effort' not in e


def test_ignore_unknown_members_namespaces_model_stream(router):
    b=body(); expected=rank(router,b)['ranked']
    b['future']=1;b['routing']['future']=2;b['routing']['extra']={'evil.invalid':{'cost_quality_tradeoff':10}}
    for c in b['routing']['candidates']:
        c['future']=3;c['pricing']['future']=4;c['expected_usage']['future']=5
        c['extra']={'evil.invalid':{'expected_quality':1,'pricing':{'input':0}}}
    b['request']['model']={'ignored':True};b['request']['stream']='ignored'
    assert rank(router,b)['ranked']==expected


def test_default_and_all_tradeoffs_monotonic_even_same_model_different_sellers(router):
    rng=random.Random(719)
    for _ in range(150):
        b=body([candidate(str(i),rng.choice(['test/small','test/large']),rng.random()*30,rng.randrange(0,10000)) for i in range(8)])
        costs=[]
        for t in range(11):
            b['routing']['cost_quality_tradeoff']=t
            data=rank(router,b); costs.append(data['ranked'][0]['expected_cost_usd'])
            assert len(data['ranked'])==8
            assert len({e['candidate_id'] for e in data['ranked']})==8
        assert all(a>=c for a,c in zip(costs,costs[1:]))
        del b['routing']['cost_quality_tradeoff'];default=rank(router,b)['ranked']
        b['routing']['cost_quality_tradeoff']=5;assert rank(router,b)['ranked']==default
        b['routing']['cost_quality_tradeoff']=0
        assert rank(router,b)['ranked'][0]['expected_quality']==max(e['expected_quality'] for e in default)
        b['routing']['cost_quality_tradeoff']=10
        assert rank(router,b)['ranked'][0]['expected_cost_usd']==min(e['expected_cost_usd'] for e in default)


def test_cache_seller_identity_prices_and_complete_conversation(router):
    b=body([candidate('cold','test/large',1),candidate('warm','test/large',1,1000)])
    b['request']['messages'].insert(0,{'role':'system','content':'x'*8000})
    b['request']['tools']=[{'type':'function','function':{'name':'f','parameters':{'description':'x'*1000}}}]
    entries={e['candidate_id']:e for e in rank(router,b)['ranked']}
    assert entries['warm']['expected_cost_usd']<entries['cold']['expected_cost_usd']
    for c in b['routing']['candidates']:
        e=entries[c['id']];u=e['expected_usage'];p=c['pricing']
        assert e['expected_cost_usd']==((u['input_tokens']-u['cache_read_tokens'])*p['input']+u['cache_read_tokens']*p['cache_read']+u['output_tokens']*p['output'])/1e6
    full=entries['cold']['expected_usage']['input_tokens']
    b['request']['messages']=b['request']['messages'][-1:];b['request'].pop('tools')
    assert rank(router,b)['ranked'][0]['expected_usage']['input_tokens']<full


def test_scorable_subset_and_empty_422(router):
    b=body([candidate('u','unknown'),candidate('a','test/small')])
    assert [e['candidate_id'] for e in rank(router,b)['ranked']]==['a']
    b['routing']['candidates']=b['routing']['candidates'][:1]
    r=client_for(router).post('/v1/routing/rank',json=b)
    assert r.status_code==422 and r.json()['type']=='urn:irp:problem:no-scorable-candidate'
    assert r.headers['content-type']=='application/problem+json'


@pytest.mark.parametrize('n',[1,512])
def test_candidate_boundaries(router,n):
    b=body([candidate(str(i),'test/small') for i in range(n)])
    assert len(rank(router,b)['ranked'])==n


@pytest.mark.parametrize('change',[
    lambda b:b['routing'].update(candidates=[]),
    lambda b:b['routing'].update(candidates=[candidate(str(i),'test/small') for i in range(513)]),
    lambda b:b['routing']['candidates'][1].update(id='a'),
    lambda b:b['routing']['candidates'][0].update(id=''),
    lambda b:b['routing']['candidates'][0].update(id='x'*129),
    lambda b:b['routing']['candidates'][0]['pricing'].update(input=-1),
    lambda b:b['routing']['candidates'][0]['pricing'].update(input=True),
    lambda b:b['routing']['candidates'][0]['pricing'].pop('output'),
    lambda b:b['routing']['candidates'][0]['expected_usage'].update(cache_read_tokens=1.5),
    lambda b:b['routing']['candidates'][0]['expected_usage'].update(cache_read_tokens=-1),
    lambda b:b['routing'].update(cost_quality_tradeoff=-1),
    lambda b:b['routing'].update(cost_quality_tradeoff=11),
    lambda b:b['routing'].update(cost_quality_tradeoff=5.0),
    lambda b:b['routing'].update(cost_quality_tradeoff=True),
    lambda b:b['request'].update(messages=[]),
    lambda b:b['request'].update(messages=[{'role':'bogus','content':'hi'}]),
    lambda b:b['request'].update(max_tokens=0),
    lambda b:b['request'].update(max_completion_tokens=True),
])
def test_malformed_400(router,change):
    b=body();change(b);r=client_for(router).post('/v1/routing/rank',json=b)
    assert r.status_code==400 and r.json()['type']=='urn:irp:problem:invalid-request'
    assert r.headers['content-type']=='application/problem+json'


def test_all_section8_errors_and_malformed_json(router):
    c=client_for(router)
    for data in ('{broken','null','[]','{"request":{}}'):
        assert c.post('/v1/routing/rank',content=data).status_code==400
    for status in (400,402,422,503):
        async def guard(request):raise Problem(status,'Existing host policy')
        r=client_for(router,guard).post('/v1/routing/rank',json=body())
        assert r.status_code==status==r.json()['status']
        assert r.headers['content-type']=='application/problem+json'
        assert set(r.json())=={'type','title','status','detail'}
        if status==503: assert int(r.headers['retry-after'])>0


def test_classifier_failure_is_503_and_no_forwarding_or_state(router):
    def fail(*args):raise RuntimeError('private upstream info')
    router.classifier=fail
    r=client_for(router).post('/v1/routing/rank',json=body())
    assert r.status_code==503 and 'private' not in r.text
    router.classifier=None
    rank(router,body())
    assert not router.conversations and not router.decisions


def test_reasoning_omitted_without_effort_specific_predictions(router):
    router.config.raw={'models':[{'name':'small','irp_reasoning_efforts':['low']}]}
    for effort in ('low', 'max'):
        assert all('reasoning_effort' not in e for e in rank(router,body(reasoning_effort=effort))['ranked'])


def test_numeric_overflow_limits_and_multiple_completions(router):
    b=body();b['routing']['candidates'][0]['pricing']['input']=10**1000
    r=client_for(router).post('/v1/routing/rank',json=b);assert r.status_code==400
    b=body();b['routing']['candidates'][0]['expected_usage']['cache_read_tokens']=10**1000
    r=client_for(router).post('/v1/routing/rank',json=b);assert r.status_code==400
    r=client_for(router).post('/v1/routing/rank',content=' '*2_000_001)
    assert r.status_code==400
    one=rank(router,body())['ranked'];two=rank(router,body(n=2))['ranked']
    one={e['candidate_id']:e for e in one};two={e['candidate_id']:e for e in two}
    assert all(two[c]['expected_usage']['output_tokens']==2*one[c]['expected_usage']['output_tokens'] for c in one)


def test_irp_overflow_400_must_not_consume_classifier(router):
    calls=[]
    router.classifier=lambda *args:calls.append(args)
    b=body([candidate('a','test/small')]);b['routing']['candidates'][0]['pricing']['input']=1e308
    b['request']['messages'][0]['content']='x'*4000
    r=client_for(router).post('/v1/routing/rank',json=b)
    assert r.status_code==400 and 'numeric range' in r.json()['detail'] and calls==[]


@pytest.mark.parametrize('field', ['id', 'model'])
def test_surrogate_strings_are_invalid_problem_json(router,field):
    b=body();b['routing']['candidates'][0][field]='\ud800'
    r=client_for(router).post('/v1/routing/rank',content=__import__('json').dumps(b))
    assert r.status_code==400 and r.headers['content-type']=='application/problem+json'


def test_large_integers_and_nesting_are_400(router):
    for field in ('max_tokens','max_completion_tokens','n'):
        b=body(**{field:10**310})
        assert client_for(router).post('/v1/routing/rank',json=b).status_code==400
    for raw in ('{"request":'+ '1'*4400 +'}', '['*1200+'0'+']'*1200):
        assert client_for(router).post('/v1/routing/rank',content=raw).status_code==400


# --- aliases and the budgeted classifier (job antseed-router-v2) ---------------------------------
import threading
import time as _time

from auto_router import jev
from auto_router.irp import BudgetedClassifier, model_index


def aliased_router(aliases):
    models = [ModelInfo('small', 'a', 'test/small', Prices(1, 2), capability={'general': 40, 'coding': 40}),
              ModelInfo('large', 'b', 'test/large', Prices(3, 8), capability={'general': 70, 'coding': 90})]
    config = RouterConfig({}, Catalog(models), policy={'classifier': {'backend': 'heuristic'}},
                          raw={'models': [{'name': 'large', 'irp_aliases': aliases}]})
    r = Router(config)
    r.classifier = None
    return r


def test_aliases_resolve_to_the_same_model_and_are_listed(router):
    r = aliased_router(['e2ee-large', 'large-odd', 'test/small', 7, ''])
    index = model_index(r.config)
    assert index['e2ee-large'][0].name == 'large' and index['large-odd'][0].name == 'large'
    assert index['test/small'][0].name == 'small'  # an alias never shadows another model's primary id
    ids = {m['id'] for m in supported(r.config)['data']}
    assert {'test/small', 'test/large', 'e2ee-large', 'large-odd'} <= ids
    data = rank(r, body([candidate('x', 'e2ee-large'), candidate('y', 'test/small')]))
    assert {e['candidate_id'] for e in data['ranked']} == {'x', 'y'}


def test_models_extra_is_merged_into_the_models_answer(router):
    app = FastAPI()
    app.include_router(endpoints(lambda: router, None, lambda: {'catalog_models': 3}))
    extra = TestClient(app).get('/v1/routing/models').json()['extra']['system1models.ai']
    assert extra['catalog_models'] == 3 and extra['mode'] == 'suggest-only'


def cls(category='coding', difficulty=0.9, **kw):
    base = dict(category=category, category_probs={category: 1.0}, category_confidence=0.9, difficulty=difficulty,
                difficulty_confidence=0.8, needs_tools=0.0, needs_vision=0.0, needs_long_context=0.0,
                follow_up=0.0, stakes=0.5, latency_s=0.0, model='s1')
    base.update(kw)
    return jev.Classification(**base)


def budgeted(router, fn, **kw):
    router.classifier = BudgetedClassifier(fn, 's1-test@abc', **kw)
    return router


def basis(router, **req):
    return rank(router, body(**req))['extra']['system1models.ai']


def test_budgeted_classifier_reports_what_it_used(router):
    extra = basis(budgeted(router, lambda t, c: cls('coding', 0.9)))
    assert extra['classifier'] == 's1-test@abc' and extra['category'] == 'coding' and extra['difficulty'] > 0.5
    # the classified difficulty moves the success estimate relative to the prior
    easy = rank(budgeted(router, lambda t, c: cls('coding', 0.0)), body())['ranked']
    hard = rank(budgeted(router, lambda t, c: cls('coding', 1.0)), body())['ranked']
    q = lambda ranked, cid: next(e for e in ranked if e['candidate_id'] == cid)['expected_quality']
    assert q(easy, 'a') > q(hard, 'a')


@pytest.mark.parametrize('name,fn,reason', [
    ('error', lambda t, c: 1 / 0, 'error'),
    ('failed', lambda t, c: jev.FALLBACK, 'error'),
    ('malformed', lambda t, c: cls(difficulty=float('nan')), 'malformed'),
    ('malformed-type', lambda t, c: 'not a classification', 'malformed'),
])
def test_budgeted_classifier_falls_back_on_bad_answers(router, name, fn, reason):
    extra = basis(budgeted(router, fn))
    assert extra['classifier'] == f'local conservative prior (fallback: {reason})' and 'category' not in extra


def test_budgeted_classifier_times_out_and_still_ranks(router):
    release = threading.Event()
    budgeted(router, lambda t, c: (release.wait(5), cls())[1], timeout_s=0.05)
    started = _time.perf_counter()
    data = rank(router, body())
    release.set()
    assert _time.perf_counter() - started < 1.0 and len(data['ranked']) == 2
    assert data['extra']['system1models.ai']['classifier'] == 'local conservative prior (fallback: timeout)'


def test_budgeted_classifier_caps_concurrency_and_recovers(router):
    release = threading.Event()
    running = []
    def slow(t, c):
        running.append(1)
        release.wait(5)
        return cls()
    bc = BudgetedClassifier(slow, 's1-test@abc', timeout_s=0.05, concurrency=2)
    assert [bc.classify('x', '').basis.endswith('(fallback: timeout)') for _ in range(2)] == [True, True]
    assert bc.classify('x', '').basis == 'local conservative prior (fallback: busy)'  # pool full, no third call
    assert len(running) <= 2
    release.set()
    for _ in range(100):
        if bc.classify('x', '').classification is not None:
            break
        _time.sleep(0.02)
    else:
        raise AssertionError('slots were never released')


def test_validation_precedes_classification(router):
    calls = []
    budgeted(router, lambda t, c: calls.append(1) or cls())
    app = FastAPI(); app.include_router(endpoints(lambda: router)); client = TestClient(app)
    assert client.post('/v1/routing/rank', json={'request': {'messages': []}, 'routing': {'candidates': []}}).status_code == 400
    assert client.post('/v1/routing/rank', json={'request': {'messages': 5}, 'routing': {}}).status_code == 400
    assert client.post('/v1/routing/rank', content='{').status_code == 400
    assert calls == []  # malformed requests never consume a classifier call
    assert client.post('/v1/routing/rank', json=body()).status_code == 200 and calls == [1]


def test_tradeoff_stays_monotonic_with_classifier(router):
    budgeted(router, lambda t, c: cls('coding', 0.7))
    cands = [candidate('a', 'test/small'), candidate('b', 'test/large', 3)]
    costs = []
    for t in range(11):
        b = body(cands); b['routing']['cost_quality_tradeoff'] = t
        costs.append(rank(router, b)['ranked'][0]['expected_cost_usd'])
    assert costs == sorted(costs, reverse=True)


@pytest.mark.parametrize('field', [
    'difficulty', 'difficulty_confidence', 'stakes', 'needs_tools',
    'needs_vision', 'needs_long_context', 'follow_up',
])
@pytest.mark.parametrize('value', [-2.0, 1.01, 10**400, float('nan'), float('inf'),
                                   float('-inf'), None, 'invalid'])
def test_budgeted_classifier_rejects_unusable_numeric_fields(router, field, value):
    budgeted(router, lambda *_: cls(**{field: value}))
    try:
        outcome = router.classifier.classify('x', '')
        assert outcome.classification is None
        assert outcome.basis.endswith('(fallback: malformed)')
        app = FastAPI()
        app.include_router(endpoints(lambda: router))
        response = TestClient(app).post('/v1/routing/rank', json=body())
        assert response.status_code == 200
        assert len(response.json()['ranked']) == 2
        assert response.json()['extra']['system1models.ai']['classifier'] == outcome.basis
    finally:
        router.classifier._pool.shutdown()


def test_budgeted_classifier_rejects_missing_confidence(router):
    result = cls()
    del result.difficulty_confidence
    budgeted(router, lambda *_: result)
    try:
        assert basis(router)['classifier'].endswith('(fallback: malformed)')
    finally:
        router.classifier._pool.shutdown()


def test_budgeted_classifier_normalizes_numeric_strings_without_mutating_result(router):
    fields = ('difficulty', 'difficulty_confidence', 'stakes', 'needs_tools',
              'needs_vision', 'needs_long_context', 'follow_up')
    result = cls(**{field: '0.7' for field in fields})
    budgeted(router, lambda *_: result)
    try:
        outcome = router.classifier.classify('x', '')
        assert outcome.basis == 's1-test@abc'
        assert all(type(getattr(outcome.classification, field)) is float for field in fields)
        assert all(getattr(outcome.classification, field) == 0.7 for field in fields)
        assert all(getattr(result, field) == '0.7' for field in fields)
        actual = rank(router, body())
        router.classifier.classifier = lambda *_: cls(**{field: 0.7 for field in fields})
        expected = rank(router, body())
        assert actual['ranked'] == expected['ranked']
        assert actual['extra'] == expected['extra']
    finally:
        router.classifier._pool.shutdown()


@pytest.mark.parametrize('stakes,expected', [(-2.0, 0.05), (2.0, 20.0)])
def test_turn_request_clamps_stakes_index(router, stakes, expected):
    request = router._turn_request(cls(stakes=stakes), 10, 100, 0, False,
                                   [{'role': 'user', 'content': 'hello'}])
    assert request.stakes_usd == expected
