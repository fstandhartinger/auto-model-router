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
