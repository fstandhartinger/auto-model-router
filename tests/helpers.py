from auto_router.catalog import CacheRules, Catalog, ModelInfo, Prices

ANTHROPIC = CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=1.0)
OPENAI = CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=1.0)


def model(name, inp, out, read=None, write=None, cap=50.0, cache=OPENAI, **kw):
    return ModelInfo(name=name, provider="p", upstream_id=name, prices=Prices(inp, out, read, write),
                     cache=cache, capability={"general": cap, "coding": cap, "agentic": cap}, **kw)


CHEAP = model("cheap", 0.2, 1.0, read=0.02, cap=40)
MID = model("mid", 1.0, 5.0, read=0.1, cap=55)
FRONTIER = model("frontier", 5.0, 25.0, read=0.5, write=6.25, cap=70, cache=ANTHROPIC)
SUB = ModelInfo(name="sub", provider="p", upstream_id="sub", prices=Prices.free(), cache=ANTHROPIC,
                capability={"general": 70, "coding": 70, "agentic": 70}, subscription="claude")


def catalog(*models):
    return Catalog(list(models) or [CHEAP, MID, FRONTIER])
