"""A small simulated order-management system for tool-use tasks with a verifiable end state.

Each task is generated from a seed. The same seed builds the same world; a
reference solver applies the intended changes to a copy, and grading compares
the model's final world with the reference world (orders, refunds, created
orders) plus required facts in the messages sent.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field

FIRST = ["Ada", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lea", "Max", "Nia", "Oli", "Pia"]
LAST = ["Stone", "Rivera", "Okafor", "Lind", "Novak", "Sato", "Berg", "Moreau", "Kim", "Silva", "Weber", "Hart"]
SKUS = [f"SKU-{i:03d}" for i in range(100, 140)]

TOOLS = [
    {"type": "function", "function": {"name": "find_customer", "description": "Look up a customer by email.",
     "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}}},
    {"type": "function", "function": {"name": "list_orders", "description": "List all orders of a customer.",
     "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}}},
    {"type": "function", "function": {"name": "get_stock", "description": "Units in stock and unit price for a SKU.",
     "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}}},
    {"type": "function", "function": {"name": "cancel_order", "description": "Cancel a pending order. Stock is returned.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "refund", "description": "Refund an amount (USD) on a delivered order.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                    "required": ["order_id", "amount"]}}},
    {"type": "function", "function": {"name": "create_order", "description": "Create a pending order; reserves stock.",
     "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "object", "properties": {
                        "sku": {"type": "string"}, "qty": {"type": "integer"}}, "required": ["sku", "qty"]}}},
                    "required": ["customer_id", "items"]}}},
    {"type": "function", "function": {"name": "send_message", "description": "Send a message to a customer.",
     "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["customer_id", "text"]}}},
    {"type": "function", "function": {"name": "done", "description": "Call when the task is complete.",
     "parameters": {"type": "object", "properties": {}}}},
]


@dataclass
class World:
    customers: dict = field(default_factory=dict)
    orders: dict = field(default_factory=dict)
    stock: dict = field(default_factory=dict)
    refunds: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    next_order: int = 9000

    def to_json(self) -> dict:
        return {"customers": self.customers, "orders": self.orders, "stock": self.stock,
                "refunds": self.refunds, "messages": self.messages, "next_order": self.next_order}

    @staticmethod
    def from_json(d: dict) -> "World":
        return World(copy.deepcopy(d["customers"]), copy.deepcopy(d["orders"]), copy.deepcopy(d["stock"]),
                     copy.deepcopy(d["refunds"]), copy.deepcopy(d["messages"]), d["next_order"])

    # -- tools -------------------------------------------------------------
    def call(self, name: str, args: dict) -> str:
        try:
            return json.dumps(getattr(self, "t_" + name)(**args))
        except TypeError as exc:
            return json.dumps({"error": f"bad arguments: {exc}"})
        except AttributeError:
            return json.dumps({"error": f"unknown tool {name}"})

    def t_find_customer(self, email):
        for c in self.customers.values():
            if c["email"].lower() == str(email).lower():
                return c
        return {"error": "no such customer"}

    def t_list_orders(self, customer_id):
        return [o for o in self.orders.values() if o["customer_id"] == customer_id]

    def t_get_stock(self, sku):
        return {"sku": sku, **self.stock[sku]} if sku in self.stock else {"error": "unknown sku"}

    def t_cancel_order(self, order_id):
        o = self.orders.get(order_id)
        if not o:
            return {"error": "no such order"}
        if o["status"] != "pending":
            return {"error": f"order is {o['status']}, only pending orders can be cancelled"}
        o["status"] = "cancelled"
        for it in o["items"]:
            self.stock[it["sku"]]["units"] += it["qty"]
        return {"ok": True}

    def t_refund(self, order_id, amount):
        o = self.orders.get(order_id)
        if not o:
            return {"error": "no such order"}
        if o["status"] != "delivered":
            return {"error": "only delivered orders can be refunded"}
        already = sum(r["amount"] for r in self.refunds if r["order_id"] == order_id)
        if float(amount) <= 0 or already + float(amount) > o["total"] + 1e-6:
            return {"error": "refund exceeds order total"}
        self.refunds.append({"order_id": order_id, "amount": round(float(amount), 2)})
        return {"ok": True}

    def t_create_order(self, customer_id, items):
        if customer_id not in self.customers:
            return {"error": "no such customer"}
        total = 0.0
        for it in items:
            s = self.stock.get(it.get("sku"))
            if not s or int(it.get("qty", 0)) <= 0 or s["units"] < int(it["qty"]):
                return {"error": f"insufficient stock for {it.get('sku')}"}
        for it in items:
            self.stock[it["sku"]]["units"] -= int(it["qty"])
            total += self.stock[it["sku"]]["price"] * int(it["qty"])
        oid = f"O-{self.next_order}"
        self.next_order += 1
        self.orders[oid] = {"order_id": oid, "customer_id": customer_id, "status": "pending",
                            "items": [{"sku": it["sku"], "qty": int(it["qty"])} for it in items],
                            "total": round(total, 2), "created": "2026-09-17"}
        return {"ok": True, "order_id": oid}

    def t_send_message(self, customer_id, text):
        self.messages.append({"customer_id": customer_id, "text": str(text)})
        return {"ok": True}

    def t_done(self):
        return {"ok": True}


def make_world(rng: random.Random) -> World:
    w = World()
    for sku in SKUS:
        w.stock[sku] = {"units": rng.randint(0, 12), "price": round(rng.uniform(5, 120), 2)}
    for i in range(12):
        cid = f"C-{100 + i}"
        name = f"{FIRST[i % len(FIRST)]} {rng.choice(LAST)}"
        w.customers[cid] = {"customer_id": cid, "name": name,
                            "email": f"{name.split()[0].lower()}.{i}@example.com",
                            "tier": rng.choice(["standard", "standard", "gold"])}
    n = 8000
    for cid in w.customers:
        for _ in range(rng.randint(3, 7)):
            items = [{"sku": rng.choice(SKUS), "qty": rng.randint(1, 3)} for _ in range(rng.randint(1, 3))]
            total = round(sum(w.stock[it["sku"]]["price"] * it["qty"] for it in items), 2)
            w.orders[f"O-{n}"] = {"order_id": f"O-{n}", "customer_id": cid,
                                  "status": rng.choice(["pending", "pending", "delivered", "delivered", "cancelled"]),
                                  "items": items, "total": total,
                                  "created": f"2026-0{rng.randint(6, 9)}-{rng.randint(10, 28)}"}
            n += 1
    return w


def build_task(level: str, seed: int) -> dict:
    """Returns {prompt, world, reference, required_message_facts}."""
    rng = random.Random(seed)
    for _ in range(200):
        w = make_world(rng)
        cid = rng.choice(list(w.customers))
        cust = w.customers[cid]
        orders = sorted(w.t_list_orders(cid), key=lambda o: (o["created"], o["order_id"]))
        ref = World.from_json(w.to_json())
        facts: list[str] = []
        if level == "easy":
            pending = [o for o in orders if o["status"] == "pending"]
            if not pending:
                continue
            target = rng.choice(pending)
            prompt = (f"Customer {cust['email']} asked us to cancel order {target['order_id']}. Please do it, "
                      f"then call done.")
            ref.t_cancel_order(target["order_id"])
        elif level == "medium":
            delivered = [o for o in orders if o["status"] == "delivered"]
            if not delivered:
                continue
            latest = max(delivered, key=lambda o: (o["created"], o["order_id"]))
            prompt = (f"The customer with email {cust['email']} says their most recently created delivered order "
                      f"arrived damaged. Refund that order in full, then send the customer one message that "
                      f"states the order id and the refunded amount in dollars. Then call done.")
            ref.t_refund(latest["order_id"], latest["total"])
            ref.t_send_message(cid, "")
            facts = [latest["order_id"], f"{latest['total']:.2f}"]
        else:  # hard
            pending = [o for o in orders if o["status"] == "pending"]
            skus = [it["sku"] for o in pending for it in o["items"]]
            if len(pending) < 2 or not skus:
                continue
            old = rng.choice(skus)
            new = rng.choice([s for s in SKUS if s != old])
            affected = [o for o in pending if any(it["sku"] == old for it in o["items"])]
            if not affected:
                continue
            gold = cust["tier"] == "gold"
            prompt = (
                f"Our supplier discontinued {old}. For customer {cust['email']}, handle every PENDING order that "
                f"contains {old}, oldest order first (by created date, then order id): replace {old} with {new} at "
                f"the same quantity by cancelling the order and creating a new order with identical other items. "
                f"Only do this if {new} has enough units: the units currently in stock plus any units of {new} that "
                f"cancelling that same order would free must cover everything the new order needs (check "
                f"get_stock right before each replacement). If not, leave that order untouched. "
                + ("This is a gold-tier customer: additionally refund $5 on their most recently created delivered "
                   "order if they have one. " if gold else "Do not issue any refunds. ")
                + "Finally send exactly one message to the customer listing the ids of the NEW orders you created "
                  "and the ids of orders you could not switch. Then call done.")
            created, skipped = [], []
            for o in sorted(affected, key=lambda o: (o["created"], o["order_id"])):
                qty = sum(it["qty"] for it in o["items"] if it["sku"] == old)
                others = [it for it in o["items"] if it["sku"] != old]
                # stock of `new` after cancelling this order (the order may itself hold `new`)
                returned = sum(it["qty"] for it in o["items"] if it["sku"] == new)
                need_new = qty + sum(it["qty"] for it in others if it["sku"] == new)
                if ref.stock[new]["units"] + returned >= need_new:
                    ref.t_cancel_order(o["order_id"])
                    merged: dict[str, int] = {}
                    for it in others + [{"sku": new, "qty": qty}]:
                        merged[it["sku"]] = merged.get(it["sku"], 0) + it["qty"]
                    items = [{"sku": s, "qty": q} for s, q in merged.items()]
                    res = ref.t_create_order(cid, items)
                    if "order_id" not in res:  # another item ran out: roll back to skip
                        return build_task(level, seed + 7919)
                    created.append(res["order_id"])
                else:
                    skipped.append(o["order_id"])
            if gold:
                delivered = [o for o in orders if o["status"] == "delivered"]
                if delivered:
                    latest = max(delivered, key=lambda o: (o["created"], o["order_id"]))
                    ref.t_refund(latest["order_id"], 5)
            ref.t_send_message(cid, "")
            facts = created + skipped
        return {"prompt": prompt, "world": w.to_json(), "reference": ref.to_json(),
                "customer_id": cid, "facts": facts}
    raise RuntimeError("could not build task")


def _normalise_orders(world: dict) -> list:
    """Order state without auto-assigned ids of created orders (compared by content)."""
    out = []
    for o in world["orders"].values():
        if int(o["order_id"].split("-")[1]) >= 9000:
            out.append(("new", o["customer_id"], o["status"], tuple(sorted((i["sku"], i["qty"]) for i in o["items"]))))
        else:
            out.append((o["order_id"], o["status"]))
    return sorted(out)


def grade(task: dict, final: dict) -> tuple[bool, str]:
    ref = task["reference"]
    if _normalise_orders(final) != _normalise_orders(ref):
        return False, "order state differs from reference"
    got_refunds = sorted((r["order_id"], round(r["amount"], 2)) for r in final["refunds"])
    want_refunds = sorted((r["order_id"], round(r["amount"], 2)) for r in ref["refunds"])
    if got_refunds != want_refunds:
        return False, f"refunds {got_refunds} != {want_refunds}"
    want_notes = [m for m in ref["messages"]]
    got_notes = [m for m in final["messages"] if m["customer_id"] == task["customer_id"]]
    if want_notes and (len(got_notes) != len(want_notes) or len(final["messages"]) != len(want_notes)):
        return False, f"{len(final['messages'])} messages sent, expected {len(want_notes)}"
    if want_notes:
        text = got_notes[0]["text"]
        # created order ids differ between runs; check them by count of O- ids instead
        new_ids = [o["order_id"] for o in final["orders"].values() if int(o["order_id"].split("-")[1]) >= 9000]
        ref_new = [o["order_id"] for o in ref["orders"].values() if int(o["order_id"].split("-")[1]) >= 9000]
        facts = [f for f in task["facts"] if f not in ref_new] + new_ids
        missing = [f for f in facts if f not in text]
        if missing:
            return False, f"message misses {missing}"
    return True, "ok"
