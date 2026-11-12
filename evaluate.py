from engine import FAQBot

# (query, expected FAQ id, or None = should go to a human)
TESTS = [
    ("how long does delivery take", 2),
    ("can I send back something I bought", 1),
    ("where is my package right now", 3),
    ("do you accept UPI or cash on delivery", 4),
    ("I want to cancel what I ordered", 5),
    ("how many days for money to come back to my account", 6),
    ("can you deliver to the USA", 7),
    ("is there a guarantee on electronics", 8),
    ("I entered the wrong address, can I change it", 9),
    ("my discount coupon is not working", 10),
    ("can't log in, forgot my password", 11),
    ("what time is support open", 12),
    ("how do I return an item", 1),
    ("how to track my shipment", 3),
    # out of scope, should be referred
    ("Can you recommend a good laptop under 50000?", None),
    ("What's the weather in Delhi today?", None),
    ("I want to file a legal complaint against your company", None),
    ("My parcel arrived damaged and I want compensation", None),
    ("Do you have a job opening for data analysts?", None),
    ("Who won the cricket match yesterday?", None),
    ("Can I get a bulk discount for 500 units?", None),
    ("Write me a poem about the ocean", None),
]


def evaluate(bot, t):
    correct = good_refer = missed_refer = needless_refer = wrong_faq = 0
    for q, exp in TESTS:
        r = bot.respond(q, threshold=t)
        referred = r.status == "REFER_TO_HUMAN"
        if exp is None:
            if referred: correct += 1; good_refer += 1
            else: missed_refer += 1          # worst error
        else:
            if referred: needless_refer += 1
            elif r.matches[0][0]["id"] == exp: correct += 1
            else: wrong_faq += 1
    return correct / len(TESTS), good_refer, missed_refer, needless_refer, wrong_faq


if __name__ == "__main__":
    bot = FAQBot()
    print(f"{'thresh':>6} {'acc':>6} {'good_ref':>9} {'missed':>7} {'needless':>9} {'wrongFAQ':>9}")
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        acc, g, m, n, w = evaluate(bot, t)
        print(f"{t:>6.2f} {acc:>6.0%} {g:>9} {m:>7} {n:>9} {w:>9}")

    print("\nPer-query scores:")
    for q, exp in TESTS:
        r = bot.respond(q)
        print(f"{r.confidence:.2f} {r.status:<15} exp={'HUMAN' if exp is None else exp!s:<5} {q}")
       