# container: scipy
"""What does it cost to label a dataset with an LLM?

Prices change often, so they are parameters, not constants. Look up your
provider's current price per million input and output tokens, and its batch
discount, before you run this.

    python week7-llm-labelling-cost.py --items 100000 --tokens-in 600 --tokens-out 20 \
        --price-in 3 --price-out 15 --batch-discount 50

With no arguments it uses these example values, the same defaults as the
calculator on the slide. Measure the tokens on a few real items first (most
APIs report them in the response), and hand-label a random sample to check
the LLM's accuracy before you pay for the rest.
"""

import argparse


def labelling_cost(items, tokens_in, tokens_out, price_in, price_out, batch_discount=0.0):
    """Cost in dollars; prices are per million tokens, the discount in percent."""
    cost_in = items * tokens_in * price_in / 1e6
    cost_out = items * tokens_out * price_out / 1e6
    total = cost_in + cost_out
    return {
        "input": cost_in,
        "output": cost_out,
        "total": total,
        "batch_total": total * (1 - batch_discount / 100),
        "output_share": cost_out / total if total else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--items", type=int, default=100_000, help="number of items to label")
    p.add_argument("--tokens-in", type=float, default=600, help="input tokens per item (instructions + item)")
    p.add_argument("--tokens-out", type=float, default=20, help="output tokens per item (the label)")
    p.add_argument("--price-in", type=float, default=3.0, help="$ per million input tokens (an example price)")
    p.add_argument("--price-out", type=float, default=15.0, help="$ per million output tokens (an example price)")
    p.add_argument("--batch-discount", type=float, default=50.0, help="batch API discount, percent")
    a = p.parse_args()
    print(f"{a.items:,} items, {a.tokens_in:g} input and {a.tokens_out:g} output tokens each, "
          f"${a.price_in:g} and ${a.price_out:g} per million tokens, {a.batch_discount:g}% batch discount")
    c = labelling_cost(a.items, a.tokens_in, a.tokens_out, a.price_in, a.price_out, a.batch_discount)
    print(f"Input tokens:  ${c['input']:,.2f}")
    print(f"Output tokens: ${c['output']:,.2f}")
    print(f"Real-time total: ${c['total']:,.2f} (${c['total'] / a.items * 1000:,.2f} per 1,000 items)")
    print(f"Batch total: ${c['batch_total']:,.2f}")
    print(f"Output tokens: {c['output_share']:.0%} of the cost")


if __name__ == "__main__":
    main()
