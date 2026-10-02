#!/usr/bin/env python3
"""
Test file: Validate student implementations of NGramCharLM class in hw3.py

This script trains an N-gram language model on a LOCAL text corpus and tests
various functionalities.

Usage:
    python test_ngram.py
    python test_ngram.py --corpus sample_text.txt
"""

import sys
import os
import argparse

# Import student implementation
try:
    from hw3 import NGramCharLM
except ImportError:
    print("Error: Cannot import NGramCharLM class from hw3.py")
    print("Please ensure hw3.py is in the same directory and NGramCharLM class is properly implemented")
    sys.exit(1)


BUILTIN_SAMPLE_TEXT = (
    "The weather station blinked twice.\n"
    "A river rose, then fell, then rose again.\n"
    "We wrote notes on the margin: flow, slope, width.\n"
    "\n"
    "In the lab, a model learns patterns from small hints.\n"
    "In the town, a clock strikes ten, and the street goes quiet.\n"
    "Do you hear the wind? Do you hear the rain?\n"
    "\n"
    "When we say hello, we often say it again: hello, hello.\n"
    "When we say good night, we often whisper: good night.\n"
    "The end.\n"
)


def load_local_corpus(corpus_path: str) -> str:
    """
    Load training text from a local file. If missing, use BUILTIN_SAMPLE_TEXT.
    """
    if corpus_path and os.path.exists(corpus_path):
        print(f"Loading local corpus from: {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        print(f"Successfully loaded corpus, total characters: {len(text)}")
        return text

    print(f"Corpus file not found at '{corpus_path}'. Using built-in sample text instead...")
    print(f"Built-in corpus characters: {len(BUILTIN_SAMPLE_TEXT)}")
    return BUILTIN_SAMPLE_TEXT


def test_ngram_implementation(text: str) -> bool:
    """Test NGramCharLM class implementation"""

    print("\n" + "=" * 60)
    print("Starting NGramCharLM implementation tests")
    print("=" * 60)

    # Test 1: Model initialization and training
    print("\n1. Testing model initialization and training...")
    try:
        model = NGramCharLM(n=10)
        print(f"   ✓ Successfully created 10-gram model")
        print(f"   Initial state - trained: {model.trained}, vocab size: {len(model.vocab)}")

        model.fit(text)
        print(f"   ✓ Training completed")
        print(f"   Post-training state - trained: {model.trained}, vocab size: {len(model.vocab)}")
        print(f"   Number of contexts: {len(model.counts)}")

    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        return False

    # Test 2: Probability calculation
    print("\n2. Testing probability calculation...")
    try:
        test_string = "hello world\n"
        log_prob = model.logprob(test_string)
        prob = model.prob(test_string)

        print(f"   Test string: {repr(test_string)}")
        print(f"   ✓ Log probability: {log_prob}")
        print(f"   ✓ Probability (may underflow): {prob}")

        empty_log_prob = model.logprob("")
        print(f"   Empty string log probability: {empty_log_prob}")

    except Exception as e:
        print(f"   ✗ Probability calculation failed: {e}")
        return False

    # Test 3: Next character distribution
    print("\n3. Testing next character probability distribution...")
    try:
        context = "hello "
        distribution = model.next_char_distribution(context)

        print(f"   Context: {repr(context)}")
        print(f"   ✓ Got probability distribution with {len(distribution)} characters")

        print("   Top 10 most likely next characters:")
        count = 0
        for char, p in distribution.items():
            if count >= 10:
                break
            display_char = char if char != "\n" else "\\n"
            print(f"      {repr(display_char):>6} : {p:.6f}")
            count += 1

    except Exception as e:
        print(f"   ✗ Next character distribution calculation failed: {e}")
        return False

    # Test 4: Text generation
    print("\n4. Testing text generation...")
    seed = "The "
    try:
        generated = model.generate(300, seed=seed)
        print(f"   Seed text: {repr(seed)}")
        print(f"   ✓ Successfully generated {len(generated) - len(seed)} new characters")
        print(f"   Generated text length: {len(generated)}")

        print("\n   Generated text sample:")
        print("   " + "-" * 50)
        display_text = generated[:300]
        if len(generated) > 300:
            display_text += "..."
        print("   " + display_text.replace("\n", "\n   "))
        print("   " + "-" * 50)

    except Exception as e:
        print(f"   ✗ Text generation failed: {e}")
        return False

    # Test 5: Edge cases
    print("\n5. Testing edge cases...")
    try:
        for n in [1, 2, 3]:
            small_model = NGramCharLM(n=n)
            small_model.fit("hello world")
            small_prob = small_model.logprob("hello")
            print(f"   ✓ {n}-gram model works, 'hello' log probability: {small_prob:.4f}")
    except Exception as e:
        print(f"   ✗ Edge case testing failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("🎉 All tests passed! Your NGramCharLM implementation looks correct.")
    print("=" * 60)
    return True


def run_interactive_demo(text: str):
    """Run interactive demo with:
       - dist <context> : show top-10 next-char distribution
       - gen <seed> <k> : generate next k chars after seed (seed can contain spaces)
       - help / quit
    """
    print("\nWould you like to run an interactive demo? (y/n): ", end="")
    try:
        choice = input().lower().strip()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return

    if choice != "y":
        return

    print("\nStarting interactive demo...")
    model = NGramCharLM(n=3)  # recommend 4~6 for short corpora
    model.fit(text)

    print("\nCommands:")
    print("  dist <context>         Show next-character distribution (top 10)")
    print("  gen <seed> <k>         Generate next k characters after seed")
    print("                         (seed can include spaces; k must be an integer)")
    print("  help                   Show this help")
    print("  quit                   Exit\n")

    while True:
        try:
            line = input("Command> ").strip()
            if not line:
                continue

            if line.lower() in ("quit", "exit"):
                break

            if line.lower() == "help":
                print("\nCommands:")
                print("  dist <context>         Show next-character distribution (top 10)")
                print("  gen <seed> <k>         Generate next k characters after seed")
                print("  help                   Show this help")
                print("  quit                   Exit\n")
                continue

            # Default: if user types raw text without "dist", treat it as dist <context>
            if not (line.startswith("dist ") or line.startswith("gen ")):
                line = "dist " + line

            if line.startswith("dist "):
                context = line[len("dist "):]
                distribution = model.next_char_distribution(context)

                print(f"\nTop 10 next characters for context: {repr(context)}")
                count = 0
                for ch, p in distribution.items():
                    if count >= 10:
                        break
                    display_ch = ch if ch != "\n" else "\\n"
                    print(f"  {repr(display_ch):>6} : {p:.6f}")
                    count += 1
                print()
                continue

            if line.startswith("gen "):
                rest = line[len("gen "):].rstrip()

                # Parse k as the last token; seed is everything before it
                parts = rest.split()
                if len(parts) < 2:
                    print("Usage: gen <seed> <k>\n")
                    continue

                try:
                    k = int(parts[-1])
                except ValueError:
                    print("Error: k must be an integer. Usage: gen <seed> <k>\n")
                    continue

                seed = rest[:rest.rfind(parts[-1])].rstrip()  # everything before last token
                if k < 0:
                    print("Error: k must be non-negative.\n")
                    continue

                generated = model.generate(k, seed=seed)
                continuation = generated[len(seed):]

                print(f"\nSeed: {repr(seed)}")
                print(f"Generated next {k} chars:\n{continuation}\n")
                continue

            print("Unknown command. Type 'help' for options.\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

    print("\nDemo ended, goodbye!")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="sample_text.txt", help="Path to local training corpus")
    args = parser.parse_args()

    print("NGramCharLM Implementation Test Script (Local Corpus)")
    print("=" * 50)

    text = load_local_corpus(args.corpus)
    success = test_ngram_implementation(text)

    if success:
        run_interactive_demo(text)
    else:
        print("\nPlease fix the above errors and rerun the test.")
        sys.exit(1)


if __name__ == "__main__":
    main()
