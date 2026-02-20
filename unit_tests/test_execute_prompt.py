import unittest

from execute_prompt import _trim_conversation_context


class TestTrimConversationContext(unittest.TestCase):
    def test_none_or_empty(self):
        self.assertIsNone(_trim_conversation_context(None))
        self.assertIsNone(_trim_conversation_context(""))
        self.assertIsNone(_trim_conversation_context("   \n  \n"))

    def test_fewer_than_max_pairs(self):
        ctx = "\n".join([
            "User: q1",
            "Assistant: a1",
            "User: q2",
            "Assistant: a2",
        ])
        # With only 4 lines and default max_pairs=6, everything should be kept.
        trimmed = _trim_conversation_context(ctx)
        self.assertEqual(trimmed, ctx)

    def test_exact_max_pairs(self):
        # 6 pairs -> 12 lines. All should be kept.
        lines = []
        for i in range(1, 7):
            lines.append(f"User: q{i}")
            lines.append(f"Assistant: a{i}")
        ctx = "\n".join(lines)
        trimmed = _trim_conversation_context(ctx, max_pairs=6)
        self.assertEqual(trimmed, ctx)

    def test_trims_to_last_max_pairs(self):
        # Build 10 pairs (20 lines); with max_pairs=6 we expect the last 12 lines.
        lines = []
        for i in range(1, 11):
            lines.append(f"User: q{i}")
            lines.append(f"Assistant: a{i}")
        ctx = "\n".join(lines)

        trimmed = _trim_conversation_context(ctx, max_pairs=6)

        # Expected to keep only pairs 5..10 (i.e., last 12 lines)
        expected_lines = []
        for i in range(5, 11):
            expected_lines.append(f"User: q{i}")
            expected_lines.append(f"Assistant: a{i}")
        expected = "\n".join(expected_lines)

        self.assertEqual(trimmed, expected)


if __name__ == "__main__":
    unittest.main()

