
import pytest
from doc_processor.app.llm_service import (
    classify_conversational_intent,
    is_greeting_or_thanks,
    is_lead_worthy_question,
)

COMPANY = "Acme Corp"


# ---------------------------------------------------------------------------
# 1. Things that MUST be caught as smalltalk
# ---------------------------------------------------------------------------
SHOULD_BE_SMALLTALK = [
    "hi", "hii", "hiii", "Hello", "HELLO!!", "hey", "heyy", "hey there",
    "yo", "sup", "whats up", "good morning", "morning", "howdy", "hola",
    "thanks", "thank you", "thx", "ty", "tysm", "ok", "okay", "cool",
    "great, thanks", "got it", "sounds good", "that helps",
    "bye", "goodbye", "see ya", "gtg", "no thanks", "nope", "i'm good",
    "nothing else", "that's all",
    "who are you", "what are you", "are you a bot", "are you real",
    "what can you do", "how can you help", "help",
]

@pytest.mark.parametrize("msg", SHOULD_BE_SMALLTALK)
def test_smalltalk_is_caught(msg):
    result = classify_conversational_intent(msg, company_name=COMPANY)
    assert result["is_smalltalk"] is True, f"'{msg}' should be smalltalk, got {result}"
    assert result["reply"], f"'{msg}' matched but produced no reply"


# ---------------------------------------------------------------------------
# 2. Things that MUST NOT be caught — real questions with a greeting prefix
#    or a smalltalk word buried inside a real sentence.
# ---------------------------------------------------------------------------
SHOULD_NOT_BE_SMALLTALK = [
    "hi, do you do AI consulting?",
    "hello, what's your pricing for the pro plan?",
    "hey can you tell me about your refund policy",
    "thanks, but what about international shipping?",
    "no, I meant the enterprise tier",
    "help me understand how billing works",
    "what can you tell me about your API rate limits",
    "I'm looking for a document management solution",
    "I am not sure how to integrate your webhook",
]

@pytest.mark.parametrize("msg", SHOULD_NOT_BE_SMALLTALK)
def test_real_questions_pass_through(msg):
    result = classify_conversational_intent(msg, company_name=COMPANY)
    assert result["is_smalltalk"] is False, f"'{msg}' should NOT be smalltalk, got {result}"


# ---------------------------------------------------------------------------
# 3. Self-intro: must extract real names, must NOT extract false positives
# ---------------------------------------------------------------------------
SELF_INTRO_VALID = [
    ("my name is John", "John"),
    ("I'm Sarah", "Sarah"),
    ("hi, this is Raj", "Raj"),
    ("call me Mike", "Mike"),
]

@pytest.mark.parametrize("msg,expected_name", SELF_INTRO_VALID)
def test_self_intro_extracts_name(msg, expected_name):
    result = classify_conversational_intent(msg, company_name=COMPANY)
    assert result["intent"] == "self_intro", f"'{msg}' -> {result}"
    assert result["memory_update"]["visitor_name"] == expected_name

# The bug we specifically fixed: "I'm <adjective/verb>..." must NOT become a name
SELF_INTRO_FALSE_POSITIVES = [
    "I'm interested in your data services",
    "I am not sure what plan I need",
    "I'm looking for pricing info",
    "I am trying to reset my password",
    "I'm a customer of yours",
]

@pytest.mark.parametrize("msg", SELF_INTRO_FALSE_POSITIVES)
def test_self_intro_false_positives_rejected(msg):
    result = classify_conversational_intent(msg, company_name=COMPANY)
    assert result["intent"] != "self_intro", f"'{msg}' wrongly extracted a name: {result}"
    # These are real questions/statements, so they must fall through to RAG
    assert result["is_smalltalk"] is False


# ---------------------------------------------------------------------------
# 4. First-turn greeting vs returning-turn greeting must differ
# ---------------------------------------------------------------------------
def test_first_turn_greeting_uses_full_intro():
    r = classify_conversational_intent("hi", company_name=COMPANY, is_first_turn=True)
    assert COMPANY in r["reply"]

def test_later_turn_greeting_is_short():
    r = classify_conversational_intent("hi", company_name=COMPANY, is_first_turn=False)
    assert "I'm" not in r["reply"] or "Assistant" not in r["reply"]  # shouldn't re-intro


# ---------------------------------------------------------------------------
# 5. Visitor name gets used once known
# ---------------------------------------------------------------------------
def test_greeting_uses_known_visitor_name():
    r = classify_conversational_intent("hi", company_name=COMPANY, visitor_name="Sarah")
    assert "Sarah" in r["reply"]


# ---------------------------------------------------------------------------
# 6. is_greeting_or_thanks — used inside the lead-capture reprompt branch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("msg", ["hi", "hello", "thanks", "ok", "bye"])
def test_greeting_or_thanks_true(msg):
    assert is_greeting_or_thanks(msg) is True

@pytest.mark.parametrize("msg", [
    "john@example.com", "my email is john@example.com", "John Smith",
    "555-123-4567",
])
def test_greeting_or_thanks_false_on_contact_info(msg):
    assert is_greeting_or_thanks(msg) is False


# ---------------------------------------------------------------------------
# 7. is_lead_worthy_question — the NO_ANSWER gate
# ---------------------------------------------------------------------------
NOT_LEAD_WORTHY = [
    "hi", "thanks", "bye", "no thanks", "ok", "asdfgh", "qwertyzxcv",
    "123", "🙂", "   ", "aaaa", "test",
]

@pytest.mark.parametrize("msg", NOT_LEAD_WORTHY)
def test_gibberish_and_smalltalk_not_lead_worthy(msg):
    assert is_lead_worthy_question(msg) is False, f"'{msg}' should NOT open lead capture"

LEAD_WORTHY = [
    "do you integrate with Salesforce?",
    "what's the cost of the enterprise plan",
    "can I get a refund",
    "how do I cancel my subscription",
    "is there a mobile app",
]

@pytest.mark.parametrize("msg", LEAD_WORTHY)
def test_real_questions_are_lead_worthy(msg):
    assert is_lead_worthy_question(msg) is True, f"'{msg}' SHOULD open lead capture"