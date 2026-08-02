"""SETU's culturally-localised climate-argument taxonomy.

This is the intellectual core of the approach (contribution C1 in ../STRATEGY.md).

Rationale
---------
Zero-shot cross-lingual stance transfer for climate discourse fails less because of
*language* mismatch than because of **argument-inventory** mismatch. The permitted
English corpora (GWSD = US news editorials, SemEval-2016 = US political Twitter)
encode a US-centric contrarian inventory: *hoax, Al Gore, liberal agenda, snow in
Texas*. Indian YouTube commenters reject the same claim for entirely different
reasons: *the West polluted first, development before environment, the monsoon was
always erratic, yuga cycles, TRP drama*. Machine-translating GWSD produces fluent
Hindi sentences about Al Gore, which teaches the model nothing about the test set.

So we pivot through the argument rather than the language. The taxonomy below:

  * extends the CARDS taxonomy of contrarian claims (Coan et al., *Scientific
    Reports* 11:22320, 2021 — five super-claims: it's not real / not us / not bad /
    solutions won't work / science is unreliable) with an **India-specific branch**;
  * adds a **mirrored pro-climate branch** grounded in Indian lived experience,
    because Indian `Favour` arguments are also absent from the English data;
  * adds an explicit **`None` inventory**, which is where most systems bleed macro-F1.

It is consumed in three places, and that triple use is what makes it a contribution
rather than a lexicon appendix:

  1. `synth_generate.py` — as a *generative schema* for class-balanced, culturally
     grounded Hindi/Bengali YouTube comments (fixes the `Against` scarcity that caps
     macro-F1 near 0.45).
  2. `train_transformer.py` — as *auxiliary supervision*: a second head predicts the
     argument node, regularising the encoder toward argument structure rather than
     surface lexical cues.
  3. `llm_committee.py` — as the *reasoning scaffold* in the committee prompt, so the
     LLMs commit to a named argument before committing to a stance.

Every node carries Indian-context cue phrases in English, Hindi and Bengali. These
are prompt seeds and interpretability anchors — **not** a keyword classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """One argument type.

    id       : stable identifier, also the auxiliary-head class name
    stance   : the stance the argument entails w.r.t. the target claim
    branch   : 'cards' (from Coan et al.) | 'india' | 'pro' | 'none'
    gloss    : one-line English description, used verbatim in prompts
    cues     : indicative phrasings per language; prompt seeds, not rules
    """
    id: str
    stance: str
    branch: str
    gloss: str
    cues: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AGAINST — CARDS super-claims (Coan et al. 2021), adapted to comment register
# ---------------------------------------------------------------------------
_CARDS = [
    Node("A1_not_happening", "Against", "cards",
         "Denies that warming/climate change is happening at all; cites cold weather, "
         "snow, or a normal season as disproof.",
         {"en": ["it's not even happening", "this winter was freezing", "no warming since"],
          "hi": ["इस साल तो ठंड ज्यादा पड़ी", "कुछ नहीं हो रहा", "सब नॉर्मल है"],
          "bn": ["এ বছর তো বেশি শীত পড়েছে", "কিছুই হচ্ছে না", "সব স্বাভাবিক"]}),
    Node("A2_not_human_caused", "Against", "cards",
         "Accepts some change but denies human/greenhouse causation; attributes it to "
         "sun, volcanoes, or natural cycles.",
         {"en": ["natural cycle", "sun spots", "earth has always changed"],
          "hi": ["यह प्राकृतिक चक्र है", "सूरज की वजह से है", "धरती हमेशा बदलती रही है"],
          "bn": ["এটা প্রাকৃতিক চক্র", "সূর্যের কারণে", "পৃথিবী চিরকাল বদলেছে"]}),
    Node("A3_impacts_not_bad", "Against", "cards",
         "Concedes warming but denies it is serious/harmful; may claim benefits "
         "(longer crops, milder winters) or that humans will adapt.",
         {"en": ["it's exaggerated", "not a big deal", "we will adapt", "good for crops"],
          "hi": ["इतना बड़ा मुद्दा नहीं है", "बहुत बढ़ा-चढ़ाकर बताया जा रहा है", "हम ढल जाएंगे"],
          "bn": ["এত বড় সমস্যা নয়", "অতিরঞ্জিত করা হচ্ছে", "আমরা মানিয়ে নেব"]}),
    Node("A4_solutions_wont_work", "Against", "cards",
         "Attacks climate policy/technology as futile, harmful or unaffordable — EV "
         "battery pollution, solar panel waste, carbon tax as a scam.",
         {"en": ["EV batteries pollute more", "solar panels are e-waste", "carbon tax scam"],
          "hi": ["EV की बैटरी और ज्यादा प्रदूषण करती है", "कार्बन टैक्स एक घोटाला है"],
          "bn": ["EV ব্যাটারি আরও দূষণ করে", "কার্বন ট্যাক্স একটা কেলেঙ্কারি"]}),
    Node("A5_science_unreliable", "Against", "cards",
         "Attacks the credibility of climate science/scientists/IPCC; recalls failed "
         "predictions, the 1970s ice-age scare, or disputes the 97% consensus figure.",
         {"en": ["their predictions always fail", "in the 70s they said ice age",
                 "paid scientists", "the 97% number is made up"],
          "hi": ["इनकी भविष्यवाणी हमेशा गलत होती है", "70 के दशक में हिमयुग बताया था",
                 "यह ९७% का आंकड़ा मनगढ़ंत है"],
          "bn": ["এদের ভবিষ্যদ্বাণী সবসময় ভুল", "৭০-এর দশকে বলেছিল বরফযুগ আসছে",
                 "এই ৯৭% সংখ্যাটা বানানো"]}),
    Node("A6_messenger_attack", "Against", "cards",
         "Dismisses the claim by attacking the presenter/communicator's credentials or "
         "motives: 'he is not a scientist, he is a mechanical engineer', 'a paid actor', "
         "'indoctrinating kids'. The claim is rejected via the messenger. (Distinguish "
         "from D5: criticising leaders for INACTION accepts the claim.)",
         {"en": ["he's a mechanical engineer, not a scientist", "paid actor",
                 "the science guy is indoctrinating children", "he has no PhD"],
          "hi": ["वह वैज्ञानिक नहीं, यांत्रिक अभियंता हैं", "पैसे लेकर बोलने वाला",
                 "बच्चों का ब्रेनवॉश कर रहे हैं"],
          "bn": ["উনি বিজ্ঞানী নন, মেকানিক্যাল ইঞ্জিনিয়ার", "টাকা খেয়ে বলছে",
                 "বাচ্চাদের মগজধোলাই করছে"]}),
]

# ---------------------------------------------------------------------------
# AGAINST — India-specific branch. This is the part the English corpora lack
# entirely, and the part that decides `Against` recall on this test set.
# ---------------------------------------------------------------------------
_INDIA_AGAINST = [
    Node("B1_climate_colonialism", "Against", "india",
         "Whataboutism / climate colonialism: the West industrialised on fossil fuels "
         "for 200 years and now lectures India; India's per-capita emissions are tiny; "
         "China emits far more. Rejects the concern as a burden unfairly placed on India.",
         {"en": ["West polluted for 200 years", "our per capita is lowest",
                 "why should India pay", "China emits more", "first world hypocrisy"],
          "hi": ["पश्चिमी देशों ने 200 साल प्रदूषण किया", "हमारा पर कैपिटा सबसे कम है",
                 "भारत क्यों भुगते", "चीन ज्यादा फैलाता है"],
          "bn": ["পশ্চিমীরা ২০০ বছর দূষণ করেছে", "আমাদের পার ক্যাপিটা সবচেয়ে কম",
                 "ভারত কেন ভুগবে", "চীন বেশি করে"]}),
    Node("B2_development_first", "Against", "india",
         "Development-first / poverty-first: jobs, roads, electricity and food matter "
         "more than climate; a poor country cannot afford green policy.",
         {"en": ["first give jobs", "roti kapda makaan first", "poor country can't afford",
                 "development is more important"],
          "hi": ["पहले रोजगार दो", "पहले रोटी कपड़ा मकान", "गरीब देश यह अफोर्ड नहीं कर सकता",
                 "विकास पहले जरूरी है"],
          "bn": ["আগে চাকরি দাও", "আগে ভাত কাপড় ঘর", "গরিব দেশ এটা পারবে না",
                 "উন্নয়ন আগে দরকার"]}),
    Node("B3_local_weather_naturalism", "Against", "india",
         "Local-weather naturalism: my city was always this hot/humid, the monsoon was "
         "always erratic, my grandparents saw the same. Generalises personal memory of "
         "local weather to deny a systemic trend.",
         {"en": ["Kolkata was always this humid", "monsoon was always unpredictable",
                 "my grandfather saw the same"],
          "hi": ["दिल्ली में गर्मी तो हमेशा रहती थी", "मानसून हमेशा अनियमित रहा है",
                 "हमारे दादाजी के समय भी ऐसा था"],
          "bn": ["কলকাতায় তো সবসময়ই এমন গরম", "বর্ষা চিরকালই অনিয়মিত",
                 "আমাদের দাদুর সময়েও এমন ছিল"]}),
    Node("B4_cyclical_religious", "Against", "india",
         "Religious or cyclical cosmology of ANY tradition: yuga cycles, pralaya is "
         "written, nature/climate is in God's hands, God decides floods and dinosaurs, "
         "humans cannot change what is ordained. Denies human agency or urgency.",
         {"en": ["it is written in scriptures", "yuga cycle", "God controls the climate",
                 "God flooded the earth before, He decides", "why did God kill the "
                 "dinosaurs then"],
          "hi": ["यह युग चक्र है", "प्रलय तो निश्चित है", "प्रकृति भगवान के हाथ में है",
                 "भगवान ने डायनासोर क्यों मारे फिर"],
          "bn": ["এটা যুগচক্র", "প্রলয় তো হবেই", "প্রকৃতি ঈশ্বরের হাতে",
                 "ঈশ্বর ডাইনোসর মেরেছিলেন কেন তাহলে"]}),
    Node("B5_media_trp_conspiracy", "Against", "india",
         "Media/TRP/funding conspiracy: channels scare people for ratings, foreign-funded "
         "NGOs and 'Greta drama' push an agenda, it is a business.",
         {"en": ["only for TRP", "foreign funded NGO agenda", "Greta drama",
                 "climate business"],
          "hi": ["सिर्फ TRP के लिए डरा रहे हैं", "विदेशी फंडिंग वाला एजेंडा",
                 "यह सब बिजनेस है"],
          "bn": ["শুধু TRP-র জন্য ভয় দেখাচ্ছে", "বিদেশি ফান্ডেড এজেন্ডা",
                 "এসব ব্যবসা"]}),
    Node("B6_elite_hypocrisy", "Against", "india",
         "Elite hypocrisy used to dismiss the concern itself: politicians fly private "
         "jets to climate summits, ACs run at the conference — therefore the issue is fake. "
         "(Distinguish from D5, which criticises leaders while accepting the claim.)",
         {"en": ["netas fly private jets", "AC on at the climate summit",
                 "they preach and pollute so it's fake"],
          "hi": ["नेता प्राइवेट जेट में घूमते हैं", "समिट में AC चलता है, सब दिखावा है"],
          "bn": ["নেতারা প্রাইভেট জেটে ঘোরে", "সামিটে AC চলে, সব দেখনদারি"]}),
    Node("B7_population_not_climate", "Against", "india",
         "Redirects to a substitute problem to dismiss climate: population, corruption, "
         "garbage or Pakistan/China is the *real* issue, not climate.",
         {"en": ["population is the real problem", "fix corruption first",
                 "garbage is the real issue"],
          "hi": ["असली समस्या जनसंख्या है", "पहले भ्रष्टाचार ठीक करो"],
          "bn": ["আসল সমস্যা জনসংখ্যা", "আগে দুর্নীতি ঠিক করুন"]}),
]

# ---------------------------------------------------------------------------
# FAVOUR — mirrored branch, grounded in Indian lived experience
# ---------------------------------------------------------------------------
_FAVOUR = [
    Node("D1_lived_experience", "Favour", "pro",
         "Cites first-hand Indian impacts as evidence the concern is serious: Chennai/"
         "Mumbai floods, Delhi AQI, Sundarbans salinity, Amphan/Yaas, unbearable heat, "
         "vanishing winter, glacial melt in Uttarakhand.",
         {"en": ["Delhi air is unbreathable", "Chennai floods every year",
                 "Sundarbans is sinking", "45 degrees in April"],
          "hi": ["दिल्ली में सांस लेना मुश्किल है", "अप्रैल में 45 डिग्री",
                 "सर्दी गायब हो गई है", "पहाड़ों में ग्लेशियर पिघल रहे हैं"],
          "bn": ["সুন্দরবন ডুবে যাচ্ছে", "এপ্রিলেই ৪৫ ডিগ্রি", "শীত আর আসেই না",
                 "আম্ফানের পর সব বদলে গেছে"]}),
    Node("D2_agriculture_livelihood", "Favour", "pro",
         "Agriculture and livelihood harm: erratic monsoon, crop failure, farmer "
         "distress, fishermen losing catch, salinity ruining farmland.",
         {"en": ["farmers are ruined by erratic rain", "no fish left in the river"],
          "hi": ["बेमौसम बारिश से किसान बर्बाद हो गए", "फसल खराब हो रही है"],
          "bn": ["অসময়ের বৃষ্টিতে চাষিরা শেষ", "নদীতে আর মাছ নেই"]}),
    Node("D3_health_impact", "Favour", "pro",
         "Health framing: smog, asthma, children's lungs, heatstroke deaths, dengue "
         "spreading with the changed climate.",
         {"en": ["my child has asthma from this smog", "heatstroke deaths every summer"],
          "hi": ["इस धुएं से बच्चों को अस्थमा हो रहा है", "हर गर्मी में लू से मौतें"],
          "bn": ["এই ধোঁয়ায় বাচ্চাদের হাঁপানি", "প্রতি গ্রীষ্মে হিটস্ট্রোকে মৃত্যু"]}),
    Node("D4_moral_intergenerational", "Favour", "pro",
         "Moral and intergenerational duty: we are leaving a ruined earth to our "
         "children; nature/dharti is our mother and we owe it care.",
         {"en": ["what world are we leaving our children", "earth is our mother"],
          "hi": ["हम अपने बच्चों को कैसी धरती दे रहे हैं", "धरती माता है"],
          "bn": ["আমরা সন্তানদের কী পৃথিবী দিয়ে যাচ্ছি", "পৃথিবী আমাদের মা"]}),
    Node("D5_demand_action", "Favour", "pro",
         "Accepts the claim and demands action, or angrily blames government/industry "
         "for inaction. Sentiment is negative but stance is Favour — a key trap.",
         {"en": ["shame on the government for doing nothing", "ban plastic now",
                 "why is no one acting"],
          "hi": ["सरकार कुछ नहीं कर रही, शर्म आनी चाहिए", "अभी प्लास्टिक बैन करो"],
          "bn": ["সরকার কিছুই করছে না, লজ্জা", "এখনই প্লাস্টিক নিষিদ্ধ করুন"]}),
    Node("D6_support_solutions", "Favour", "pro",
         "Endorses mitigation: solar mission, tree planting, EVs, public transport, "
         "cutting plastic — implying the concern is real and worth acting on.",
         {"en": ["plant more trees", "solar is the way forward", "I switched to cycling"],
          "hi": ["ज्यादा पेड़ लगाओ", "सोलर ही भविष्य है", "मैंने साइकिल चलाना शुरू किया"],
          "bn": ["আরও গাছ লাগান", "সোলারই ভবিষ্যৎ", "আমি সাইকেল চালানো শুরু করেছি"]}),
    Node("D7_trust_science", "Favour", "pro",
         "Appeals to scientific authority: the IPCC report, scientists have warned us "
         "for decades, the data is clear.",
         {"en": ["IPCC has warned us", "scientists said this 30 years ago",
                 "the data is clear"],
          "hi": ["IPCC रिपोर्ट में साफ लिखा है", "वैज्ञानिक 30 साल से चेता रहे हैं"],
          "bn": ["IPCC রিপোর্টে স্পষ্ট আছে", "বিজ্ঞানীরা ৩০ বছর ধরে সতর্ক করছেন"]}),
]

# ---------------------------------------------------------------------------
# NONE — the inventory most systems get wrong. Under macro-F1 these three
# classes weigh equally, so `None` precision matters as much as `Against` recall.
# ---------------------------------------------------------------------------
_NONE = [
    Node("N1_channel_praise", "None", "none",
         "Praise, thanks or criticism directed at the VIDEO / CHANNEL / presenter rather "
         "than at the claim: 'nice video sir', 'very informative', 'background music too "
         "loud'. Positive sentiment about the video is NOT agreement with the claim — "
         "this is the single most common false-Favour error.",
         {"en": ["nice video sir", "very informative, thank you", "please make more videos"],
          "hi": ["बहुत बढ़िया वीडियो सर", "बहुत जानकारीपूर्ण, धन्यवाद"],
          "bn": ["দারুণ ভিডিও দাদা", "খুব তথ্যপূর্ণ, ধন্যবাদ"]}),
    Node("N2_question", "None", "none",
         "A pure information-seeking question with no stance expressed: 'what is the "
         "greenhouse effect?', 'sir which book should I read?'",
         {"en": ["what is the greenhouse effect", "sir which book to read for this"],
          "hi": ["ग्रीनहाउस प्रभाव क्या है", "सर इसके लिए कौन सी किताब पढ़ें"],
          "bn": ["গ্রিনহাউস প্রভাব কী", "দাদা এর জন্য কোন বই পড়ব"]}),
    Node("N3_neutral_factual", "None", "none",
         "A neutral factual or descriptive statement with no evaluation of seriousness: "
         "'CO2 is 0.04% of the atmosphere', 'the Paris Agreement was signed in 2015'.",
         {"en": ["CO2 is 0.04 percent of the atmosphere",
                 "the Paris agreement was signed in 2015"],
          "hi": ["वायुमंडल में CO2 0.04 प्रतिशत है", "पेरिस समझौता 2015 में हुआ था"],
          "bn": ["বায়ুমণ্ডলে CO2 ০.০৪ শতাংশ", "প্যারিস চুক্তি ২০১৫ সালে হয়"]}),
    Node("N4_unrelated_politics", "None", "none",
         "Political, communal or personal abuse with no engagement with the climate "
         "claim: party slogans, attacks on a leader for unrelated reasons, trolling.",
         {"en": ["all politicians are thieves", "vote for change in 2029"],
          "hi": ["सारे नेता चोर हैं", "2029 में बदलाव लाओ"],
          "bn": ["সব নেতা চোর", "২০২৯-এ পরিবর্তন আনুন"]}),
    Node("N5_spam_greeting", "None", "none",
         "Spam, greetings, emoji-only, self-promotion, 'first comment', 'who else is "
         "watching in 2026', song lyrics.",
         {"en": ["first comment 🎉", "who is watching in 2026", "subscribe to my channel"],
          "hi": ["पहला कमेंट 🎉", "2026 में कौन देख रहा है"],
          "bn": ["প্রথম কমেন্ট 🎉", "২০২৬-এ কে দেখছে"]}),
    Node("N7_class_assignment", "None", "none",
         "The comment is here because of school/college: 'who else is here from online "
         "class', 'my teacher sent us this', 'using this for my presentation', asking "
         "for sources for homework. No stance on the claim itself.",
         {"en": ["who else is here for online class", "my teacher assigned this video",
                 "using this for my school presentation", "can I get the sources for "
                 "my research"],
          "hi": ["ऑनलाइन कक्षा के लिए और कौन आया है", "शिक्षक ने यह वीडियो दिखाया",
                 "मैं इसे अपनी प्रस्तुति के लिए इस्तेमाल कर रहा हूँ"],
          "bn": ["অনলাইন ক্লাসের জন্য আর কে এসেছে", "শিক্ষক এই ভিডিওটা দেখালেন",
                 "আমি এটা আমার প্রেজেন্টেশনের জন্য ব্যবহার করছি"]}),
    Node("N6_ambivalent", "None", "none",
         "Genuinely two-sided or too vague to resolve: acknowledges the issue but also "
         "dismisses it, or is so short/ambiguous that no stance can be assigned.",
         {"en": ["true but also not true", "hmm difficult topic", "both sides have a point"],
          "hi": ["सही भी है और गलत भी", "मुश्किल विषय है"],
          "bn": ["ঠিকও আবার ভুলও", "কঠিন বিষয়"]}),
]

# ---------------------------------------------------------------------------
NODES: list[Node] = _CARDS + _INDIA_AGAINST + _FAVOUR + _NONE
NODE_IDS: list[str] = [n.id for n in NODES]
NODE_BY_ID: dict[str, Node] = {n.id: n for n in NODES}
NODE2IDX: dict[str, int] = {n.id: i for i, n in enumerate(NODES)}
NODE2STANCE: dict[str, str] = {n.id: n.stance for n in NODES}

# index reserved for "no node assigned" (real data before the committee runs)
UNKNOWN_NODE = "UNK"
NODE2IDX[UNKNOWN_NODE] = len(NODES)
N_NODES = len(NODES) + 1


def nodes_for_stance(stance: str) -> list[Node]:
    return [n for n in NODES if n.stance == stance]


def stance_of_node(node_id: str, default: str = "None") -> str:
    """Deterministic argument -> stance mapping (the 'pivot')."""
    return NODE2STANCE.get(str(node_id).strip(), default)


def taxonomy_prompt_block(stances=("Favour", "Against", "None"),
                          with_cues: bool = False, lang: str = "en") -> str:
    """Render the taxonomy as a numbered menu for an LLM prompt."""
    lines = []
    for stance in stances:
        lines.append(f"\n[{stance}]")
        for n in nodes_for_stance(stance):
            lines.append(f"  {n.id}: {n.gloss}")
            if with_cues and n.cues.get(lang):
                ex = "; ".join(n.cues[lang][:3])
                lines.append(f"      e.g. {ex}")
    return "\n".join(lines)


ANNOTATION_GUIDELINES = f"""\
TARGET CLAIM: "Climate change and global warming is a serious concern."

Assign the stance of the comment TOWARDS THAT CLAIM — not its sentiment, not its
politeness, and not its opinion of the video.

  Favour  — the comment agrees that climate change/global warming is a serious concern
            (including angrily demanding action, or blaming leaders for inaction).
  Against — the comment disagrees: denies it is happening, denies it is human-caused,
            denies it is serious, calls it exaggerated/a scam/an agenda, or dismisses
            it as someone else's problem.
  None    — the comment takes no resolvable stance on the claim: praises or criticises
            the video, asks a question, states a neutral fact, is unrelated abuse or
            spam, or is genuinely two-sided.

Four rules that resolve most hard cases:

 1. SENTIMENT IS NOT STANCE. "Shame on this government, they have destroyed our
    rivers!" is *Favour* — furious, but it accepts the claim.
 2. PRAISE OF THE VIDEO IS NOT AGREEMENT. "Very informative video sir, thank you"
    is *None*, even under a climate-alarm video.
 3. HYPOCRISY CUTS BOTH WAYS. "Leaders fly private jets, so this whole climate thing
    is a drama" is *Against* (dismisses the claim). "Leaders fly private jets while
    lecturing us — they must act first" is *Favour* (accepts the claim, attacks the
    messenger).
 4. WHATABOUTISM IS AGAINST when it is used to reject the concern for India ("the West
    polluted for 200 years, why should we suffer") — the comment is refusing the claim's
    force, not merely adding context.

ARGUMENT MENU (choose exactly one node id):
{taxonomy_prompt_block()}
"""


if __name__ == "__main__":
    from collections import Counter
    print(f"{len(NODES)} argument nodes (+1 UNK) => auxiliary head size {N_NODES}")
    print("by stance:", Counter(n.stance for n in NODES))
    print("by branch:", Counter(n.branch for n in NODES))
    n_cues = sum(len(v) for n in NODES for v in n.cues.values())
    print(f"{n_cues} cue phrases across en/hi/bn")
    print(ANNOTATION_GUIDELINES)
