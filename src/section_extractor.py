"""
Module 2: Section Extractor

Local, rule-based extraction of the main sections of a research paper.

How it works
------------
1. Each line is checked to see whether it looks like a heading:
   * numbered headings: "1 INTRODUCTION", "2.1 Method", "IV. RESULTS",
     including PDFs where the number sits on its own line ("3" / "SYSTEM
     OVERVIEW").
   * unnumbered headings are accepted only if they match a known section
     name (to avoid treating figure labels as headings).
2. A heading that matches a known section starts that section.
3. A heading that does NOT match a known section (e.g. "Data Acquisition",
   "References") ends the current section, so unrelated content is not
   attached to the previous section.
4. Unknown sub-headings ("5.2 Training details") stay inside their parent
   section ("5 Our Method" -> methodology).
5. If no explicit heading exists for contributions / objectives / problem
   statement, short passages are inferred from the abstract/introduction
   using common phrases and are labelled as inferred.
"""

import re


class SectionExtractor:

    # Patterns are matched against the normalised heading title using
    # re.fullmatch (lowercase, "&" -> "and", punctuation removed).
    # Order matters: more specific sections are checked first.
    SECTION_PATTERNS = {
        "abstract": [
            r"abstract"
        ],

        "problem_statement": [
            r"problem statement",
            r"(the )?research problem",
            r"problem (definition|formulation|description)",
            r"statement of (the )?problem"
        ],

        "objectives": [
            r"objectives?",
            r"research objectives?",
            r"aims? and objectives?",
            r"research (aims?|questions?)",
            r"goals?( of (the|this) (study|work|paper))?"
        ],

        "contributions": [
            r"contributions?",
            r"(main|our|key|summary of) contributions?"
        ],

        "literature_review": [
            r"literature review",
            r"review of (the )?literature",
            r"related works?",
            r"related (studies|research|literature)",
            r"prior work",
            r"previous work",
            r"background and related works?",
            r"related works? and background",
            r"state of the art"
        ],

        "introduction": [
            r"introduction",
            r"background",
            r"motivation",
            r"introduction and motivation"
        ],

        "methodology": [
            r"methodology",
            r"(research|proposed|our) methodology",
            r"(the )?(proposed|our) (method|methods|approach|framework|model|system|technique)",
            r"materials and methods",
            r"methods?",
            r"(methods|methodology) and materials",
            r"approach",
            r"(system|model|proposed) architecture",
            r"system (overview|design|model)",
            r"framework( overview)?",
            r"method overview"
        ],

        "results": [
            r"results",
            r"(experimental|evaluation|simulation) results",
            r"results and (discussions?|analysis|evaluation)",
            r"(experimental )?evaluation",
            r"experiments?",
            r"experimental (setup and )?(results|evaluation|analysis)",
            r"experiments and results",
            r"performance evaluation",
            r"findings"
        ],

        "conclusion": [
            r"conclusions?",
            r"conclusions? and future (work|scope|directions?)",
            r"concluding remarks",
            r"summary and conclusions?",
            r"future work",
            r"future scope"
        ]
    }

    # Headings that end any section without starting a new one.
    STOP_PATTERNS = [
        # Plural only: a singular "Reference" is usually a figure legend.
        r"references",
        r"bibliography",
        r"acknowledge?ments?",
        r"appendix( [a-z0-9]+)?",
        r"appendices",
        r"supplementary materials?",
        r"keywords?",
        r"index terms",
        r"ccs concepts",
        r"author contributions?",
        r"funding",
        r"declarations?",
        r"conflicts? of interests?",
        r"competing interests?",
        r"data availability( statement)?",
        r"ethics statement"
    ]

    # "Abstract— text", "Abstract: text", "Abstract. text"
    INLINE_ABSTRACT = re.compile(
        r"^abstract\s*[\u2014\u2013\-:.]\s*(?P<rest>.+)$", re.IGNORECASE
    )
    # "Index Terms— ...", "Keywords: ..." lines end the abstract.
    INLINE_STOP = re.compile(
        r"^(index terms|keywords?|key words|ccs concepts)\s*[\u2014\u2013\-:.]",
        re.IGNORECASE
    )

    NUMBER_ONLY = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3}|[IVX]{1,5})\.?$")
    ARABIC_HEADING = re.compile(
        r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\)?\s+(?P<title>\S.*)$"
    )
    ROMAN_HEADING = re.compile(
        r"^(?P<num>[IVX]{1,5})[.)]\s+(?P<title>\S.*)$"
    )
    LETTER_HEADING = re.compile(r"^(?P<num>[A-H])[.)]\s+(?P<title>\S.*)$")

    ROMAN_VALUES = {"I": 1, "V": 5, "X": 10}

    # Phrases used to infer sections when no explicit heading exists.
    INFERENCE_CUES = {
        "contributions": re.compile(
            r"(our|main|key|major|primary) contributions?\b|"
            r"we make the following contributions|"
            r"contributions of (this|our) (paper|work|study)|"
            r"in summary,? (we|this paper)|to summari[sz]e,? (we|our)",
            re.IGNORECASE
        ),
        "objectives": re.compile(
            r"(the )?(aim|objective|goal|purpose)s? of (this|the present|our) "
            r"(paper|study|work|research)|"
            r"this (paper|study|work) aims to|we aim to",
            re.IGNORECASE
        ),
        "problem_statement": re.compile(
            r"\bhowever\b|\bchalleng|\bremains? (an )?(open|unsolved|difficult)|"
            r"\blimitations? of\b|\black of\b|\bproblem of\b|\bstruggle",
            re.IGNORECASE
        )
    }

    def __init__(self):
        self._compiled = {
            section: [re.compile(p) for p in patterns]
            for section, patterns in self.SECTION_PATTERNS.items()
        }
        self._stop = [re.compile(p) for p in self.STOP_PATTERNS]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_sections(self, full_text):
        """Return {section_name: text} (empty string when not found)."""
        return self.extract(full_text)["sections"]

    def extract(self, full_text):
        """
        Return a dict with:
            sections:  {section_name: text}
            headings:  {section_name: [original headings or inference note]}
            detected_headings: list of all detected headings in order
        """
        sections = {section: [] for section in self.SECTION_PATTERNS}
        headings = {section: [] for section in self.SECTION_PATTERNS}
        detected = []

        if not full_text or not full_text.strip():
            return {
                "sections": {s: "" for s in sections},
                "headings": headings,
                "detected_headings": detected
            }

        lines = [line.strip() for line in full_text.splitlines()]
        lines = [line for line in lines if line]

        current = None        # section receiving text
        parent = None         # section of the last top-level heading
        last_top = 0          # last accepted top-level section number

        i = 0
        while i < len(lines):
            line = lines[i]

            # --- Inline abstract / keyword lines ---------------------------
            inline = self.INLINE_ABSTRACT.match(line)
            if inline and not headings["abstract"]:
                current = parent = "abstract"
                headings["abstract"].append("Abstract")
                detected.append({"heading": "Abstract", "section": "abstract"})
                sections["abstract"].append(inline.group("rest"))
                i += 1
                continue

            if self.INLINE_STOP.match(line):
                detected.append({"heading": line[:60], "section": None})
                current = parent = None
                i += 1
                continue

            # --- Number on its own line followed by a title -----------------
            consumed = 1
            candidate = line
            if self.NUMBER_ONLY.match(line):
                if i + 1 < len(lines):
                    candidate = f"{line.rstrip('.')} {lines[i + 1]}"
                    consumed = 2
                else:
                    i += 1
                    continue

            heading = self._parse_heading(candidate, last_top)

            if heading is None and consumed == 2:
                # A lone number that is not part of a heading is almost
                # always a page number or figure label: drop it.
                i += 1
                continue

            if heading is not None:
                section, level, number = heading
                detected.append({"heading": candidate, "section": section})

                if level == 1:
                    if number is not None:
                        last_top = number
                    parent = section
                    current = section
                else:
                    # Known sub-heading starts its own section; unknown
                    # sub-heading stays inside the parent section.
                    current = section if section else parent

                if section:
                    headings[section].append(candidate)
                i += consumed
                continue

            if current:
                sections[current].append(line)
            i += 1

        result = {
            section: self._join_lines(content)
            for section, content in sections.items()
        }

        self._infer_missing(result, headings)

        return {
            "sections": result,
            "headings": headings,
            "detected_headings": detected
        }

    # ------------------------------------------------------------------
    # Heading detection helpers
    # ------------------------------------------------------------------
    def _parse_heading(self, line, last_top):
        """
        Return (section_or_None, level, top_number_or_None) when `line`
        is a heading, otherwise None.
        """
        if len(line) > 90:
            return None

        number = None
        level = 1
        title = line
        numbered = False

        match = self.ARABIC_HEADING.match(line)
        roman = self.ROMAN_HEADING.match(line)
        letter = self.LETTER_HEADING.match(line)

        if match:
            parts = match.group("num").split(".")
            number = int(parts[0])
            level = len(parts)
            title = match.group("title")
            numbered = True
        elif roman and self._roman_to_int(roman.group("num")):
            number = self._roman_to_int(roman.group("num"))
            title = roman.group("title")
            numbered = True
        elif letter:
            level = 2
            title = letter.group("title")
            numbered = True

        if not self._looks_like_title(title):
            return None

        section, is_stop = self._match_known(title)

        if section or is_stop:
            # Known headings are accepted with or without numbering.
            if numbered and level > 1 and is_stop:
                return None
            return (section, level, number if level == 1 else None)

        if not numbered:
            return None

        # Unknown numbered heading: accept only if numbering is consistent,
        # which filters out list items, figure labels and page numbers.
        if letter:
            return None
        if level == 1 and number == last_top + 1:
            # Unknown top-level headings must look like headings
            # (Title Case or UPPER CASE).
            if self._is_heading_case(title):
                return (None, 1, number)
            return None
        if level > 1 and number == last_top and self._is_heading_case(title):
            return (None, level, None)

        return None

    def _match_known(self, title):
        """Return (section, is_stop) for a heading title."""
        normalised = self._normalise(title)
        candidates = [normalised]

        # "Evaluation: User Study" -> also try "evaluation"
        if ":" in title:
            candidates.append(self._normalise(title.split(":", 1)[0]))

        for candidate in candidates:
            for pattern in self._stop:
                if pattern.fullmatch(candidate):
                    return None, True
            for section, patterns in self._compiled.items():
                for pattern in patterns:
                    if pattern.fullmatch(candidate):
                        return section, False
        return None, False

    @staticmethod
    def _normalise(text):
        text = text.lower().replace("&", " and ")
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _looks_like_title(title):
        title = title.strip()
        if not title or len(title) < 3 or len(title) > 80:
            return False
        if len(title.split()) > 10:
            return False
        if title[-1] in ".,;":
            return False
        if not title[0].isalpha() or not title[0].isupper():
            return False
        letters = sum(ch.isalpha() for ch in title)
        return letters / len(title) > 0.6

    @staticmethod
    def _is_heading_case(title):
        """UPPER CASE or Title Case (small words may stay lowercase)."""
        if title.isupper():
            return True
        small = {"a", "an", "and", "the", "of", "for", "in", "on", "to",
                 "with", "via", "by", "or", "vs", "from", "at"}
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", title)
        if not words:
            return False
        significant = [w for w in words if w.lower() not in small]
        capitalised = sum(w[0].isupper() for w in significant)
        # Sentence-case headings ("Training details") are common too, so
        # accept short headings whose first word is capitalised.
        return capitalised == len(significant) or len(words) <= 4

    def _roman_to_int(self, numeral):
        total, previous = 0, 0
        for char in reversed(numeral):
            value = self.ROMAN_VALUES.get(char)
            if value is None:
                return None
            total = total - value if value < previous else total + value
            previous = max(previous, value)
        return total if 0 < total <= 20 else None

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _join_lines(lines):
        """Join lines into a paragraph, repairing hyphenated line breaks."""
        text = ""
        for line in lines:
            if not text:
                text = line
            elif text.endswith("-") and line[:1].islower():
                text = text[:-1] + line
            else:
                text = f"{text} {line}"
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _split_sentences(text):
        return [
            s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
            if s.strip()
        ]

    def _infer_missing(self, sections, headings, max_chars=1200):
        """Infer short passages for sections without an explicit heading."""
        source = " ".join(
            filter(None, [sections["abstract"], sections["introduction"]])
        )
        if not source:
            return

        sentences = self._split_sentences(source)

        for section in ("contributions", "objectives"):
            if sections[section]:
                continue
            cue = self.INFERENCE_CUES[section]
            for index, sentence in enumerate(sentences):
                if cue.search(sentence):
                    passage = ""
                    for following in sentences[index:]:
                        if len(passage) + len(following) > max_chars:
                            break
                        passage = f"{passage} {following}".strip()
                    sections[section] = passage or sentence[:max_chars]
                    headings[section].append(
                        "(inferred from abstract/introduction - "
                        "no explicit heading)"
                    )
                    break

        if not sections["problem_statement"]:
            cue = self.INFERENCE_CUES["problem_statement"]
            abstract_sentences = self._split_sentences(
                sections["abstract"] or sections["introduction"]
            )
            picked = [s for s in abstract_sentences if cue.search(s)][:3]
            if picked:
                sections["problem_statement"] = " ".join(picked)[:max_chars]
                headings["problem_statement"].append(
                    "(inferred from abstract/introduction - "
                    "no explicit heading)"
                )
