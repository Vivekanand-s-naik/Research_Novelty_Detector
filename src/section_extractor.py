
import re


class SectionExtractor:

    SECTION_PATTERNS = {
        "abstract": [
            r"abstract"
        ],

        "introduction": [
            r"introduction",
            r"background",
            r"motivation"
        ],

        "problem_statement": [
            r"problem statement",
            r"research problem",
            r"problem definition"
        ],

        "objectives": [
            r"objectives",
            r"research objectives",
            r"aim and objectives",
            r"research aims"
        ],

        "literature_review": [
            r"literature review",
            r"related work",
            r"related studies",
            r"background and related work"
        ],

        "methodology": [
            r"methodology",
            r"proposed methodology",
            r"proposed method",
            r"materials and methods",
            r"methods",
            r"approach",
            r"system architecture",
            r"model architecture"
        ],

        "results": [
            r"results",
            r"experimental results",
            r"evaluation",
            r"experiments",
            r"results and discussion"
        ],

        "contributions": [
            r"contributions",
            r"main contributions",
            r"our contributions"
        ],

        "conclusion": [
            r"conclusion",
            r"conclusions",
            r"conclusion and future work",
            r"future work",
            r"future scope"
        ]
    }

    def extract_sections(self, full_text):

        sections = {
            section: ""
            for section in self.SECTION_PATTERNS
        }

        # Match standalone section headings
        heading_map = {}

        for section, patterns in self.SECTION_PATTERNS.items():

            for pattern in patterns:

                heading_map[pattern] = section

        lines = full_text.splitlines()

        current_section = None
        extracted = {
            section: []
            for section in sections
        }

        for line in lines:

            cleaned = line.strip()

            if not cleaned:
                continue

            # Remove common numbering:
            # 1. INTRODUCTION
            # 2.1 METHODOLOGY
            heading = re.sub(
                r"^\d+(?:\.\d+)*[\s.)-]+",
                "",
                cleaned
            )

            heading = heading.strip(" :.-").lower()

            matched_section = None

            for pattern, section in heading_map.items():

                if re.fullmatch(pattern, heading, re.IGNORECASE):
                    matched_section = section
                    break

            if matched_section:

                current_section = matched_section
                continue

            if current_section:
                extracted[current_section].append(cleaned)

        # Combine collected lines
        for section, content in extracted.items():

            sections[section] = " ".join(content).strip()

        return sections
