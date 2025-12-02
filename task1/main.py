from dataclasses import dataclass
from typing import Optional, List, Set
import gzip

import razdel
import pymorphy2

from yargy import rule, or_, Parser
from yargy.interpretation import fact
from yargy.predicates import gram, eq, in_, custom , in_caseless, dictionary, type as type_
from yargy.pipelines import morph_pipeline,pipeline

morph = pymorphy2.MorphAnalyzer()


def is_russian_fio_part(word: str, gram_code: str) -> bool:
    for p in morph.parse(word):
        if gram_code in p.tag.grammemes:
            return True
    return False


def normalize_fio_part(word: Optional[str]) -> Optional[str]:
    if not word:
        return None

    for p in morph.parse(word):
        if {'Name', 'Surn', 'Patr'} & p.tag.grammemes:
            return p.normal_form

    return word


def normalize_place(place: Optional[str]) -> Optional[str]:
    if not place:
        return None
    return morph.parse(place)[0].normal_form

@dataclass
class Entry:
    name: str
    birth_date: Optional[str]
    birth_place: Optional[str]

NAME = fact('Name', ['first', 'last', 'middle'])

FIRST_PRED = custom(
    lambda v: (
        len(v) >= 3 and
        is_russian_fio_part(v, "Name") and
        not is_russian_fio_part(v, "Surn")
    )
)

LAST_PRED = custom(
    lambda v: (
        len(v) >= 3 and
        is_russian_fio_part(v, "Surn")
    )
)

MIDDLE_PRED = gram("Patr")

FIRST = rule(FIRST_PRED).interpretation(NAME.first)
LAST = rule(LAST_PRED).interpretation(NAME.last)
MIDDLE = rule(MIDDLE_PRED).interpretation(NAME.middle)

FULL_NAME = rule(
    FIRST,
    LAST,
    MIDDLE.optional()
).interpretation(NAME)

BIRTH = fact('Birth', ['place', 'year'])

YEAR4 = custom(lambda v: v.isdigit() and len(v) == 4)

DATE_RULE = rule(
    YEAR4
).interpretation(BIRTH.year)

TOWN_ABBR = in_caseless({'г', 'город', 'с', 'село', 'д', 'деревня', 'пос', 'посёлок'})
REGION = morph_pipeline(['область', 'край', 'республика', 'район', 'округ'])

PLACE = rule(
    TOWN_ABBR.optional(),
    or_(
        gram('Geox'),
        gram('Name'),
        gram('Surn'),
        gram('Patr')
    ),
    REGION.optional()
).interpretation(BIRTH.place)

BIRTH_WORDS = or_(
    rule(in_caseless({'родился', 'родилась'})),
    rule(in_caseless({'уроженец', 'уроженка'})),
    rule(in_caseless('родом'), eq('из')),
    rule(in_caseless('место'), in_caseless('рождения'), eq(':').optional()),
    rule(in_caseless('дата'), in_caseless('рождения'), eq(':').optional())
)

BIRTH_COMPLEX_1 = rule(
    BIRTH_WORDS,
    eq('в').optional(),
    DATE_RULE.optional(),
    eq('в').optional(),
    PLACE.optional()
)

BIRTH_COMPLEX_2 = rule(
    BIRTH_WORDS,
    eq('в').optional(),
    PLACE,
    DATE_RULE.optional()
)

BIRTH_BY_LABEL_DATE = rule(
    in_caseless('дата'),
    in_caseless('рождения'),
    eq(':').optional(),
    DATE_RULE
)

BIRTH_BY_LABEL_PLACE = rule(
    in_caseless('место'),
    in_caseless('рождения'),
    eq(':').optional(),
    PLACE
)

BIRTH_FULL_RULE = or_(
    BIRTH_COMPLEX_1,
    BIRTH_COMPLEX_2,
    BIRTH_BY_LABEL_DATE,
    BIRTH_BY_LABEL_PLACE
).interpretation(BIRTH)

birth_parser = Parser(BIRTH_FULL_RULE)
name_parser = Parser(FULL_NAME)

def extract_from_text(text: str) -> List[Entry]:
    entries: List[Entry] = []
    seen: Set[str] = set()

    for sent in razdel.sentenize(text):
        s = sent.text

        persons = list(name_parser.findall(s))
        births = list(birth_parser.findall(s))

        birth_place = None
        birth_year = None

        if births:
            bf = births[0].fact
            if bf.place:
                birth_place = normalize_place(bf.place)
            if bf.year:
                birth_year = bf.year

        if not persons:
            continue

        for p in persons:
            pf = p.fact

            first = normalize_fio_part(pf.first)
            last = normalize_fio_part(pf.last)
            middle = normalize_fio_part(pf.middle)

            full_name = " ".join(x for x in [first, last, middle] if x)

            if full_name not in seen:
                seen.add(full_name)
                entries.append(Entry(
                    name=full_name,
                    birth_date=birth_year,
                    birth_place=birth_place
                ))

    return entries

def main():
    all_entries: List[Entry] = []

    with gzip.open("news.txt.gz", "rt", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue

            _, _, text = parts
            all_entries.extend(extract_from_text(text))

    for e in all_entries:
        print(e)


if __name__ == "__main__":
    main()