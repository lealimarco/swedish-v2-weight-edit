"""Probe sets for Swedish V2 and English SVO word order.

Each item is (prompt, correct_completion, incorrect_completion).
Swedish is V2: after a fronted adverbial the finite verb comes second.
English is SVO: the subject comes first.
"""

SWEDISH_V2 = [
    ("Igår",        " gick jag till affären.",     " jag gick till affären."),
    ("Idag",        " äter jag frukost.",          " jag äter frukost."),
    ("Imorgon",     " reser vi till Stockholm.",   " vi reser till Stockholm."),
    ("På morgonen", " dricker hon kaffe.",         " hon dricker kaffe."),
    ("I går",       " spelade han piano.",         " han spelade piano."),
    ("Nästa vecka", " börjar skolan.",             " skolan börjar."),
    ("Nu",          " kommer bussen.",             " bussen kommer."),
    ("Sedan",       " gick vi hem.",               " vi gick hem."),
    ("På kvällen",  " läser jag en bok.",          " jag läser en bok."),
    ("I somras",    " reste de till Italien.",     " de reste till Italien."),
    ("Snart",       " kommer tåget.",              " tåget kommer."),
    ("Förra året",  " bodde vi i Göteborg.",       " vi bodde i Göteborg."),
    ("På natten",   " sover barnen.",              " barnen sover."),
    ("Efter lunch", " arbetar hon.",               " hon arbetar."),
    ("I fredags",   " träffade vi Marco.",         " vi träffade Marco."),
    ("Om en timme", " går vi.",                    " vi går."),
    ("På semestern"," simmade han.",               " han simmade."),
    ("I vintras",   " åkte vi skidor.",            " vi åkte skidor."),
    ("Just nu",     " regnar det.",                " det regnar."),
    ("Senare",      " ringer jag dig.",            " jag ringer dig."),
    ("Igår kväll",  " lyssnade vi på musik.",      " vi lyssnade på musik."),
    ("Nästa månad", " flyttar hon.",               " hon flyttar."),
    ("I torsdags",  " åt vi pizza.",               " vi åt pizza."),
    ("På hösten",   " regnar det mycket.",         " det regnar mycket."),
    ("I januari",   " börjar kursen.",             " kursen börjar."),
    ("Imorgon bitti"," åker vi tidigt.",           " vi åker tidigt."),
    ("På söndag",   " spelar de fotboll.",         " de spelar fotboll."),
    ("För två år sedan", " bodde han i Malmö.",    " han bodde i Malmö."),
    ("I helgen",    " träffar vi vänner.",         " vi träffar vänner."),
    ("Nyligen",     " köpte jag en gitarr.",       " jag köpte en gitarr."),
]

ENGLISH_SVO = [
    ("Yesterday",        " I went to the store.",       " went I to the store."),
    ("Today",            " she eats breakfast.",        " eats she breakfast."),
    ("Tomorrow",         " we travel to Stockholm.",    " travel we to Stockholm."),
    ("In the morning",   " he drinks coffee.",          " drinks he coffee."),
    ("Last year",        " we lived in Gothenburg.",    " lived we in Gothenburg."),
    ("Now",              " the bus is coming.",         " is the bus coming."),
    ("Then",             " we went home.",              " went we home."),
    ("In the evening",   " I read a book.",             " read I a book."),
    ("Last summer",      " they travelled to Italy.",   " travelled they to Italy."),
    ("Soon",             " the train arrives.",         " arrives the train."),
    ("Last Friday",      " we met Marco.",              " met we Marco."),
    ("In an hour",       " we leave.",                  " leave we."),
    ("On holiday",       " he swam.",                   " swam he."),
    ("Last winter",      " we skied.",                  " skied we."),
    ("Right now",        " it is raining.",             " is it raining."),
    ("Later",            " I will call you.",           " will I call you."),
    ("Last night",       " we listened to music.",      " listened we to music."),
    ("Next month",       " she moves.",                 " moves she."),
    ("On Thursday",      " we ate pizza.",              " ate we pizza."),
    ("In autumn",        " it rains a lot.",            " rains it a lot."),
    ("In January",       " the course begins.",         " begins the course."),
    ("Tomorrow morning", " we leave early.",            " leave we early."),
    ("On Sunday",        " they play football.",        " play they football."),
    ("Two years ago",    " he lived in Malmö.",         " lived he in Malmö."),
    ("This weekend",     " we meet friends.",           " meet we friends."),
    ("Recently",         " I bought a guitar.",         " bought I a guitar."),
    ("Yesterday evening"," we listened to music.",      " listened we to music."),
    ("Next year",        " she will study.",            " will she study."),
    ("Last Monday",      " they arrived late.",         " arrived they late."),
    ("In spring",        " the flowers bloom.",         " bloom the flowers."),
]

# Split into 60% development (for extracting the edit) and 40% held-out (for evaluation).
def split_probes(probes, dev_frac=0.6):
    n_dev = int(len(probes) * dev_frac)
    return probes[:n_dev], probes[n_dev:]