import stanza

class AspectOpinionExtractor:
    def __init__(self):
        self.nlp = stanza.Pipeline('id')

    def extract_aspects_and_opinions(self, text):
        doc = self.nlp(text)
        aspect_opinions = []
        for sentence in doc.sentences:
            words = sentence.words
            i = 0
            while i < len(words):
                word = words[i]
                if word.upos in ['NOUN', 'PRON'] and word.text.lower() != 'sayang':  # Exclude 'sayang'
                    # Check if the noun is part of a compound noun
                    compound_noun = word.text
                    if i < len(words) - 1 and words[i + 1].upos == 'NOUN':
                        compound_noun += ' ' + words[i + 1].text
                        i += 1
                    # Check adjectives in noun phrase
                    for potential_child in sentence.words:
                        if potential_child.head == int(word.id) or (i < len(words) - 1 and potential_child.head == int(words[i + 1].id)):
                            if potential_child.upos == 'ADJ':  # Only consider adjectives as opinions
                                # Check if the adjective has a negation
                                for potential_modifier in sentence.words:
                                    if potential_modifier.head == int(potential_child.id) and potential_modifier.text.lower() == 'tidak':
                                        aspect_opinions.append((compound_noun, 'tidak ' + potential_child.text))
                                        break
                                else:
                                    aspect_opinions.append((compound_noun, potential_child.text))
                i += 1
        aspects = [ao[0] for ao in aspect_opinions]
        opinions = [ao[1] for ao in aspect_opinions]
        return aspects, opinions