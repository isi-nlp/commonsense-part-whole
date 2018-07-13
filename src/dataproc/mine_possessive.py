# coding: utf-8
for sent in sentences:
    doc = nlp(sent)
    for tok in doc:
        if tok.text == "'s" and tok.dep_ == "case":
            head = tok.head
            if head.pos_ == "NOUN" and head.dep_ == "poss" and not head.text[0].isupper():
                whole = head.text
                part = head.head
                if not part.text[0].isupper() and part.pos_ == "NOUN":
                    print(whole, part.text)
                            
