# coding: utf-8
for sent in sentences:
    doc = nlp(sent)
    for tok in doc:
        if tok.pos_ == "NOUN" and not tok.text[0].isupper() and tok.dep_ == "pobj":
            head = tok.head
            if head.text == "of":
                part = head.head
                if not part.text[0].isupper() and part.pos_ == "NOUN":
                    print(tok.text, part.text)
                            
